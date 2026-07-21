"""Spectro core engine — multi-model query and spectrum analysis.

The Spectrograph sends a prompt to multiple models in parallel,
collects responses, and produces a SpectrumResult containing
convergences, divergences, and unique insights.

Design principles:
- Models are perspectives, not quality levels.
- Convergences = high confidence. Divergences = exploration targets.
- The ensemble is the experiment.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

import httpx

from spectro.exceptions import (
    APIKeyMissing,
    AnalysisTimeout,
    ModelUnavailable,
    ResponseMalformed,
)

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults — the "repertory company" from the casting experiments.
# These are documented in TOOLS.md and the casting-call repo.
# ---------------------------------------------------------------------------

DEFAULT_MODELS: list[str] = [
    "deepseek-ai/DeepSeek-V4-Flash",      # cheap workhorse, 8/10
    "ByteDance/Seed-2.0-mini",             # thin chart, ideation
    "deepreinforce-ai/Ornith-1.0-35B",     # best fiction, punches above weight
    "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B",  # structural, sergeant
    "ByteDance/Seed-2.0-pro",              # 9/10 best overall, lyrical
]

DEFAULT_BASE_URL: str = "https://api.deepinfra.com/v1/openai"
DEFAULT_TIMEOUT: float = 120.0
DEFAULT_MAX_TOKENS: int = 2048
DEFAULT_TEMPERATURE: float = 0.7

# Retry constants
MAX_RETRIES: int = 3
RETRYABLE_STATUSES: frozenset[int] = frozenset({429, 500, 502, 503, 504})
RETRY_BASE_DELAY: float = 1.0
RETRY_MAX_DELAY: float = 10.0


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ModelResponse:
    """A single model's response within the spectrum."""

    model: str
    content: str
    elapsed_ms: float
    error: str | None = None
    tokens: int = 0

    @property
    def ok(self) -> bool:
        return self.error is None and bool(self.content.strip())


@dataclass
class SpectrumResult:
    """The full spectral analysis across all models."""

    prompt: str
    responses: list[ModelResponse]
    convergences: list[dict[str, Any]] = field(default_factory=list)
    divergences: list[dict[str, Any]] = field(default_factory=list)
    unique_insights: list[dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0
    elapsed_ms: float = 0.0

    @property
    def n_models(self) -> int:
        return len(self.responses)

    @property
    def n_ok(self) -> int:
        return sum(1 for r in self.responses if r.ok)

    def summary(self) -> str:
        """One-line summary of the spectrum."""
        lines: list[str] = [
            f"Spectrum: {self.n_ok}/{self.n_models} models responded",
            f"Confidence: {self.confidence:.0%} agreement",
            f"Convergences: {len(self.convergences)}",
            f"Divergences: {len(self.divergences)}",
            f"Unique insights: {len(self.unique_insights)}",
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Serialize result to a JSON-safe dictionary."""
        return {
            "prompt": self.prompt,
            "confidence": self.confidence,
            "elapsed_ms": self.elapsed_ms,
            "n_models": self.n_models,
            "n_ok": self.n_ok,
            "convergences": self.convergences,
            "divergences": self.divergences,
            "unique_insights": self.unique_insights,
            "responses": [
                {
                    "model": r.model,
                    "content": (
                        r.content[:500] + "..."
                        if len(r.content) > 500
                        else r.content
                    ),
                    "elapsed_ms": r.elapsed_ms,
                    "tokens": r.tokens,
                    "error": r.error,
                }
                for r in self.responses
            ],
        }


# ---------------------------------------------------------------------------
# Spectrograph — main engine
# ---------------------------------------------------------------------------


class Spectrograph:
    """The multi-model cognitive spectrograph.

    Sends prompts to multiple models in parallel and analyzes the
    convergences and divergences in their responses.

    Usage:
        spec = Spectrograph()  # reads DEEPINFRA_API_KEY
        result = spec.analyze("What makes code maintainable?")
        print(result.summary())

    Supports context manager usage:
        async with Spectrograph() as spec:
            result = await spec.analyze_async("...")
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        models: list[str] | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        self.api_key: str = self._resolve_api_key(api_key)
        self.base_url: str = base_url.rstrip("/")
        self.models: list[str] = models or list(DEFAULT_MODELS)
        self.max_tokens: int = max_tokens
        self.temperature: float = temperature
        self.timeout: float = timeout
        self._client: httpx.AsyncClient | None = None

        logger.debug(
            "Spectrograph initialized: %d models, base=%s, timeout=%.1fs",
            len(self.models),
            self.base_url,
            self.timeout,
        )

    # ------------------------------------------------------------------
    # API key resolution
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_api_key(api_key: str | None) -> str:
        """Resolve the API key from explicit value, env vars, or key file.

        Raises APIKeyMissing if no key is found.
        """
        key = (
            api_key
            or os.environ.get("DEEPINFRA_API_KEY")
            or os.environ.get("DEEPINFRA_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or Spectrograph._read_key_file()
        )
        if not key:
            raise APIKeyMissing(
                "No API key found. Set DEEPINFRA_API_KEY, OPENAI_API_KEY, "
                "or create ~/.openclaw/.deepinfra-key"
            )
        return key

    @staticmethod
    def _read_key_file() -> str:
        """Read API key from the known DeepInfra key file."""
        for path in (
            os.path.expanduser("~/.openclaw/.deepinfra-key"),
            os.path.expanduser("~/.deepinfra-key"),
        ):
            try:
                with open(path) as f:
                    return f.read().strip()
            except OSError:
                continue
        return ""

    # ------------------------------------------------------------------
    # Client lifecycle
    # ------------------------------------------------------------------

    def _get_client(self) -> httpx.AsyncClient:
        """Get or create the shared HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    def close(self) -> None:
        """Close the underlying HTTP client (if open)."""
        if self._client is not None:
            logger.debug("Closing HTTP client")
            import asyncio

            try:
                loop = asyncio.get_running_loop()
                if loop.is_running():
                    loop.create_task(self._client.aclose())
                else:
                    asyncio.run(self._client.aclose())
            except RuntimeError:
                asyncio.run(self._client.aclose())
            self._client = None

    async def aclose(self) -> None:
        """Async close the underlying HTTP client."""
        if self._client is not None:
            logger.debug("Closing HTTP client (async)")
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> Spectrograph:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        await self.aclose()

    # ------------------------------------------------------------------
    # Single model query with retry
    # ------------------------------------------------------------------

    async def _query_one(
        self, client: httpx.AsyncClient, model: str, prompt: str
    ) -> ModelResponse:
        """Query a single model with exponential-backoff retry.

        Retries on 429 (rate-limit) and 5xx (server) status codes.
        Returns an error-model response on final failure.
        """
        last_error: Exception | None = None

        for attempt in range(1, MAX_RETRIES + 2):  # initial + retries
            t0 = time.monotonic()
            try:
                resp = await client.post(
                    f"{self.base_url}/chat/completions",
                    json={
                        "model": model,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": self.max_tokens,
                        "temperature": self.temperature,
                    },
                    headers={"Authorization": f"Bearer {self.api_key}"},
                )
                elapsed = (time.monotonic() - t0) * 1000

                # Retryable status?
                if resp.status_code in RETRYABLE_STATUSES and attempt <= MAX_RETRIES:
                    delay = min(
                        RETRY_BASE_DELAY * (2 ** (attempt - 1)),
                        RETRY_MAX_DELAY,
                    )
                    logger.warning(
                        "Model %s returned %d (attempt %d/%d), "
                        "retrying in %.1fs",
                        model,
                        resp.status_code,
                        attempt,
                        MAX_RETRIES,
                        delay,
                    )
                    await asyncio.sleep(delay)
                    continue

                resp.raise_for_status()
                data: dict[str, Any] = resp.json()

                choices: list[dict[str, Any]] | None = data.get("choices")
                if not choices:
                    raise ResponseMalformed(
                        f"Model {model} returned no choices"
                    )

                content: str = choices[0]["message"]["content"]
                tokens: int = data.get("usage", {}).get("total_tokens", 0)

                logger.debug(
                    "Model %s responded in %.0fms (%d tokens)",
                    model,
                    elapsed,
                    tokens,
                )
                return ModelResponse(
                    model=model,
                    content=content,
                    elapsed_ms=elapsed,
                    tokens=tokens,
                )

            except httpx.TimeoutException as e:
                last_error = e
                elapsed = (time.monotonic() - t0) * 1000
                logger.warning(
                    "Model %s timed out (attempt %d/%d)",
                    model,
                    attempt,
                    MAX_RETRIES,
                )
                if attempt <= MAX_RETRIES:
                    delay = min(
                        RETRY_BASE_DELAY * (2 ** (attempt - 1)),
                        RETRY_MAX_DELAY,
                    )
                    await asyncio.sleep(delay)
                    continue
                return ModelResponse(
                    model=model,
                    content="",
                    elapsed_ms=elapsed,
                    error=f"Timeout after {attempt} attempts: {e}",
                )

            except httpx.HTTPStatusError as e:
                last_error = e
                elapsed = (time.monotonic() - t0) * 1000
                # Non-retryable status (4xx other than 429)
                status = e.response.status_code
                if status not in RETRYABLE_STATUSES or attempt > MAX_RETRIES:
                    return ModelResponse(
                        model=model,
                        content="",
                        elapsed_ms=elapsed,
                        error=f"HTTP {status}: {e}",
                    )
                delay = min(
                    RETRY_BASE_DELAY * (2 ** (attempt - 1)),
                    RETRY_MAX_DELAY,
                )
                logger.warning(
                    "Model %s returned %d (attempt %d/%d), retrying in %.1fs",
                    model,
                    status,
                    attempt,
                    MAX_RETRIES,
                    delay,
                )
                await asyncio.sleep(delay)

            except Exception as e:
                last_error = e
                elapsed = (time.monotonic() - t0) * 1000
                if attempt <= MAX_RETRIES:
                    delay = min(
                        RETRY_BASE_DELAY * (2 ** (attempt - 1)),
                        RETRY_MAX_DELAY,
                    )
                    logger.warning(
                        "Model %s error (attempt %d/%d): %s, retrying in %.1fs",
                        model,
                        attempt,
                        MAX_RETRIES,
                        e,
                        delay,
                    )
                    await asyncio.sleep(delay)
                    continue
                return ModelResponse(
                    model=model,
                    content="",
                    elapsed_ms=elapsed,
                    error=f"Error after {attempt} attempts: {e}",
                )

        # Should not reach here, but satisfy the type checker
        return ModelResponse(
            model=model,
            content="",
            elapsed_ms=0.0,
            error=f"Max retries ({MAX_RETRIES}) exhausted: {last_error}",
        )

    async def _query_all(self, prompt: str) -> list[ModelResponse]:
        """Query all models in parallel."""
        client = self._get_client()
        tasks = [self._query_one(client, m, prompt) for m in self.models]
        return await asyncio.gather(*tasks)

    async def _query_stream(
        self, prompt: str
    ) -> AsyncIterator[ModelResponse]:
        """Query all models and yield responses as each completes.

        Uses asyncio.as_completed so early finishers appear first.
        """
        client = self._get_client()
        tasks = {
            asyncio.create_task(
                self._query_one(client, m, prompt)
            ): m
            for m in self.models
        }

        for done in asyncio.as_completed(tasks):
            yield await done

    # ------------------------------------------------------------------
    # Public API — sync
    # ------------------------------------------------------------------

    def analyze(
        self,
        prompt: str,
        models: list[str] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> SpectrumResult:
        """Run a spectral analysis across multiple models.

        Args:
            prompt: The question or prompt to analyze.
            models: Override the default model list.
            max_tokens: Override max tokens per response.
            temperature: Override temperature.

        Returns:
            SpectrumResult with convergences, divergences, and insights.
        """
        # Allow per-call overrides
        old_models = self.models
        old_max = self.max_tokens
        old_temp = self.temperature
        if models is not None:
            self.models = models
        if max_tokens is not None:
            self.max_tokens = max_tokens
        if temperature is not None:
            self.temperature = temperature

        t0 = time.monotonic()
        try:
            responses = asyncio.run(self._query_all(prompt))
        finally:
            # Restore overrides (except if models was never overridden)
            if models is not None:
                self.models = old_models
            if max_tokens is not None:
                self.max_tokens = old_max
            if temperature is not None:
                self.temperature = old_temp

        elapsed = (time.monotonic() - t0) * 1000
        ok_responses = [r for r in responses if r.ok]

        # Run analysis
        from spectro.analysis import analyze_spectrum

        convergences, divergences, unique_insights, confidence = (
            analyze_spectrum(ok_responses)
        )

        result = SpectrumResult(
            prompt=prompt,
            responses=responses,
            convergences=convergences,
            divergences=divergences,
            unique_insights=unique_insights,
            confidence=confidence,
            elapsed_ms=elapsed,
        )

        logger.info(
            "Analysis complete: %d/%d models OK, %.0fms, confidence=%.0f%%",
            result.n_ok,
            result.n_models,
            elapsed,
            confidence * 100,
        )
        return result

    # ------------------------------------------------------------------
    # Public API — async
    # ------------------------------------------------------------------

    async def analyze_async(
        self,
        prompt: str,
        models: list[str] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> SpectrumResult:
        """Async version of :meth:`analyze`.

        Use inside an existing event loop (e.g., FastAPI, asyncio.run).
        """
        old_models = self.models
        old_max = self.max_tokens
        old_temp = self.temperature
        if models is not None:
            self.models = models
        if max_tokens is not None:
            self.max_tokens = max_tokens
        if temperature is not None:
            self.temperature = temperature

        t0 = time.monotonic()
        try:
            responses = await self._query_all(prompt)
        finally:
            if models is not None:
                self.models = old_models
            if max_tokens is not None:
                self.max_tokens = old_max
            if temperature is not None:
                self.temperature = old_temp

        elapsed = (time.monotonic() - t0) * 1000
        ok_responses = [r for r in responses if r.ok]

        from spectro.analysis import analyze_spectrum

        convergences, divergences, unique_insights, confidence = (
            analyze_spectrum(ok_responses)
        )

        return SpectrumResult(
            prompt=prompt,
            responses=responses,
            convergences=convergences,
            divergences=divergences,
            unique_insights=unique_insights,
            confidence=confidence,
            elapsed_ms=elapsed,
        )

    # ------------------------------------------------------------------
    # Public API — streaming
    # ------------------------------------------------------------------

    async def analyze_stream(
        self,
        prompt: str,
        models: list[str] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> AsyncIterator[ModelResponse]:
        """Stream partial results as each model completes.

        Yields ModelResponse objects as they arrive, rather than
        waiting for all models to finish. Useful for progressive
        display or UIs.

        Args:
            prompt: The question or prompt to analyze.
            models: Override the default model list.
            max_tokens: Override max tokens per response.
            temperature: Override temperature.

        Yields:
            ModelResponse objects, one per model, in completion order.
        """
        if models is not None:
            saved_models = self.models
            self.models = models
        if max_tokens is not None:
            saved_tokens = self.max_tokens
            self.max_tokens = max_tokens
        if temperature is not None:
            saved_temp = self.temperature
            self.temperature = temperature

        try:
            async for response in self._query_stream(prompt):
                yield response
        finally:
            if models is not None:
                self.models = saved_models  # type: ignore[used-before-def]
            if max_tokens is not None:
                self.max_tokens = saved_tokens  # type: ignore[used-before-def]
            if temperature is not None:
                self.temperature = saved_temp  # type: ignore[used-before-def]
