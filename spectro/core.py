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
import os
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

# ---------------------------------------------------------------------------
# Defaults — the "repertory company" from the casting experiments.
# These are documented in TOOLS.md and the casting-call repo.
# ---------------------------------------------------------------------------

DEFAULT_MODELS = [
    "deepseek-ai/DeepSeek-V4-Flash",      # cheap workhorse, 8/10
    "ByteDance/Seed-2.0-mini",             # thin chart, ideation
    "deepreinforce-ai/Ornith-1.0-35B",     # best fiction, punches above weight
    "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B",  # structural, sergeant
    "ByteDance/Seed-2.0-pro",              # 9/10 best overall, lyrical
]

DEFAULT_BASE_URL = "https://api.deepinfra.com/v1/openai"
DEFAULT_TIMEOUT = 120.0
DEFAULT_MAX_TOKENS = 2048
DEFAULT_TEMPERATURE = 0.7


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
        return self.error is None and bool(self.content)


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
        lines = [
            f"Spectrum: {self.n_ok}/{self.n_models} models responded",
            f"Confidence: {self.confidence:.0%} agreement",
            f"Convergences: {len(self.convergences)}",
            f"Divergences: {len(self.divergences)}",
            f"Unique insights: {len(self.unique_insights)}",
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
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
                    "content": r.content[:500] + "..." if len(r.content) > 500 else r.content,
                    "elapsed_ms": r.elapsed_ms,
                    "tokens": r.tokens,
                    "error": r.error,
                }
                for r in self.responses
            ],
        }


class Spectrograph:
    """The multi-model cognitive spectrograph.

    Sends prompts to multiple models in parallel and analyzes the
    convergences and divergences in their responses.

    Usage:
        spec = Spectrograph()  # reads DEEPINFRA_API_KEY
        result = spec.analyze("What makes code maintainable?")
        print(result.summary())
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        models: list[str] | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        timeout: float = DEFAULT_TIMEOUT,
    ):
        self.api_key = api_key or os.environ.get("DEEPINFRA_API_KEY", "") or os.environ.get("DEEPINFRA_KEY", "")
        if not self.api_key:
            self.api_key = (
                os.environ.get("OPENAI_API_KEY", "")
                or self._read_key_file()
            )
        if not self.api_key:
            raise ValueError(
                "No API key found. Set DEEPINFRA_API_KEY, OPENAI_API_KEY, "
                "or create ~/.openclaw/.deepinfra-key"
            )

        self.base_url = base_url.rstrip("/")
        self.models = models or DEFAULT_MODELS
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout

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

    async def _query_one(
        self, client: httpx.AsyncClient, model: str, prompt: str
    ) -> ModelResponse:
        """Query a single model."""
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
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            tokens = data.get("usage", {}).get("total_tokens", 0)
            return ModelResponse(
                model=model,
                content=content,
                elapsed_ms=elapsed,
                tokens=tokens,
            )
        except Exception as e:
            elapsed = (time.monotonic() - t0) * 1000
            return ModelResponse(
                model=model,
                content="",
                elapsed_ms=elapsed,
                error=str(e),
            )

    async def _query_all(self, prompt: str) -> list[ModelResponse]:
        """Query all models in parallel."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            tasks = [self._query_one(client, m, prompt) for m in self.models]
            return await asyncio.gather(*tasks)

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
        if models:
            self.models = models
        if max_tokens:
            self.max_tokens = max_tokens
        if temperature is not None:
            self.temperature = temperature

        t0 = time.monotonic()
        try:
            responses = asyncio.run(self._query_all(prompt))
        finally:
            self.models = old_models
            self.max_tokens = old_max
            self.temperature = old_temp

        elapsed = (time.monotonic() - t0) * 1000
        ok_responses = [r for r in responses if r.ok]

        # Run analysis
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
