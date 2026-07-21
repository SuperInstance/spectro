# Changelog

All notable changes to Spectro will be documented in this file.

## [0.2.0] — 2026-07-21

### Added
- **Custom exceptions**: `SpectroError` (base), `APIKeyMissing`, `ModelUnavailable`, `AnalysisTimeout`, `ResponseMalformed` — every error has a name and a family
- **Retry logic**: `_query_one()` now retries with exponential backoff (1s, 2s, 4s up to 10s) on 429, 500, 502, 503, 504 errors. Max 3 retries.
- **Streaming support**: `analyze_stream()` async generator yields `ModelResponse` objects as each model completes (uses `asyncio.as_completed`)
- **Async API**: `analyze_async()` for use inside existing event loops (FastAPI, asyncio)
- **Context manager**: `async with Spectrograph() as spec:` with `__aenter__`/`__aexit__` support
- **`.close()` / `.aclose()`**: explicit client lifecycle management
- **Logging**: proper `logging.getLogger(__name__)` in every module, with `--log-level` CLI flag
- **Semantic analysis (optional)**: when `sentence-transformers` is installed (`pip install spectro-spectrograph[semantic]`), divergence detection includes cosine similarity between sentence embeddings
- **Edge case tests**: 14 new tests covering whitespace responses, malformed JSON, missing fields, HTTP errors, network failures, unicode, long responses, CLI output, and state restoration

### Changed
- **Type hints**: strict typing (`str | None`, `list[str]`, `Any`) throughout all modules
- **`_resolve_api_key()`**: extracted into dedicated static method with chain-of-responsibility fallback
- **API key resolution**: now raises `APIKeyMissing` instead of generic `ValueError`
- **State restoration**: `analyze()` properly restores `models`, `max_tokens`, and `temperature` after overrides
- **`ModelResponse.ok`**: uses `strip()` to reject whitespace-only responses
- **`pyproject.toml`**: added `[project.optional-dependencies] semantic = ["sentence-transformers>=2.2", "numpy>=1.24"]` and pytest config

### Fixed
- CLI output no longer depends on a non-existent `Colors` class
- Empty content responses are correctly marked `not ok`

## [0.1.0] — 2026-07-20

### Added
- **Core engine**: `Spectrograph` class with async parallel multi-model queries via any OpenAI-compatible API
- **Spectral analysis**: convergence detection (shared concepts), divergence mapping (disagreement patterns), unique insight extraction (single-model perspectives)
- **CLI**: `spectro "prompt"` with text/JSON output, verbose mode, custom model selection
- **Default ensemble**: 5-model repertory company (DeepSeek, Seed Mini, Ornith, Nemotron, Seed Pro) based on documented casting experiments
- **26 tests**: full coverage of dataclasses, text utilities, analysis engine, and initialization
- **API key discovery**: env vars → key file → explicit constructor, supports DeepInfra and OpenAI

### Design Principles
- **Models are perspectives, not quality levels** — the ensemble is the experiment
- **Convergences = high confidence** — concepts shared across models
- **Divergences = exploration targets** — where models disagree
- **The spectrum is the finding** — not any single model's answer
