# Changelog

All notable changes to Spectro will be documented in this file.

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
