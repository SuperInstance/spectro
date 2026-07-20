# Spectro — Multi-Model Cognitive Spectrograph

> Split the beam. Read the spectrum. The convergences are the coastline. The divergences are the interesting water.

## What

Spectro sends your prompt to N different AI models in parallel, then analyzes what they agree on (convergences) and where they diverge (divergences). The output isn't any single model's answer — it's the *pattern* across all of them.

This is not model routing (picking the best model). This is not ensemble voting (averaging answers). This is **spectral analysis of model cognition** — treating the differences between models as signal, not noise.

## Why

Every paradigm essay in the SuperInstance corpus points at the same truth: **different models are different perspectives, not different quality**. The convergences between independent models reveal high-confidence territory. The divergences reveal the interesting edges — the places where the question is genuinely uncertain, or where a model sees something the others miss.

- A question where all 5 models agree → high confidence, low exploration value
- A question where models disagree → genuinely uncertain, needs human judgment  
- A question where one model is unique → either a blind spot or a breakthrough

Spectro makes this visible.

## Install

```bash
pip install spectro-spectrograph
```

## Quick Start

```bash
# Set your DeepInfra API key (or any OpenAI-compatible provider)
export DEEPINFRA_API_KEY=...

# Run a spectral analysis across 5 models
spectro "What is the most important quality in a senior engineer?"
```

## How It Works

```
Your Prompt
    │
    ▼
┌─────────────────────────────────┐
│  Spectro Engine (parallel)      │
│                                 │
│  Model A ──→ Response A         │
│  Model B ──→ Response B         │
│  Model C ──→ Response C         │
│  Model D ──→ Response D         │
│  Model E ──→ Response E         │
│                                 │
└─────────────┬───────────────────┘
              │
              ▼
┌─────────────────────────────────┐
│  Analysis Layer                 │
│                                 │
│  1. Extract key concepts        │
│  2. Find convergences (shared)  │
│  3. Find divergences (unique)   │
│  4. Map the agreement space     │
│                                 │
└─────────────┬───────────────────┘
              │
              ▼
┌─────────────────────────────────┐
│  Spectrum Report                │
│                                 │
│  ████ CONVERGENCE (high conf)   │
│  ██░░ PARTIAL AGREEMENT         │
│  ░░░░ DIVERGENCE (explore here) │
│  █░█░ UNIQUE INSIGHTS           │
│                                 │
└─────────────────────────────────┘
```

## CLI

```bash
# Default ensemble (5 models)
spectro "Should we use microservices?"

# Choose specific models
spectro "What causes bad code?" \
  --models deepseek,seed-pro,ornith,nemotron

# Focus on a specific analysis type
spectro "Is this architecture sound?" \
  --focus divergence

# Output as JSON
spectro "Best practices for API design" \
  --format json

# Verbose mode (show full responses + analysis)
spectro "What is consciousness?" \
  --verbose
```

## Python API

```python
from spectro import Spectrograph

spec = Spectrograph(api_key="...")

result = spec.analyze(
    prompt="What's the biggest risk in microservices?",
    models=[
        "deepseek-ai/DeepSeek-V4-Flash",
        "ByteDance/Seed-2.0-pro",
        "deepreinforce-ai/Ornith-1.0-35B",
    ],
)

print(result.convergences)   # concepts all models share
print(result.divergences)    # where models disagree
print(result.unique_insights) # what each model saw alone
print(result.confidence)     # 0.0-1.0 agreement score
```

## The Theory

Spectro is built on the paradigm documented across 1,600+ essays in the SuperInstance/AI-Writings corpus:

- **The Spectrograph**: Every model output is a composite beam. Splitting it reveals composition.
- **The Ensemble Is the Experiment**: The intelligence is in the relationship, not the individual.
- **Charts Not Maps**: Each model is a different chart of the same territory.
- **Two Charts Same Ocean**: Convergences show the ocean floor. Divergences show the interesting water.
- **Cast Thin First**: Cheap models discover the territory. Expensive models synthesize.

## License

MIT
