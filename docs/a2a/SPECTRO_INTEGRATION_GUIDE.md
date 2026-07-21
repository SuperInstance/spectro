# Spectro Agent-to-Agent (A2A) Integration Guide

This document describes how external agents can integrate with and harness the Spectro package for semantic analysis, model convergence detection, and spectral analysis of AI-generated content.

## Overview

Spectro is a library for analyzing the semantic spectrum of AI model outputs. It helps agents:

- Detect convergences and divergences across multiple models
- Analyze the semantic resonance between different AI perspectives
- Generate spectrum visualizations of model agreement
- Identify unique insights that emerge from specific model combinations
- Measure the "thickness" of cognitive charts across different models

## Core Components

### 1. ModelResponse
Represents a response from an AI model:
```python
from spectro.core import ModelResponse

response = ModelResponse(
    model="deepseek-chat",
    content="The quantum fishing boat cuts through the morning fog...",
    elapsed_ms=1245,
    tokens=287,
    error=None
)
```

### 2. SpectrumResult
Result of spectral analysis:
```python
from spectro.core import SpectrumResult

result = SpectrumResult(
    convergences=["Both models agree on the maritime setting"],
    divergences=["Model A focuses on sensory details, Model B on procedural aspects"],
    unique_insights=["Model C introduces the concept of quantum buoyancy"],
    confidence=0.87,
    spectral_density=[0.2, 0.5, 0.8, 0.6, 0.3]  # Frequency domain representation
)
```

### 3. Spectrograph
Main analysis engine:
```python
from spectro.core import Spectrograph

spectrograph = Spectrograph(
    similarity_threshold=0.75,
    min_convergence_models=2,
    enable_streaming=False
)
```

## Modular Usage Patterns

### 1. Basic Spectral Analysis

Analyze responses from multiple models:

```python
from spectro.core import Spectrograph, ModelResponse

# Collect responses from different models
responses = [
    ModelResponse("deepseek-chat", "The boat moves silently through water...", 1200, 150),
    ModelResponse("seed-2.0-pro", "Observing the fishing vessel at dawn...", 1800, 200),
    ModelResponse("ornith-35b", "CoCapn checks the pheromone gradients...", 2200, 180),
    ModelResponse("nemotron-ultra", "The substrate processes sensor data...", 1000, 120)
]

# Create spectrograph and analyze
spectrograph = Spectrograph()
result = spectrograph.analyze(responses)

print(f"Convergences: {result.convergences}")
print(f"Divergences: {result.divergences}")
print(f"Unique insights: {result.unique_insights}")
print(f"Overall confidence: {result.confidence}")
```

### 2. Streaming Analysis

For real-time agent interactions:

```python
from spectro.core import Spectrograph, ModelResponse

# Create streaming spectrograph
spectrograph = Spectrograph(enable_streaming=True, window_size=5)

# As each model response arrives, add it to the analysis
for model_name in ["deepseek", "seed", "ornith"]:
    # Get response from model (simulated)
    response = get_model_response(model_name, prompt)
    
    # Add to streaming analysis
    spectrograph.add_response(ModelResponse(
        model=model_name,
        content=response,
        elapsed_ms=measure_latency(),
        tokens=count_tokens(response)
    ))
    
    # Get current spectrum analysis
    if spectrograph.has_enough_data():
        result = spectrograph.get_current_spectrum()
        act_on_spectral_analysis(result)
```

### 3. Divergence-Focused Analysis

When you specifically want to find where models disagree:

```python
from spectro.analysis import DivergenceHunter

# Focus on finding meaningful disagreements
hunter = DivergenceHunter(min_disagreement_score=0.6)

responses = get_model_responses(prompt)  # From your agent fleet
divergences = hunter.find_meaningful_divergences(responses)

for divergence in divergences:
    print(f"Models {divergence.models} disagree: {divergence.explanation}")
    print(f"  Suggestion: {divergence.resolution_hint}")
```

### 4. Convergence Validation

When you need to verify that models agree on critical points:

```python
from spectro.analysis import ConvergenceValidator

validator = ConvergenceValidator(required_confidence=0.9)
responses = get_authoritative_model_responses(critical_question)

if validator.validate(responses, ["safety_constraint", "physical_law", "ethical_boundary"]):
    proceed_with_action()
else:
    request_human_oversight()
```

## Advanced Integration Patterns

### 1. Adaptive Model Selection

Use spectral analysis to dynamically choose which models to consult:

```python
from spectro.core import Spectrograph
from spectro.analysis import ModelAnimator

spectrograph = Spectrograph()
animator = ModelAnimator(spectrograph)

# Start with cheap, fast models
current_models = ["deepseek-chat", "seed-mini"]

# Analyze responses
responses = [get_response(m, prompt) for m in current_models]
result = spectrograph.analyze(responses)

# If confidence is low, add more expensive/different models
if result.confidence < 0.7:
    current_models.extend(["seed-2.0-pro", "ornith-35b"])
    # Re-analyze with expanded model set
    # ... 

# If we're seeing lots of unique insights, we might be over-specializing
if len(result.unique_insights) > len(current_models) * 0.5:
    current_models = ["deepseek-chat"]  # Reset to baseline
```

### 2. Spectral Feedback Loops

Create feedback loops where agents adjust based on spectral analysis:

```python
class SpectralAgent:
    def __init__(self):
        self.spectrograph = Spectrograph()
        self.confidence_threshold = 0.8
        self.attempts = 0
        self.max_attempts = 3
    
    def act(self, prompt):
        self.attempts += 1
        
        # Get responses from model ensemble
        responses = self.get_ensemble_responses(prompt)
        
        # Analyze the spectrum
        result = self.spectrograph.analyze(responses)
        
        # If we have high confidence convergence, act
        if result.confidence > self.confidence_threshold and result.convergences:
            return self.synthesize_action(result.convergences)
        
        # If we're stuck in divergence, try a different approach
        if len(result.divergences) > 2 and self.attempts < self.max_attempts:
            return self.reframe_prompt(prompt, result.divergences)
        
        # Otherwise, return the best available insight
        return result.unique_insights[0] if result.unique_insights else "no clear path forward"
    
    def get_ensemble_responses(self, prompt):
        # Implementation would call your model fleet
        pass
    
    def synthesize_action(self, convergences):
        # Implementation would create action from agreed-upon points
        pass
    
    def reframe_prompt(self, prompt, divergences):
        # Implementation would adjust prompt based on disagreements
        pass
```

### 3. Knowledge Distillation via Spectral Analysis

Use spectral analysis to distill knowledge from multiple models into a unified understanding:

```python
from spectro.core import Spectrograph
from spectro.analysis import InsightSynthesizer

spectrograph = Spectrograph()
synthesizer = InsightSynthesizer()

# Collect diverse perspectives on a topic
perspectives = []
for model in ["deepseek-chat", "seed-2.0-pro", "ornith-35b", "nemotron-ultra", "hermes-405b"]:
    response = get_model_response(model, "Explain the conservation law of intelligence")
    perspectives.append(ModelResponse(model, response, 0, 0))

# Analyze the spectrum
result = spectrograph.analyze(perspectives)

# Synthesize the insights into a unified understanding
unified_understanding = synthesizer.synthesize(
    convergences=result.convergences,
    unique_insights=result.unique_insights,
    model_weights=get_model_authority_weights()  # Weight by demonstrated expertise
)

# Store in your agent's knowledge base
knowledge_base.store("conservation_law_intelligence", unified_understanding)
```

## Configuration and Customization

### Spectrograph Configuration
```python
from spectro.core import Spectrograph

# High precision analysis (slower but more detailed)
spectrograph = Spectrograph(
    similarity_threshold=0.85,  # Higher threshold for stricter convergence
    min_convergence_models=3,   # Require agreement from more models
    enable_streaming=False,
    cache_embeddings=True       # Cache model embeddings for repeated use
    )

# Fast, low-latency analysis (good for real-time feedback)
spectrograph = Spectrograph(
    similarity_threshold=0.65,  # Lower threshold catches more similarities
    min_convergence_models=2,
    enable_streaming=True,
    window_size=3               # Only look at last 3 responses
)
```

### Custom Similarity Metrics
Plug in your own similarity metrics for specialized domains:

```python
from spectro.core import Spectrograph
from spectro.similarity import SimilarityMetric

class FishingDomainSimilarity(SimilarityMetric):
    def compute(self, text1: str, text2: str) -> float:
        # Custom similarity metrics for fishing domain
        # Might weigh maritime terminology, procedural details, safety considerations
        # More heavily than general semantic similarity
        pass

spectrograph = Spectrograph(similarity_metric=FishingDomainSimilarity())
```

## State Management and Persistence

### Saving and Loading Analyses
```python
import json
from spectro.core import SpectrumResult

# Save analysis results
def save_spectrum_result(result: SpectrumResult, filepath: str):
    data = {
        "convergences": result.convergences,
        "divergences": result.divergences,
        "unique_insights": result.unique_insights,
        "confidence": result.confidence,
        "spectral_density": result.spectral_density,
        "timestamp": time.time()
    }
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

# Load analysis results
def load_spectrum_result(filepath: str) -> SpectrumResult:
    with open(filepath, 'r') as f:
        data = json.load(f)
    return SpectrumResult(
        convergences=data["convergences"],
        divergences=data["divergences"],
        unique_insights=data["unique_insights"],
        confidence=data["confidence"],
        spectral_density=data["spectral_density"]
    )
```

### Agent State Storage (Git-Agent Pattern)

Like hanging up a drysuit after a productive scuba dive, agents can store their spectral analysis state:

```bash
# After completing a analysis session
git add docs/analysis/  # Or specific analysis directories
git commit -m "agent spectral session: analyzed model convergence for tide prediction model"
git push origin main

# To resume or build upon previous work
git pull origin main
# Previous analyses are available in the commit history
```

## Best Practices for Agent Integration

### 1. Start with Clear Prompts
The quality of spectral analysis depends on the clarity and specificity of your prompts.

### 2. Use Appropriate Model Ensembles
Choose models that complement each other rather than duplicate perspectives.

### 3. Monitor Spectral Density
Watch the spectral density array - it shows the frequency distribution of agreement/disagreement.

### 4. Trust the Convergences, Learn from the Divergences
Convergences show where multiple independent perspectives agree (likely ground truth).
Divergences show where perspectives differ (potential for new insights or hidden assumptions).

### 5. Unique Insights are Gold
Pay special attention to `unique_insights` - these are perspectives that only one model brought to the table.

### 6. Combine with Other Substrates
Spectro works excellently with VaaS - use Spectro to analyze the outputs of VaaS agents, or use VaaS to implement the agents whose outputs you're analyzing with Spectro.

## Example: Model Election Agent

Here's how an agent might use Spectro to democratically choose among competing proposals:

```python
from spectro.core import Spectrograph
from spectro.analysis import ProposalEvaluator

class DemocraticModelAgent:
    def __init__(self):
        self.spectrograph = Spectrograph(similarity_threshold=0.7)
        self.evaluator = ProposalEvaluator()
        self.proposal_history = []
    
    def propose_and_vote(self, proposals):
        """
        Uses spectral analysis to help choose between competing proposals
        """
        # Get model evaluations of each proposal
        proposal_responses = {}
        for proposal_id, proposal_text in proposals.items():
            responses = []
            for model in ["deepseek-chat", "seed-2.0-pro", "ornith-35b"]:
                response = get_model_evaluation(model, proposal_text)
                responses.append(ModelResponse(model, response, 0, 0))
            proposal_responses[proposal_id] = responses
        
        # Analyze each proposal's spectral signature
        proposal_spectra = {}
        for proposal_id, responses in proposal_responses.items():
            spectrum = self.spectrograph.analyze(responses)
            proposal_spectra[proposal_id] = spectrum
        
        # Evaluate proposals based on their spectral properties
        ranked_proposals = self.evaluator.rank_proposals(proposal_spectra)
        
        # Return the proposal with best spectral properties
        return ranked_proposals[0] if ranked_proposals else None
    
    def get_model_evaluation(self, model, proposal):
        # Would call the model to evaluate the proposal
        # Return: "This proposal is strong because..." or "This proposal has issues with..."
        pass
```

## Safety and Ethics

### 1. Beware of False Convergences
Just because multiple models agree doesn't mean they're correct - they could all be making the same mistake.

### 2. Value Thoughtful Divergence
Sometimes the lone dissenting model has detected something important that the others missed.

### 3. Check for Model Collusion
If you're using models from the same family or with similar training, they may converge due to shared biases rather than truth.

### 4. Consider the Full Spectrum
Don't just look at the peaks (convergences) - the troughs (divergences) and the overall shape (spectral density) tell important stories too.

### 5. Audit Regularly
Periodically review your spectral analyses to ensure they're leading to good outcomes, not just pretty patterns.
