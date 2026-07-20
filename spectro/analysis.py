"""Spectral analysis — finding convergences, divergences, and unique insights.

The analysis layer takes N model responses and produces:
1. Convergences — concepts/claims that appear across multiple responses
2. Divergences — where models disagree or take different angles
3. Unique insights — concepts that appear in only one response
4. Confidence score — how much the models agree overall

This is the heart of the spectrograph. The analysis is intentionally
lightweight (no external ML dependencies) so the package stays fast
and portable. We use n-gram overlap, keyword extraction, and
structural comparison rather than embedding models.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

from spectro.core import ModelResponse


# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------

STOPWORDS = frozenset(
    """
    a an the and or but in on at to for of with by from is are was were be been
    being have has had do does did will would could should may might must shall
    can need this that these those it its as if then than so not no yes you your
    they their them he she his her we our us i my me him one two three first last
    also more most some any all each every other same new old good bad big small
    about into over under out up down off again further once here there when where
    why how what which who whom whose
    """.split()
)


def tokenize(text: str) -> list[str]:
    """Split text into lowercase word tokens."""
    return re.findall(r"[a-z]{3,}", text.lower())


def extract_keywords(text: str, min_freq: int = 1) -> set[str]:
    """Extract meaningful keywords from text."""
    tokens = [
        t for t in tokenize(text)
        if t not in STOPWORDS and len(t) >= 4
    ]
    counter = Counter(tokens)
    return {w for w, c in counter.items() if c >= min_freq}


def extract_key_phrases(text: str) -> set[str]:
    """Extract 2-3 word phrases from text."""
    tokens = tokenize(text)
    phrases = set()
    # Bigrams
    for i in range(len(tokens) - 1):
        if tokens[i] not in STOPWORDS and tokens[i + 1] not in STOPWORDS:
            phrases.add(f"{tokens[i]} {tokens[i + 1]}")
    # Trigrams
    for i in range(len(tokens) - 2):
        if (
            tokens[i] not in STOPWORDS
            and tokens[i + 2] not in STOPWORDS
        ):
            phrases.add(f"{tokens[i]} {tokens[i + 1]} {tokens[i + 2]}")
    return phrases


def extract_sentences(text: str) -> list[str]:
    """Split text into sentences."""
    # Simple but effective sentence splitter
    sentences = re.split(r"[.!?]+", text)
    return [s.strip() for s in sentences if len(s.strip()) > 20]


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def _concept_overlap(responses: list[ModelResponse]) -> dict[str, set[str]]:
    """Build a concept → models map."""
    concept_map: dict[str, set[str]] = {}
    for resp in responses:
        # Combine keywords + key phrases
        concepts = extract_keywords(resp.content) | extract_key_phrases(resp.content)
        for concept in concepts:
            if concept not in concept_map:
                concept_map[concept] = set()
            concept_map[concept].add(resp.model)
    return concept_map


def _find_convergences(
    concept_map: dict[str, set[str]],
    n_models: int,
    min_agreement: int = 2,
) -> list[dict[str, Any]]:
    """Find concepts shared by multiple models."""
    convergences = []
    for concept, models in concept_map.items():
        agreement = len(models)
        if agreement >= min_agreement:
            convergences.append({
                "concept": concept,
                "models": sorted(models),
                "agreement": agreement,
                "strength": agreement / n_models,
            })
    # Sort by strength descending
    convergences.sort(key=lambda c: c["strength"], reverse=True)
    return convergences


def _find_divergences(
    responses: list[ModelResponse],
    concept_map: dict[str, set[str]],
) -> list[dict[str, Any]]:
    """Find areas where models take different angles."""
    divergences = []

    # A divergence is a concept cluster where models use different vocabulary
    # for related ideas. We detect this via low-overlap key phrases.
    model_phrases = {}
    for resp in responses:
        phrases = extract_key_phrases(resp.content)
        model_phrases[resp.model] = phrases

    # Compare each pair of models
    for i, resp_a in enumerate(responses):
        for resp_b in responses[i + 1:]:
            shared = model_phrases[resp_a.model] & model_phrases[resp_b.model]
            total = model_phrases[resp_a.model] | model_phrases[resp_b.model]
            if total:
                overlap = len(shared) / len(total)
                if overlap < 0.15:  # low phrase overlap = different angle
                    divergences.append({
                        "model_a": resp_a.model,
                        "model_b": resp_b.model,
                        "phrase_overlap": round(overlap, 3),
                        "note": "Models approached the question from different angles",
                    })

    # Also detect contradicting sentiment on the same concept
    for concept, models in concept_map.items():
        if len(models) >= 2:
            # Check if models that mention this concept express it differently
            mentions = {}
            for resp in responses:
                if resp.model in models:
                    # Find sentences containing the concept
                    relevant = [
                        s for s in extract_sentences(resp.content)
                        if concept in s.lower()
                    ]
                    if relevant:
                        mentions[resp.model] = relevant[:2]  # top 2 sentences

            if len(mentions) >= 2:
                # Simple check: do they use negation differently?
                has_negation = {
                    m: any("not " in s or "n't" in s or "never" in s
                           for s in sents)
                    for m, sents in mentions.items()
                           }
                neg_values = list(has_negation.values())
                if len(set(neg_values)) > 1:
                    divergences.append({
                        "concept": concept,
                        "models": sorted(models),
                        "note": "Models disagree on this concept (negation mismatch)",
                        "evidence": {
                            m: sents[:1]
                            for m, sents in mentions.items()
                        },
                    })

    return divergences


def _find_unique_insights(
    responses: list[ModelResponse],
    concept_map: dict[str, set[str]],
) -> list[dict[str, Any]]:
    """Find concepts that appear in only one model's response."""
    unique = []
    for concept, models in concept_map.items():
        if len(models) == 1:
            model_name = next(iter(models))
            # Find the sentence containing this concept
            for resp in responses:
                if resp.model == model_name:
                    relevant = [
                        s for s in extract_sentences(resp.content)
                        if concept in s.lower()
                    ]
                    unique.append({
                        "concept": concept,
                        "model": model_name,
                        "context": relevant[0] if relevant else "",
                    })
                    break

    # Sort by concept length (longer phrases are usually more interesting)
    unique.sort(key=lambda u: len(u["concept"]), reverse=True)
    return unique[:20]  # top 20


def _confidence_score(convergences: list[dict], n_models: int) -> float:
    """Calculate overall confidence based on convergence strength."""
    if not convergences or n_models == 0:
        return 0.0
    # Average of top-10 convergence strengths
    top = convergences[:10]
    return sum(c["strength"] for c in top) / len(top) if top else 0.0


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

def analyze_spectrum(
    responses: list[ModelResponse],
) -> tuple[
    list[dict[str, Any]],  # convergences
    list[dict[str, Any]],  # divergences
    list[dict[str, Any]],  # unique insights
    float,                  # confidence
]:
    """Analyze a set of model responses for spectral patterns.

    Returns (convergences, divergences, unique_insights, confidence).
    """
    if not responses:
        return [], [], [], 0.0

    n = len(responses)
    concept_map = _concept_overlap(responses)

    convergences = _find_convergences(concept_map, n)
    divergences = _find_divergences(responses, concept_map)
    unique_insights = _find_unique_insights(responses, concept_map)
    confidence = _confidence_score(convergences, n)

    return convergences, divergences, unique_insights, confidence
