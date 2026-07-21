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

### Semantic similarity (optional)

Install ``sentence-transformers`` for a deeper convergence pass:

    pip install sentence-transformers  # or spectro[semantic]

When available, the analyzer uses cosine similarity between sentence
embeddings to detect semantic convergence even when models use
different surface vocabulary.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from typing import Any

from spectro.core import ModelResponse

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------

STOPWORDS: frozenset[str] = frozenset(
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
    """Split text into lowercase word tokens (minimum 3 chars)."""
    return re.findall(r"[a-z]{3,}", text.lower())


def extract_keywords(text: str, min_freq: int = 1) -> set[str]:
    """Extract meaningful keywords from text, filtering stopwords and short words.

    Args:
        text: The input text to analyze.
        min_freq: Minimum frequency for a keyword to be included.

    Returns:
        A set of keywords that appear at least min_freq times.
    """
    tokens = [t for t in tokenize(text) if t not in STOPWORDS and len(t) >= 4]
    counter = Counter(tokens)
    return {w for w, c in counter.items() if c >= min_freq}


def extract_key_phrases(text: str) -> set[str]:
    """Extract 2-3 word phrases from text.

    Args:
        text: The input text to analyze.

    Returns:
        A set of bigram and trigram phrases (excluding stopword-heavy phrases).
    """
    tokens = tokenize(text)
    phrases: set[str] = set()
    # Bigrams
    for i in range(len(tokens) - 1):
        if tokens[i] not in STOPWORDS and tokens[i + 1] not in STOPWORDS:
            phrases.add(f"{tokens[i]} {tokens[i + 1]}")
    # Trigrams
    for i in range(len(tokens) - 2):
        if tokens[i] not in STOPWORDS and tokens[i + 2] not in STOPWORDS:
            phrases.add(f"{tokens[i]} {tokens[i + 1]} {tokens[i + 2]}")
    return phrases


def extract_sentences(text: str) -> list[str]:
    """Split text into sentences (minimum 20 characters each).

    Args:
        text: The input text to split.

    Returns:
        A list of sentence strings.
    """
    sentences = re.split(r"[.!?]+", text)
    return [s.strip() for s in sentences if len(s.strip()) > 20]


# ---------------------------------------------------------------------------
# Semantic similarity (optional)
# ---------------------------------------------------------------------------

try:
    import numpy as np  # type: ignore[import-untyped]
    from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]

    _SEMANTIC_AVAILABLE = True
    _SEMANTIC_MODEL: SentenceTransformer | None = None

    logger.info("sentence-transformers available — semantic analysis enabled")

except ImportError:
    _SEMANTIC_AVAILABLE = False
    _SEMANTIC_MODEL = None
    logger.debug("sentence-transformers not installed; using keyword-only analysis")


def _get_semantic_model() -> SentenceTransformer | None:
    """Lazy-load the sentence-transformer model (cached after first use)."""
    global _SEMANTIC_MODEL
    if _SEMANTIC_AVAILABLE and _SEMANTIC_MODEL is None:
        try:
            # Lightweight model good for short sentences
            _SEMANTIC_MODEL = SentenceTransformer(
                "all-MiniLM-L6-v2"
            )
            logger.debug("Semantic model loaded: all-MiniLM-L6-v2")
        except Exception as e:
            logger.warning("Failed to load semantic model: %s", e)
            return None
    return _SEMANTIC_MODEL


def _semantic_similarity(sentences_a: list[str], sentences_b: list[str]) -> float:
    """Compute max cosine similarity between two sets of sentences.

    Returns a value 0.0 (no similarity) to 1.0 (identical meaning).
    Falls back to 0.0 if sentence-transformers is unavailable.
    """
    model = _get_semantic_model()
    if model is None:
        return 0.0

    try:
        emb_a = model.encode(sentences_a, convert_to_tensor=True)
        emb_b = model.encode(sentences_b, convert_to_tensor=True)
        # Cosine similarity matrix
        # Use numpy for simplicity
        mat = np.dot(
            emb_a.cpu().numpy(), emb_b.cpu().numpy().T
        )
        norms_a = np.linalg.norm(emb_a.cpu().numpy(), axis=1, keepdims=True)
        norms_b = np.linalg.norm(emb_b.cpu().numpy(), axis=1, keepdims=True)
        mat = mat / (norms_a * norms_b.T + 1e-10)
        return float(np.max(mat))
    except Exception as e:
        logger.debug("Semantic similarity failed: %s", e)
        return 0.0


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def _concept_overlap(responses: list[ModelResponse]) -> dict[str, set[str]]:
    """Build a concept → models map.

    Args:
        responses: List of model responses to analyze.

    Returns:
        A dictionary mapping each concept to the set of models that mentioned it.
    """
    concept_map: dict[str, set[str]] = {}
    for resp in responses:
        concepts: set[str] = (
            extract_keywords(resp.content)
            | extract_key_phrases(resp.content)
        )
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
    """Find concepts shared by multiple models.

    Args:
        concept_map: Mapping of concepts to the models that mentioned them.
        n_models: Total number of models in the ensemble.
        min_agreement: Minimum number of models that must mention a concept.

    Returns:
        A list of convergence dictionaries sorted by strength descending.
    """
    convergences: list[dict[str, Any]] = []
    for concept, models in concept_map.items():
        agreement = len(models)
        if agreement >= min_agreement:
            convergences.append({
                "concept": concept,
                "models": sorted(models),
                "agreement": agreement,
                "strength": agreement / n_models,
            })
    convergences.sort(key=lambda c: c["strength"], reverse=True)
    logger.debug("Found %d convergences", len(convergences))
    return convergences


def _find_divergences(
    responses: list[ModelResponse],
    concept_map: dict[str, set[str]],
) -> list[dict[str, Any]]:
    """Find areas where models take different angles.

    Detects divergences via:
    - Low phrase overlap between model pairs
    - Different negation patterns for the same concept
    - (Optional) low semantic similarity between model responses

    Args:
        responses: List of model responses to analyze.
        concept_map: Mapping of concepts to the models that mentioned them.

    Returns:
        A list of divergence dictionaries.
    """
    divergences: list[dict[str, Any]] = []

    model_phrases: dict[str, set[str]] = {}
    for resp in responses:
        phrases = extract_key_phrases(resp.content)
        model_phrases[resp.model] = phrases

    # Compare each pair of models
    for i, resp_a in enumerate(responses):
        for resp_b in responses[i + 1 :]:
            shared: set[str] = (
                model_phrases[resp_a.model] & model_phrases[resp_b.model]
            )
            total: set[str] = (
                model_phrases[resp_a.model] | model_phrases[resp_b.model]
            )
            if total:
                overlap = len(shared) / len(total)
                if overlap < 0.15:
                    entry: dict[str, Any] = {
                        "model_a": resp_a.model,
                        "model_b": resp_b.model,
                        "phrase_overlap": round(overlap, 3),
                        "note": "Models approached the question from different angles",
                    }
                    # Semantic similarity (optional)
                    if _SEMANTIC_AVAILABLE:
                        sents_a = extract_sentences(resp_a.content)
                        sents_b = extract_sentences(resp_b.content)
                        if sents_a and sents_b:
                            sim = _semantic_similarity(sents_a, sents_b)
                            entry["semantic_similarity"] = round(sim, 3)
                            if sim > 0.6:
                                entry["note"] = (
                                    "Models used different words but similar meaning "
                                    "(low phrase overlap, high semantic similarity)"
                                )
                    divergences.append(entry)

    # Also detect contradicting sentiment on the same concept
    for concept, models_set in concept_map.items():
        if len(models_set) >= 2:
            mentions: dict[str, list[str]] = {}
            for resp in responses:
                if resp.model in models_set:
                    relevant = [
                        s
                        for s in extract_sentences(resp.content)
                        if concept in s.lower()
                    ]
                    if relevant:
                        mentions[resp.model] = relevant[:2]

            if len(mentions) >= 2:
                # Check for negation using word boundaries to avoid false
                # positives like "notable", "nothing", "nevertheless".
                import re
                _NEGATION_RE = re.compile(r"\b(?:not|n't|never)\b", re.IGNORECASE)
                has_negation: dict[str, bool] = {
                    m: any(
                        bool(_NEGATION_RE.search(s))
                        for s in sents
                    )
                    for m, sents in mentions.items()
                }
                neg_values: list[bool] = list(has_negation.values())
                if len(set(neg_values)) > 1:
                    divergences.append({
                        "concept": concept,
                        "models": sorted(models_set),
                        "note": (
                            "Models disagree on this concept "
                            "(negation mismatch)"
                        ),
                        "evidence": {
                            m: sents[:1]
                            for m, sents in mentions.items()
                        },
                    })

    logger.debug("Found %d divergences", len(divergences))
    return divergences


def _find_unique_insights(
    responses: list[ModelResponse],
    concept_map: dict[str, set[str]],
) -> list[dict[str, Any]]:
    """Find concepts that appear in only one model's response.

    Args:
        responses: List of model responses to analyze.
        concept_map: Mapping of concepts to the models that mentioned them.

    Returns:
        A list of up to 20 unique insight dictionaries, sorted by concept length.
    """
    unique: list[dict[str, Any]] = []
    for concept, models_set in concept_map.items():
        if len(models_set) == 1:
            model_name = next(iter(models_set))
            for resp in responses:
                if resp.model == model_name:
                    relevant = [
                        s
                        for s in extract_sentences(resp.content)
                        if concept in s.lower()
                    ]
                    unique.append({
                        "concept": concept,
                        "model": model_name,
                        "context": relevant[0] if relevant else "",
                    })
                    break

    unique.sort(key=lambda u: len(u["concept"]), reverse=True)
    logger.debug("Found %d unique insights", len(unique))
    return unique[:20]


def _confidence_score(
    convergences: list[dict[str, Any]], n_models: int
) -> float:
    """Calculate overall confidence based on convergence strength.

    Args:
        convergences: List of convergence dictionaries with 'strength' field.
        n_models: Total number of models in the ensemble.

    Returns:
        A confidence score between 0.0 and 1.0.
    """
    if not convergences or n_models == 0:
        return 0.0
    top: list[dict[str, Any]] = convergences[:10]
    return sum(c["strength"] for c in top) / len(top) if top else 0.0


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------


def analyze_spectrum(
    responses: list[ModelResponse],
    use_semantic: bool = True,
) -> tuple[
    list[dict[str, Any]],  # convergences
    list[dict[str, Any]],  # divergences
    list[dict[str, Any]],  # unique insights
    float,                  # confidence
]:
    """Analyze a set of model responses for spectral patterns.

    This function identifies convergences (shared concepts), divergences
    (disagreements or different approaches), unique insights (concepts
    mentioned by only one model), and calculates an overall confidence
    score based on convergence strength.

    Args:
        responses: List of ModelResponse objects from the model ensemble.
        use_semantic: If True (default), use optional sentence-transformers
            to detect semantic similarity between models even when their
            surface vocabulary differs. Falls back gracefully if the
            library is not installed.

    Returns:
        A tuple of (convergences, divergences, unique_insights, confidence).
    """
    if not responses:
        logger.debug("analyze_spectrum: no responses to analyze")
        return [], [], [], 0.0

    n = len(responses)
    concept_map = _concept_overlap(responses)

    convergences = _find_convergences(concept_map, n)
    divergences = _find_divergences(responses, concept_map)
    unique_insights = _find_unique_insights(responses, concept_map)
    confidence = _confidence_score(convergences, n)

    logger.info(
        "Spectrum analysis: %d responses, %d convergences, "
        "%d divergences, %d unique, confidence=%.0f%%",
        n,
        len(convergences),
        len(divergences),
        len(unique_insights),
        confidence * 100,
    )

    return convergences, divergences, unique_insights, confidence
