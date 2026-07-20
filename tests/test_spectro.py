"""Tests for Spectro — the multi-model cognitive spectrograph."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from spectro.core import Spectrograph, ModelResponse, SpectrumResult
from spectro.analysis import (
    analyze_spectrum,
    extract_keywords,
    extract_key_phrases,
    tokenize,
)


# ---------------------------------------------------------------------------
# Core dataclass tests
# ---------------------------------------------------------------------------

class TestModelResponse:
    def test_ok_with_content(self):
        r = ModelResponse(model="test", content="hello", elapsed_ms=100)
        assert r.ok

    def test_not_ok_with_error(self):
        r = ModelResponse(model="test", content="hello", elapsed_ms=100, error="timeout")
        assert not r.ok

    def test_not_ok_empty(self):
        r = ModelResponse(model="test", content="", elapsed_ms=100)
        assert not r.ok


class TestSpectrumResult:
    def test_n_models(self):
        responses = [
            ModelResponse(model="a", content="x", elapsed_ms=1),
            ModelResponse(model="b", content="y", elapsed_ms=1),
        ]
        result = SpectrumResult(prompt="test", responses=responses)
        assert result.n_models == 2
        assert result.n_ok == 2

    def test_summary(self):
        responses = [
            ModelResponse(model="a", content="hello world", elapsed_ms=1),
            ModelResponse(model="b", content="hello there", elapsed_ms=1),
        ]
        result = SpectrumResult(
            prompt="test",
            responses=responses,
            convergences=[{"concept": "hello", "strength": 1.0}],
            divergences=[],
            unique_insights=[],
            confidence=0.8,
        )
        s = result.summary()
        assert "2/2" in s
        assert "80%" in s

    def test_to_dict(self):
        responses = [ModelResponse(model="a", content="x", elapsed_ms=1, tokens=10)]
        result = SpectrumResult(prompt="test", responses=responses, confidence=0.5)
        d = result.to_dict()
        assert d["prompt"] == "test"
        assert d["confidence"] == 0.5
        assert len(d["responses"]) == 1


# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------

class TestTokenize:
    def test_basic(self):
        assert "hello" in tokenize("Hello, world!")

    def test_min_length(self):
        tokens = tokenize("a an the cat dog")
        assert "cat" in tokens
        assert "dog" in tokens
        assert "an" not in tokens  # 2 chars, filtered by min length 3

    def test_lowercase(self):
        assert all(t == t.lower() for t in tokenize("HELLO World"))

    def test_empty(self):
        assert tokenize("") == []


class TestKeywords:
    def test_filters_stopwords(self):
        kws = extract_keywords("the quick brown fox jumps over the lazy dog")
        assert "quick" in kws or "brown" in kws
        assert "the" not in kws
        assert "over" not in kws

    def test_min_length(self):
        kws = extract_keywords("cat dog xylophone ab")
        assert "xylophone" in kws
        assert "ab" not in kws  # too short (2 chars)

    def test_returns_set(self):
        assert isinstance(extract_keywords("hello world"), set)


class TestKeyPhrases:
    def test_bigrams(self):
        phrases = extract_key_phrases("machine learning models")
        assert "machine learning" in phrases

    def test_trigrams(self):
        phrases = extract_key_phrases("the neural network architecture")
        assert "neural network architecture" in phrases

    def test_filters_stopword_phrases(self):
        phrases = extract_key_phrases("the and but")
        assert len(phrases) == 0


# ---------------------------------------------------------------------------
# Analysis tests
# ---------------------------------------------------------------------------

class TestAnalyzeSpectrum:
    def _make_responses(self, *contents):
        return [
            ModelResponse(model=f"model_{i}", content=c, elapsed_ms=1)
            for i, c in enumerate(contents)
        ]

    def test_identical_responses_high_confidence(self):
        text = "Clean code is essential for maintainability and readability"
        responses = self._make_responses(text, text, text)
        conv, div, uniq, conf = analyze_spectrum(responses)
        assert conf > 0.5
        assert len(conv) > 0

    def test_completely_different_low_confidence(self):
        r1 = "quantum entanglement photon polarization bell inequality"
        r2 = "pasta recipes italian cooking flour semolina"
        r3 = "medieval architecture gothic cathedrals flying buttresses"
        responses = self._make_responses(r1, r2, r3)
        conv, div, uniq, conf = analyze_spectrum(responses)
        assert conf < 0.3
        assert len(uniq) > 0

    def test_unique_insights(self):
        r1 = "maintainability testing documentation"
        r2 = "maintainability testing performance"
        r3 = "maintainability testing security"
        responses = self._make_responses(r1, r2, r3)
        conv, div, uniq, conf = analyze_spectrum(responses)
        # "documentation", "performance", "security" should be unique
        unique_concepts = {u["concept"] for u in uniq}
        assert "documentation" in unique_concepts
        assert "performance" in unique_concepts
        assert "security" in unique_concepts

    def test_empty_responses(self):
        conv, div, uniq, conf = analyze_spectrum([])
        assert conv == []
        assert conf == 0.0

    def test_single_response(self):
        responses = self._make_responses("code quality matters")
        conv, div, uniq, conf = analyze_spectrum(responses)
        # With one model, everything is unique, confidence should be low-medium
        assert len(uniq) > 0

    def test_convergence_detection(self):
        shared = "architecture performance scalability"
        r1 = f"{shared} documentation deployment"
        r2 = f"{shared} monitoring logging"
        r3 = f"{shared} testing security"
        responses = self._make_responses(r1, r2, r3)
        conv, div, uniq, conf = analyze_spectrum(responses)

        # Shared concepts should be in convergences
        conv_concepts = {c["concept"] for c in conv}
        assert "architecture" in conv_concepts
        assert "performance" in conv_concepts
        assert "scalability" in conv_concepts

        # Verify agreement counts
        for c in conv:
            if c["concept"] in ("architecture", "performance", "scalability"):
                assert c["agreement"] == 3


# ---------------------------------------------------------------------------
# Spectrograph initialization tests
# ---------------------------------------------------------------------------

class TestSpectrograph:
    def test_init_with_api_key(self):
        spec = Spectrograph(api_key="test-key")
        assert spec.api_key == "test-key"
        assert len(spec.models) == 5  # default ensemble

    def test_init_with_custom_models(self):
        models = ["model-a", "model-b"]
        spec = Spectrograph(api_key="test-key", models=models)
        assert spec.models == models

    def test_init_without_key_raises(self, monkeypatch):
        monkeypatch.delenv("DEEPINFRA_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        # Also need to handle key file
        monkeypatch.setattr(
            Spectrograph, "_read_key_file", staticmethod(lambda: "")
        )
        with pytest.raises(ValueError, match="No API key"):
            Spectrograph()

    def test_read_key_file_from_env(self, monkeypatch, tmp_path):
        key_file = tmp_path / "key"
        key_file.write_text("secret-key-123")
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.delenv("DEEPINFRA_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        # Create .openclaw dir
        openclaw_dir = tmp_path / ".openclaw"
        openclaw_dir.mkdir()
        (openclaw_dir / ".deepinfra-key").write_text("file-key-456")

        key = Spectrograph._read_key_file()
        assert key == "file-key-456"
