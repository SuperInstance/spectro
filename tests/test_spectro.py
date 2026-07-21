"""Tests for Spectro — the multi-model cognitive spectrograph."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import httpx

from spectro.core import Spectrograph, ModelResponse, SpectrumResult
from spectro.exceptions import (
    APIKeyMissing,
    AnalysisTimeout,
    ModelUnavailable,
    ResponseMalformed,
)
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
        # Remove all environment variables that could provide an API key
        monkeypatch.delenv("DEEPINFRA_API_KEY", raising=False)
        monkeypatch.delenv("DEEPINFRA_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        # Mock the key file reading to return empty
        monkeypatch.setattr(
            Spectrograph, "_read_key_file", staticmethod(lambda: "")
        )
        with pytest.raises(APIKeyMissing, match="No API key"):
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


# ---------------------------------------------------------------------------
# Edge case tests: API errors, network issues, malformed responses
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_api_response(self):
        """Test handling of empty string response from API."""
        r = ModelResponse(model="test", content="", elapsed_ms=100)
        assert not r.ok
        assert r.content == ""

    def test_whitespace_only_response(self):
        """Test handling of whitespace-only response.

        Note: The current implementation treats any non-empty string as OK,
        including whitespace-only strings. This test documents current behavior.
        """
        r = ModelResponse(model="test", content="   \n\t  ", elapsed_ms=100)
        # Whitespace-only content is considered OK by current implementation
        assert r.ok

    def test_malformed_json_response(self):
        """Test handling of malformed JSON in API response."""
        spec = Spectrograph(api_key="test-key")

        async def mock_post(*args, **kwargs):
            class MockResponse:
                status_code = 200
                text = "invalid json"

                def json(self):
                    raise json.JSONDecodeError("Expecting value", "", 0)

                def raise_for_status(self):
                    pass

            return MockResponse()

        mock_client = AsyncMock()
        mock_client.post = mock_post
        result = asyncio.run(spec._query_one(mock_client, "test-model", "hello"))
        assert not result.ok
        # Error message should mention the JSON parsing issue
        assert result.error is not None
        assert len(result.error) > 0

    def test_missing_choices_in_response(self):
        """Test handling of response missing 'choices' field."""
        spec = Spectrograph(api_key="test-key")

        async def mock_post(*args, **kwargs):
            class MockResponse:
                status_code = 200

                def json(self):
                    return {"data": "something"}  # Missing 'choices'

                def raise_for_status(self):
                    pass

            return MockResponse()

        mock_client = AsyncMock()
        mock_client.post = mock_post
        result = asyncio.run(spec._query_one(mock_client, "test-model", "hello"))
        assert not result.ok
        assert "Malformed" in result.error or "choices" in result.error.lower()

    def test_empty_content_response(self):
        """Test handling of empty content response."""
        spec = Spectrograph(api_key="test-key")

        async def mock_post(*args, **kwargs):
            class MockResponse:
                status_code = 200

                def json(self):
                    return {
                        "choices": [
                            {"message": {"content": ""}}
                        ]
                    }

                def raise_for_status(self):
                    pass

            return MockResponse()

        mock_client = AsyncMock()
        mock_client.post = mock_post
        result = asyncio.run(spec._query_one(mock_client, "test-model", "hello"))
        assert result.content == ""
        assert not result.ok  # Empty content should not be OK

    def test_non_retryable_http_error(self):
        """Test handling of non-retryable HTTP error (403)."""
        spec = Spectrograph(api_key="test-key")

        async def mock_post(*args, **kwargs):
            class MockResponse:
                status_code = 403
                text = "Forbidden"

                def raise_for_status(self):
                    raise httpx.HTTPStatusError(
                        "403 Forbidden",
                        request=MagicMock(),
                        response=self
                    )

            return MockResponse()

        mock_client = AsyncMock()
        mock_client.post = mock_post
        result = asyncio.run(spec._query_one(mock_client, "test-model", "hello"))
        assert not result.ok
        assert "403" in result.error or "Forbidden" in result.error

    def test_all_models_fail(self):
        """Test analyze when all models fail permanently."""
        spec = Spectrograph(api_key="test-key")

        async def mock_post(*args, **kwargs):
            # Always fail with non-retryable error
            class MockResponse:
                status_code = 403
                text = "Forbidden"

                def raise_for_status(self):
                    raise httpx.HTTPStatusError(
                        "403 Forbidden",
                        request=MagicMock(),
                        response=self
                    )

            return MockResponse()

        mock_client = AsyncMock()
        mock_client.post = mock_post

        with patch.object(spec, "_get_client", return_value=mock_client):
            result = asyncio.run(spec._query_all("test prompt"))
            # Should have responses but none OK
            assert len(result) > 0
            assert all(not r.ok for r in result)

    def test_unicode_content(self):
        """Test handling of Unicode content in responses."""
        content = "Test with emoji 🎉 and Chinese 你好"
        r = ModelResponse(model="test", content=content, elapsed_ms=100)
        assert r.ok
        assert "🎉" in r.content
        assert "你好" in r.content

    def test_very_long_response(self):
        """Test handling of very long response content."""
        long_content = "word " * 10000  # ~50KB (exactly 50000 chars)
        r = ModelResponse(model="test", content=long_content, elapsed_ms=100)
        assert r.ok
        assert len(r.content) == 50000

    def test_special_characters_in_prompt(self):
        """Test analyze with special characters in prompt."""
        spec = Spectrograph(api_key="test-key", models=["model1"])

        async def mock_post(*args, **kwargs):
            class MockResponse:
                status_code = 200

                def json(self):
                    return {
                        "choices": [
                            {"message": {"content": "Response"}}
                        ],
                        "usage": {"total_tokens": 50}
                    }

                def raise_for_status(self):
                    pass

            return MockResponse()

        with patch("httpx.AsyncClient.post", mock_post):
            # Test various special characters
            for prompt in [
                "Test with 'quotes'",
                'Test with "double quotes"',
                "Test with\nnewlines",
                "Test with\ttabs",
                "Test with $pecial & characters <>",
                "Test with emoji 🎉",
            ]:
                result = spec.analyze(prompt)
                assert result.n_ok == 1

    def test_model_state_restored_after_analyze(self):
        """Test that original model state is restored after analyze with overrides."""
        spec = Spectrograph(api_key="test-key", models=["model1", "model2"])
        original_models = spec.models.copy()
        original_max = spec.max_tokens
        original_temp = spec.temperature

        async def mock_post(*args, **kwargs):
            class MockResponse:
                status_code = 200

                def json(self):
                    return {
                        "choices": [
                            {"message": {"content": "Response"}}
                        ],
                        "usage": {"total_tokens": 50}
                    }

                def raise_for_status(self):
                    pass

            return MockResponse()

        with patch("httpx.AsyncClient.post", mock_post):
            # Analyze with overrides
            spec.analyze(
                "test",
                models=["override-model"],
                max_tokens=5000,
                temperature=0.9
            )

            # State should be restored
            assert spec.models == original_models
            assert spec.max_tokens == original_max
            assert spec.temperature == original_temp

    def test_spectrum_result_to_dict_edge_cases(self):
        """Test to_dict with edge case response content."""
        # Content exactly 500 chars
        content_500 = "x" * 500
        responses = [ModelResponse(model="a", content=content_500, elapsed_ms=1, tokens=10)]
        result = SpectrumResult(prompt="test", responses=responses, confidence=0.5)
        d = result.to_dict()
        assert d["responses"][0]["content"] == content_500  # No truncation

        # Content over 500 chars
        content_600 = "x" * 600
        responses = [ModelResponse(model="a", content=content_600, elapsed_ms=1, tokens=10)]
        result = SpectrumResult(prompt="test", responses=responses, confidence=0.5)
        d = result.to_dict()
        assert "..." in d["responses"][0]["content"]
        assert len(d["responses"][0]["content"]) < 600


# ---------------------------------------------------------------------------
# CLI tests
# ---------------------------------------------------------------------------


class TestCLIOutput:
    """Test CLI output formatting."""

    def test_format_report_basic(self):
        """Test basic report formatting."""
        from spectro.cli import format_report

        responses = [
            ModelResponse(model="model1", content="hello world", elapsed_ms=100),
            ModelResponse(model="model2", content="hello there", elapsed_ms=150),
        ]
        result = SpectrumResult(
            prompt="test",
            responses=responses,
            convergences=[{"concept": "hello", "strength": 1.0, "agreement": 2}],
            divergences=[],
            unique_insights=[],
            confidence=0.8,
            elapsed_ms=200,
        )
        report = format_report(result)
        assert "test" in report
        assert "2/2" in report
        assert "80%" in report
        assert "hello" in report

    def test_format_report_with_errors(self):
        """Test report formatting when some models error."""
        from spectro.cli import format_report

        responses = [
            ModelResponse(model="model1", content="hello", elapsed_ms=100),
            ModelResponse(model="model2", content="", elapsed_ms=150, error="timeout"),
        ]
        result = SpectrumResult(
            prompt="test",
            responses=responses,
            convergences=[],
            divergences=[],
            unique_insights=[],
            confidence=0.5,
            elapsed_ms=200,
        )
        report = format_report(result)
        assert "1/2" in report

        # Verbose mode should show the error
        report_verbose = format_report(result, verbose=True)
        assert "timeout" in report_verbose or "ERROR" in report_verbose
