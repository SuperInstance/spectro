"""Custom exceptions for Spectro — the multi-model cognitive spectrograph.

Every error in the pipeline has a name and a family.
"""

from __future__ import annotations


class SpectroError(Exception):
    """Base exception for all Spectro errors."""


class APIKeyMissing(SpectroError, ValueError):
    """Raised when no API key is found in any source.

    Sources checked (in order):
    1. Explicit constructor parameter
    2. DEEPINFRA_API_KEY / DEEPINFRA_KEY env vars
    3. OPENAI_API_KEY env var
    4. DeepInfra key file (~/.openclaw/.deepinfra-key, ~/.deepinfra-key)
    """


class ModelUnavailable(SpectroError):
    """Raised when a model endpoint returns a non-retryable error.

    Distinguish from AnalysisTimeout: this is a hard failure (e.g., 403, 404),
    not a transient blip.
    """


class AnalysisTimeout(SpectroError):
    """Raised when a model request times out or a retry budget is exhausted."""


class ResponseMalformed(SpectroError):
    """Raised when a model response cannot be parsed (missing fields, etc.)."""
