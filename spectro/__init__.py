"""Spectro — Multi-Model Cognitive Spectrograph.

Split the beam. Read the spectrum. The convergences are the coastline.
The divergences are the interesting water.

Quick start:
    from spectro import Spectrograph
    spec = Spectrograph()
    result = spec.analyze("What makes a good API?")
"""

from spectro.core import Spectrograph, SpectrumResult, ModelResponse

try:
    from spectro.analysis import analyze_spectrum
except ImportError:
    analyze_spectrum = None

__version__ = "0.1.0"
__author__ = "SuperInstance"
__license__ = "MIT"

__all__ = [
    "Spectrograph",
    "SpectrumResult",
    "ModelResponse",
    "analyze_spectrum",
]
