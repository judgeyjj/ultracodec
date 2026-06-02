"""Evaluation metrics for UltraCodec.

This module provides a unified :class:`MetricCalculator` that wraps a
collection of speech-quality, intelligibility and coding metrics used in
the UltraCodec paper (Section 4.2).

Public API
----------
- :class:`MetricCalculator` -- aggregate calculator with graceful fallbacks
- :class:`UTMOSPredictor`   -- standalone UTMOS MOS predictor wrapper
- Individual scalar helpers: :func:`compute_pesq`, :func:`compute_stoi`,
  :func:`compute_si_sdr`, :func:`compute_lsd`, :func:`compute_mcd`.
"""

from __future__ import annotations

from ultracodec.metrics.evaluation import (  # noqa: F401
    MetricCalculator,
    compute_lsd,
    compute_mcd,
    compute_pesq,
    compute_si_sdr,
    compute_stoi,
)
from ultracodec.metrics.utmos import UTMOSPredictor  # noqa: F401

__all__ = [
    "MetricCalculator",
    "UTMOSPredictor",
    "compute_pesq",
    "compute_stoi",
    "compute_si_sdr",
    "compute_lsd",
    "compute_mcd",
]
