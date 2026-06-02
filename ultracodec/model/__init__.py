"""Model components for UltraCodec.

Provides the full UltraCodec neural speech codec architecture:
    - HTCE: Hierarchical Temporal Compression Encoder (50 → 3.125 Hz)
    - SPQ: Semantic Predictive Quantizer (LM-prior residual VQ)
    - AdaptiveFrameRateGate: Adaptive Frame Rate gating (information-density gated)
    - CascadedDecoder: Cascaded coarse-to-fine decoder
    - UltraCodec: Top-level model composing the above
"""

from .adaptive_framerate import AdaptiveFrameRateGate
from .decoder import CascadedDecoder
from .encoder import HTCE
from .quantizer import ResidualVQ, SPQ
from .ultracodec import UltraCodec

__all__ = [
    "UltraCodec",
    "HTCE",
    "CascadedDecoder",
    "SPQ",
    "ResidualVQ",
    "AdaptiveFrameRateGate",
]
