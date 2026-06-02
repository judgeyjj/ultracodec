"""UTMOS (UTokyo-SaruLab MOS) predictor wrapper.

UTMOS is a non-intrusive MOS prediction model for synthetic / re-synthesised
speech. The original implementation (sarulab/utmos) provides pretrained
weights that consume a 16 kHz waveform and produce a scalar MOS estimate.

This module exposes a thin wrapper :class:`UTMOSPredictor` with two
backends, tried in order:

1. ``speechmos`` package -- a maintained PyPI distribution of UTMOS
   (``pip install speechmos``) exposing :func:`speechmos.utmos22.run`.
2. ``torch.hub`` checkpoint -- ``sarulab-speech/UTMOS22`` (or
   ``tarepan/SpeechMOS``).

If neither backend is available the predictor enters a *disabled* state and
returns ``float('nan')`` from :meth:`predict`. This guarantees the rest of
the evaluation pipeline keeps working even on minimal environments.

The model weights are downloaded lazily on the first call to
:meth:`predict`. The wrapper is intentionally light-weight so it can be
constructed eagerly inside :class:`~ultracodec.metrics.MetricCalculator`.
"""

from __future__ import annotations

import logging
import warnings
from typing import Optional

import numpy as np

logger = logging.getLogger("ultracodec.metrics.utmos")


class UTMOSPredictor:
    """UTokyo-SaruLab MOS predictor.

    Args:
        device: Torch device string (``'cpu'``, ``'cuda'``, ``'cuda:0'`` ...).
            Falls back to CPU if CUDA is unavailable.
        backend: Force a specific backend (``'speechmos'`` or ``'torchhub'``).
            ``None`` lets the wrapper auto-detect.

    Attributes:
        available: Whether the backend was loaded successfully.
        backend: Active backend name, or ``None`` when unavailable.
    """

    def __init__(
        self,
        device: str = "cpu",
        backend: Optional[str] = None,
    ) -> None:
        self.device = device
        self.backend: Optional[str] = None
        self.available: bool = False
        self._model = None
        self._speechmos_module = None

        # Defer torch import to avoid hard dependency at module import time.
        try:
            import torch  # noqa: F401
        except ImportError:
            logger.warning("torch unavailable; UTMOS disabled.")
            return

        if backend in (None, "speechmos"):
            if self._try_init_speechmos():
                return
        if backend in (None, "torchhub"):
            if self._try_init_torchhub():
                return

        logger.warning(
            "UTMOS backend not available; predict() will return NaN. "
            "Install via `pip install speechmos` or ensure torch.hub access."
        )

    # ------------------------------------------------------------------
    # Backend initialisers
    # ------------------------------------------------------------------
    def _try_init_speechmos(self) -> bool:
        try:
            from speechmos import utmos22  # type: ignore[import-not-found]
        except Exception as exc:  # pragma: no cover - import guard
            logger.debug("speechmos backend unavailable: %s", exc)
            return False
        self._speechmos_module = utmos22
        self.backend = "speechmos"
        self.available = True
        logger.info("UTMOS backend initialised: speechmos.utmos22")
        return True

    def _try_init_torchhub(self) -> bool:
        try:
            import torch
        except ImportError:
            return False
        try:
            model = torch.hub.load(
                "tarepan/SpeechMOS:v1.2.0",
                "utmos22_strong",
                trust_repo=True,
            )
        except Exception as exc:  # pragma: no cover - network/cache dependent
            logger.debug("torch.hub UTMOS backend failed: %s", exc)
            return False
        device = self.device if torch.cuda.is_available() or self.device == "cpu" else "cpu"
        self._model = model.to(device).eval()
        self.device = device
        self.backend = "torchhub"
        self.available = True
        logger.info("UTMOS backend initialised: torch.hub (tarepan/SpeechMOS)")
        return True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def predict(self, audio: np.ndarray, sr: int = 16000) -> float:
        """Predict UTMOS for a waveform.

        Args:
            audio: 1-D float waveform in ``[-1, 1]`` range.
            sr: Sampling rate of ``audio`` (Hz).

        Returns:
            Estimated MOS in ``[1, 5]``, or ``float('nan')`` when the
            backend is unavailable.
        """
        if not self.available:
            return float("nan")

        audio = np.asarray(audio, dtype=np.float32).squeeze()
        if audio.ndim != 1:
            audio = audio.reshape(-1)

        try:
            if self.backend == "speechmos":
                result = self._speechmos_module.run(audio, sr)
                # speechmos returns {'utmos': float}
                if isinstance(result, dict):
                    return float(result.get("utmos", result.get("UTMOS", float("nan"))))
                return float(result)

            if self.backend == "torchhub":
                import torch

                wav = torch.from_numpy(audio).float().unsqueeze(0).to(self.device)
                with torch.no_grad():
                    score = self._model(wav, sr)
                if torch.is_tensor(score):
                    score = score.detach().squeeze().cpu().item()
                return float(score)
        except Exception as exc:
            warnings.warn(f"UTMOS prediction failed: {exc}")
            return float("nan")

        return float("nan")
