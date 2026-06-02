"""Unified evaluation metric calculator for UltraCodec.

Implements the metrics listed in paper3_ultracodec.md Section 4.2:

Reconstruction quality:
    - PESQ (Perceptual Evaluation of Speech Quality, wide-band)
    - STOI (Short-Time Objective Intelligibility)
    - ViSQOL (Virtual Speech Quality Objective Listener)
    - UTMOS (UTokyo-SaruLab MOS predictor)
    - MCD (Mel Cepstral Distortion)
    - SI-SDR (Scale-Invariant SDR)
    - LSD (Log-Spectral Distance)

Linguistic preservation:
    - WER via Whisper round-trip ASR

Coding efficiency:
    - bitrate (kbps)
    - frame_rate (Hz)

Every metric is wrapped in defensive ``try/except`` blocks: if the
underlying library is missing, the metric returns ``float('nan')`` and a
warning is logged exactly once. This means the pipeline can run on
machines that only have a subset of dependencies installed.
"""

from __future__ import annotations

import logging
import math
import shutil
import subprocess
import tempfile
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger("ultracodec.metrics.evaluation")

# Track which optional dependencies have already issued a warning so the
# evaluation log does not get spammed.
_WARNED: set = set()


def _warn_once(key: str, message: str) -> None:
    if key in _WARNED:
        return
    _WARNED.add(key)
    logger.warning(message)


# ---------------------------------------------------------------------------
# Standalone metric helpers (also re-exported from package __init__).
# ---------------------------------------------------------------------------
def compute_pesq(ref: np.ndarray, deg: np.ndarray, sr: int = 16000) -> float:
    """Wide-band PESQ (ITU-T P.862.2).

    Args:
        ref: Reference (clean) waveform, 1-D float32.
        deg: Degraded (codec output) waveform, 1-D float32.
        sr: Sampling rate. PESQ supports 8 kHz (nb) and 16 kHz (wb).

    Returns:
        PESQ score (typically 1.0--4.5) or ``nan`` if computation fails.
    """
    try:
        from pesq import pesq as _pesq  # type: ignore[import-not-found]
    except ImportError:
        _warn_once("pesq", "pesq library not installed; skipping PESQ.")
        return float("nan")

    ref = np.asarray(ref, dtype=np.float32).squeeze()
    deg = np.asarray(deg, dtype=np.float32).squeeze()
    n = min(len(ref), len(deg))
    if n == 0:
        return float("nan")
    ref, deg = ref[:n], deg[:n]
    mode = "wb" if sr >= 16000 else "nb"
    try:
        return float(_pesq(sr, ref, deg, mode))
    except Exception as exc:
        warnings.warn(f"PESQ failed: {exc}")
        return float("nan")


def compute_stoi(ref: np.ndarray, deg: np.ndarray, sr: int = 16000) -> float:
    """Short-Time Objective Intelligibility (Taal et al., 2010).

    Returns the standard (not extended) STOI in ``[0, 1]``.
    """
    try:
        from pystoi import stoi as _stoi  # type: ignore[import-not-found]
    except ImportError:
        _warn_once("pystoi", "pystoi library not installed; skipping STOI.")
        return float("nan")

    ref = np.asarray(ref, dtype=np.float32).squeeze()
    deg = np.asarray(deg, dtype=np.float32).squeeze()
    n = min(len(ref), len(deg))
    if n == 0:
        return float("nan")
    try:
        return float(_stoi(ref[:n], deg[:n], sr, extended=False))
    except Exception as exc:
        warnings.warn(f"STOI failed: {exc}")
        return float("nan")


def compute_si_sdr(ref: np.ndarray, deg: np.ndarray) -> float:
    """Scale-Invariant Signal-to-Distortion Ratio (Le Roux et al., 2019)."""
    ref = np.asarray(ref, dtype=np.float64).squeeze()
    deg = np.asarray(deg, dtype=np.float64).squeeze()
    n = min(len(ref), len(deg))
    if n == 0:
        return float("nan")
    ref = ref[:n] - ref[:n].mean()
    deg = deg[:n] - deg[:n].mean()
    ref_energy = np.dot(ref, ref) + 1e-12
    alpha = np.dot(deg, ref) / ref_energy
    target = alpha * ref
    noise = deg - target
    num = np.dot(target, target) + 1e-12
    den = np.dot(noise, noise) + 1e-12
    return float(10.0 * np.log10(num / den))


def compute_lsd(ref: np.ndarray, deg: np.ndarray, sr: int = 16000,
                n_fft: int = 2048, hop: int = 512) -> float:
    """Log-Spectral Distance averaged over time (dB)."""
    try:
        import librosa  # type: ignore[import-not-found]
    except ImportError:
        _warn_once("librosa-lsd", "librosa not installed; skipping LSD.")
        return float("nan")

    ref = np.asarray(ref, dtype=np.float32).squeeze()
    deg = np.asarray(deg, dtype=np.float32).squeeze()
    n = min(len(ref), len(deg))
    if n == 0:
        return float("nan")
    try:
        S_ref = np.abs(librosa.stft(ref[:n], n_fft=n_fft, hop_length=hop)) ** 2
        S_deg = np.abs(librosa.stft(deg[:n], n_fft=n_fft, hop_length=hop)) ** 2
        log_ref = np.log10(S_ref + 1e-10)
        log_deg = np.log10(S_deg + 1e-10)
        # 10 * sqrt(mean over freq of squared diff), then mean over frames.
        per_frame = np.sqrt(np.mean((log_ref - log_deg) ** 2, axis=0))
        return float(10.0 * np.mean(per_frame))
    except Exception as exc:
        warnings.warn(f"LSD failed: {exc}")
        return float("nan")


def compute_mcd(ref: np.ndarray, deg: np.ndarray, sr: int = 16000,
                n_mfcc: int = 13) -> float:
    """Mel-Cepstral Distortion (dB).

    Implementation: extract MFCCs (excluding c0), then average frame-wise
    Euclidean distance, scaled by ``10 / log(10) * sqrt(2)``.
    """
    try:
        import librosa  # type: ignore[import-not-found]
    except ImportError:
        _warn_once("librosa-mcd", "librosa not installed; skipping MCD.")
        return float("nan")

    ref = np.asarray(ref, dtype=np.float32).squeeze()
    deg = np.asarray(deg, dtype=np.float32).squeeze()
    n = min(len(ref), len(deg))
    if n == 0:
        return float("nan")
    try:
        mfcc_ref = librosa.feature.mfcc(y=ref[:n], sr=sr, n_mfcc=n_mfcc)[1:]
        mfcc_deg = librosa.feature.mfcc(y=deg[:n], sr=sr, n_mfcc=n_mfcc)[1:]
        T = min(mfcc_ref.shape[1], mfcc_deg.shape[1])
        if T == 0:
            return float("nan")
        diff = mfcc_ref[:, :T] - mfcc_deg[:, :T]
        per_frame = np.sqrt(np.sum(diff ** 2, axis=0))
        scale = (10.0 / math.log(10.0)) * math.sqrt(2.0)
        return float(scale * np.mean(per_frame))
    except Exception as exc:
        warnings.warn(f"MCD failed: {exc}")
        return float("nan")


# ---------------------------------------------------------------------------
# Unified calculator.
# ---------------------------------------------------------------------------
class MetricCalculator:
    """Unified evaluation metric calculator.

    The calculator lazily initialises heavy backends (Whisper, UTMOS) so
    construction is cheap. Any metric whose backend is unavailable returns
    ``float('nan')`` and logs a single warning.

    Args:
        sample_rate: Default sampling rate when not supplied per-call.
        device: Torch device string used by Whisper / UTMOS backends.
        whisper_model: HuggingFace identifier for Whisper ASR (used by
            :meth:`compute_wer`). Examples: ``'openai/whisper-small'``,
            ``'openai/whisper-large-v3'``.
        utmos_backend: Force a specific UTMOS backend (see
            :class:`UTMOSPredictor`).
        enable_utmos: Skip UTMOS initialisation entirely if False.
        enable_whisper: Skip Whisper initialisation entirely if False.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        device: str = "cpu",
        whisper_model: str = "openai/whisper-small",
        utmos_backend: Optional[str] = None,
        enable_utmos: bool = True,
        enable_whisper: bool = True,
    ) -> None:
        self.sample_rate = sample_rate
        self.device = device
        self.whisper_model_name = whisper_model
        self._utmos_predictor = None
        self._utmos_backend = utmos_backend
        self._enable_utmos = enable_utmos
        self._enable_whisper = enable_whisper
        self._whisper_pipe = None

    # ------------------------------------------------------------------
    # Lazy backend accessors
    # ------------------------------------------------------------------
    def _utmos(self):
        if not self._enable_utmos:
            return None
        if self._utmos_predictor is None:
            from ultracodec.metrics.utmos import UTMOSPredictor

            self._utmos_predictor = UTMOSPredictor(
                device=self.device, backend=self._utmos_backend
            )
        return self._utmos_predictor

    def _whisper(self):
        if not self._enable_whisper:
            return None
        if self._whisper_pipe is not None:
            return self._whisper_pipe
        try:
            import torch
            from transformers import pipeline  # type: ignore[import-not-found]
        except ImportError:
            _warn_once(
                "whisper",
                "transformers not installed; WER will be skipped.",
            )
            self._enable_whisper = False
            return None
        try:
            device_arg = 0 if (self.device != "cpu" and torch.cuda.is_available()) else -1
            self._whisper_pipe = pipeline(
                task="automatic-speech-recognition",
                model=self.whisper_model_name,
                device=device_arg,
                chunk_length_s=30,
            )
        except Exception as exc:  # pragma: no cover - network/IO
            _warn_once("whisper-load", f"Whisper failed to load: {exc}")
            self._enable_whisper = False
            return None
        return self._whisper_pipe

    # ------------------------------------------------------------------
    # Reconstruction-quality metrics
    # ------------------------------------------------------------------
    def compute_pesq(self, ref: np.ndarray, deg: np.ndarray, sr: int = 16000) -> float:
        """Wide-band PESQ (see :func:`compute_pesq`)."""
        return compute_pesq(ref, deg, sr)

    def compute_stoi(self, ref: np.ndarray, deg: np.ndarray, sr: int = 16000) -> float:
        """STOI (see :func:`compute_stoi`)."""
        return compute_stoi(ref, deg, sr)

    def compute_si_sdr(self, ref: np.ndarray, deg: np.ndarray) -> float:
        """SI-SDR (see :func:`compute_si_sdr`)."""
        return compute_si_sdr(ref, deg)

    def compute_lsd(self, ref: np.ndarray, deg: np.ndarray, sr: int = 16000) -> float:
        """LSD (see :func:`compute_lsd`)."""
        return compute_lsd(ref, deg, sr)

    def compute_mcd(self, ref: np.ndarray, deg: np.ndarray, sr: int = 16000) -> float:
        """MCD (see :func:`compute_mcd`)."""
        return compute_mcd(ref, deg, sr)

    # ------------------------------------------------------------------
    # ViSQOL
    # ------------------------------------------------------------------
    def compute_visqol(self, ref_path: str, deg_path: str) -> float:
        """ViSQOL (Virtual Speech Quality Objective Listener).

        Tries three backends, in order:
            1. Python binding ``visqol`` (Google).
            2. Command-line tool ``visqol`` on PATH.
            3. Fallback: returns PESQ on the supplied audio files (logs warning).
        """
        # Backend 1: native python binding
        try:
            from visqol import visqol_lib_py  # type: ignore[import-not-found]
            from visqol.pb2 import (  # type: ignore[import-not-found]
                similarity_result_pb2,
                visqol_config_pb2,
            )
            import soundfile as sf

            config = visqol_config_pb2.VisqolConfig()
            config.audio.sample_rate = 16000
            config.options.use_speech_scoring = True
            config.options.svr_model_path = (
                visqol_lib_py.VisqolManager.GetDefaultSpeechSVRModel()
            )
            api = visqol_lib_py.VisqolApi()
            api.Create(config)

            ref, sr_ref = sf.read(ref_path)
            deg, sr_deg = sf.read(deg_path)
            ref = ref.astype(np.float64)
            deg = deg.astype(np.float64)
            result = api.Measure(ref, deg)
            return float(result.moslqo)
        except ImportError:
            pass
        except Exception as exc:
            warnings.warn(f"ViSQOL python binding failed: {exc}")

        # Backend 2: command-line tool
        if shutil.which("visqol") is not None:
            try:
                proc = subprocess.run(
                    [
                        "visqol",
                        "--reference_file", str(ref_path),
                        "--degraded_file", str(deg_path),
                        "--use_speech_mode",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=120,
                    check=False,
                )
                # Parse "MOS-LQO: 4.123" line from stdout.
                for line in proc.stdout.splitlines():
                    if "MOS-LQO" in line or "moslqo" in line.lower():
                        token = line.split()[-1].strip(",;")
                        try:
                            return float(token)
                        except ValueError:
                            continue
            except Exception as exc:  # pragma: no cover
                warnings.warn(f"ViSQOL CLI failed: {exc}")

        # Backend 3: fallback to PESQ
        _warn_once(
            "visqol",
            "ViSQOL unavailable; falling back to PESQ as a proxy MOS estimate.",
        )
        try:
            import soundfile as sf

            ref, sr_ref = sf.read(ref_path)
            deg, sr_deg = sf.read(deg_path)
            sr = sr_ref if sr_ref == sr_deg else 16000
            return compute_pesq(ref, deg, sr)
        except Exception:
            return float("nan")

    # ------------------------------------------------------------------
    # UTMOS
    # ------------------------------------------------------------------
    def compute_utmos(self, audio: np.ndarray, sr: int = 16000) -> float:
        """UTMOS MOS prediction.

        Uses :class:`UTMOSPredictor` with ``speechmos`` or torch.hub backend.
        """
        predictor = self._utmos()
        if predictor is None or not predictor.available:
            return float("nan")
        return predictor.predict(audio, sr=sr)

    # ------------------------------------------------------------------
    # WER via Whisper round-trip
    # ------------------------------------------------------------------
    def compute_wer(
        self,
        ref_audio: np.ndarray,
        deg_audio: np.ndarray,
        sr: int = 16000,
    ) -> float:
        """Word Error Rate between Whisper transcripts of ref and deg.

        Procedure:
            1. Transcribe ``ref_audio`` (clean) with Whisper.
            2. Transcribe ``deg_audio`` (codec output) with Whisper.
            3. Compute WER between the two transcripts.
        """
        pipe = self._whisper()
        if pipe is None:
            return float("nan")
        try:
            ref_text = self._transcribe(pipe, ref_audio, sr)
            deg_text = self._transcribe(pipe, deg_audio, sr)
            return _word_error_rate(ref_text, deg_text)
        except Exception as exc:
            warnings.warn(f"WER computation failed: {exc}")
            return float("nan")

    @staticmethod
    def _transcribe(pipe: Any, audio: np.ndarray, sr: int) -> str:
        audio = np.asarray(audio, dtype=np.float32).squeeze()
        out = pipe({"array": audio, "sampling_rate": sr})
        if isinstance(out, dict):
            return str(out.get("text", "")).strip().lower()
        return str(out).strip().lower()

    # ------------------------------------------------------------------
    # Coding-side metrics
    # ------------------------------------------------------------------
    def compute_bitrate(self, model: Any, audio: Any) -> float:
        """Compute actual bitrate (kbps) for ``audio`` through ``model``.

        Args:
            model: An :class:`UltraCodec` instance providing :meth:`encode`
                and ``sample_rate``.
            audio: Tensor ``[1, 1, T]`` waveform on the model's device.

        Returns:
            Bitrate in kbps. Falls back to ``nan`` on failure.
        """
        try:
            import torch  # noqa: F401

            with __import__("torch").no_grad():
                result = model.encode(audio)
            codes = result["codes"]
            gate = result.get("gate", None)
            sr = getattr(model, "sample_rate", self.sample_rate)
            audio_seconds = audio.shape[-1] / float(sr)
            if hasattr(model, "get_bitrate"):
                return float(model.get_bitrate(audio_seconds, codes, gate=gate))

            # Manual fallback identical to UltraCodec.get_bitrate
            num_codebooks = codes.size(1)
            seq_len = codes.size(2)
            cb_size = (
                model.config.get("quantizer", {}).get("codebook_size", 1024)
                if hasattr(model, "config")
                else 1024
            )
            bits_per_code = math.log2(cb_size)
            if gate is not None:
                effective = (gate > 0.5).float().sum(dim=-1).mean().item()
            else:
                effective = seq_len
            total_bits = effective * num_codebooks * bits_per_code
            return float(total_bits / (audio_seconds * 1000.0))
        except Exception as exc:
            warnings.warn(f"compute_bitrate failed: {exc}")
            return float("nan")

    def compute_frame_rate(self, model: Any, audio: Any) -> float:
        """Compute effective frame rate (Hz) after AFR gating."""
        try:
            with __import__("torch").no_grad():
                result = model.encode(audio)
            codes = result["codes"]
            gate = result.get("gate", None)
            sr = getattr(model, "sample_rate", self.sample_rate)
            audio_seconds = audio.shape[-1] / float(sr)
            if gate is not None:
                effective = (gate > 0.5).float().sum(dim=-1).mean().item()
            else:
                effective = codes.size(2)
            return float(effective / audio_seconds)
        except Exception as exc:
            warnings.warn(f"compute_frame_rate failed: {exc}")
            return float("nan")

    # ------------------------------------------------------------------
    # Convenience: compute everything available.
    # ------------------------------------------------------------------
    def compute_all(
        self,
        ref: np.ndarray,
        deg: np.ndarray,
        sr: int = 16000,
        ref_path: Optional[str] = None,
        deg_path: Optional[str] = None,
        metrics: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """Compute all reconstruction-quality metrics.

        Args:
            ref: Reference 1-D waveform.
            deg: Degraded 1-D waveform.
            sr: Sampling rate.
            ref_path: Optional path to ``ref`` on disk (required for ViSQOL).
            deg_path: Optional path to ``deg`` on disk (required for ViSQOL).
            metrics: Optional whitelist of metric names. If ``None`` all
                supported metrics are computed.

        Returns:
            Dict ``{metric_name: float}``. Missing metrics yield ``nan``.
        """
        wanted = set(metrics) if metrics else None
        out: Dict[str, float] = {}

        def maybe(name: str, fn) -> None:
            if wanted is not None and name not in wanted:
                return
            out[name] = fn()

        maybe("pesq", lambda: self.compute_pesq(ref, deg, sr))
        maybe("stoi", lambda: self.compute_stoi(ref, deg, sr))
        maybe("si_sdr", lambda: self.compute_si_sdr(ref, deg))
        maybe("lsd", lambda: self.compute_lsd(ref, deg, sr))
        maybe("mcd", lambda: self.compute_mcd(ref, deg, sr))
        maybe("utmos", lambda: self.compute_utmos(deg, sr))

        # ViSQOL needs files; create temporary ones if not supplied.
        if wanted is None or "visqol" in wanted:
            cleanup: List[Path] = []
            try:
                if ref_path is None or deg_path is None:
                    try:
                        import soundfile as sf

                        tmp_ref = Path(tempfile.mkstemp(suffix=".wav")[1])
                        tmp_deg = Path(tempfile.mkstemp(suffix=".wav")[1])
                        sf.write(tmp_ref, ref, sr)
                        sf.write(tmp_deg, deg, sr)
                        ref_path = str(tmp_ref)
                        deg_path = str(tmp_deg)
                        cleanup.extend([tmp_ref, tmp_deg])
                    except Exception:
                        out["visqol"] = float("nan")
                        ref_path = None
                if ref_path is not None and deg_path is not None:
                    out["visqol"] = self.compute_visqol(ref_path, deg_path)
            finally:
                for p in cleanup:
                    try:
                        p.unlink(missing_ok=True)
                    except Exception:
                        pass

        if wanted is None or "wer" in wanted:
            out["wer"] = self.compute_wer(ref, deg, sr)

        return out


# ---------------------------------------------------------------------------
# Word Error Rate utility (Levenshtein on tokens).
# ---------------------------------------------------------------------------
def _word_error_rate(ref_text: str, hyp_text: str) -> float:
    """Token-level WER between two transcripts.

    Implements the standard Levenshtein edit distance over whitespace
    tokens. Returns ``0.0`` when the reference is empty *and* the hypothesis
    matches it; otherwise returns ``1.0`` when reference is empty but
    hypothesis is not.
    """
    ref_tokens = ref_text.split()
    hyp_tokens = hyp_text.split()
    R, H = len(ref_tokens), len(hyp_tokens)
    if R == 0:
        return 0.0 if H == 0 else 1.0

    prev = list(range(H + 1))
    for i in range(1, R + 1):
        curr = [i] + [0] * H
        for j in range(1, H + 1):
            cost = 0 if ref_tokens[i - 1] == hyp_tokens[j - 1] else 1
            curr[j] = min(
                prev[j] + 1,        # deletion
                curr[j - 1] + 1,    # insertion
                prev[j - 1] + cost,  # substitution
            )
        prev = curr
    return float(prev[H] / R)
