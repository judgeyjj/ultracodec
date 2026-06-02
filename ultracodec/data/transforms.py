"""Audio preprocessing transforms used by UltraCodec datasets."""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio


@dataclass
class AudioTransform:
    """Audio preprocessing pipeline.

    Operations (applied in order):
        1. Resample to ``target_sample_rate``.
        2. Convert to mono (if multi-channel).
        3. Random crop / right-pad (zero) to ``segment_length`` samples.
        4. Peak normalisation to ``peak_target`` (skipped if ``normalize`` is False).
        5. Optional augmentations: noise mixing, volume perturbation.

    Parameters
    ----------
    target_sample_rate:
        The sample rate every output tensor will have.
    segment_length:
        Output length in samples. ``-1`` keeps the full clip.
    normalize:
        Whether to peak-normalise the waveform.
    peak_target:
        Target peak amplitude when ``normalize`` is ``True``.
    augment:
        Whether to apply augmentation (only meaningful in training).
    add_noise_prob, snr_range, noise_paths:
        Configure additive noise augmentation.
    volume_perturb_prob, volume_range:
        Configure volume perturbation augmentation.
    random_crop:
        If ``True`` randomly crops a segment from longer clips; otherwise
        takes from the start (deterministic, useful for evaluation).
    """

    target_sample_rate: int = 16000
    segment_length: int = 48000
    normalize: bool = True
    peak_target: float = 0.95
    augment: bool = False
    add_noise_prob: float = 0.0
    snr_range: Tuple[float, float] = (5.0, 25.0)
    noise_paths: List[str] = field(default_factory=list)
    volume_perturb_prob: float = 0.0
    volume_range: Tuple[float, float] = (0.5, 1.5)
    random_crop: bool = True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def __call__(self, wav: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """Apply the transform to ``wav``.

        Parameters
        ----------
        wav:
            Tensor of shape ``[C, T]`` or ``[T]``.
        sample_rate:
            Original sample rate of ``wav``.

        Returns
        -------
        Tensor of shape ``[1, segment_length]`` (or ``[1, T_resampled]`` when
        ``segment_length`` is ``-1``).
        """
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        if wav.dim() != 2:
            raise ValueError(f"Expected 1D or 2D waveform, got shape {tuple(wav.shape)}")

        wav = self._resample(wav, sample_rate)
        wav = self._to_mono(wav)
        wav = self._fix_length(wav, self.segment_length)

        if self.augment:
            if self.volume_perturb_prob > 0.0 and random.random() < self.volume_perturb_prob:
                wav = self._volume_perturb(wav)
            if self.add_noise_prob > 0.0 and self.noise_paths and random.random() < self.add_noise_prob:
                wav = self._add_noise(wav)

        if self.normalize:
            wav = self._peak_normalize(wav, self.peak_target)
        return wav

    # ------------------------------------------------------------------
    # Building blocks
    # ------------------------------------------------------------------
    def _resample(self, wav: torch.Tensor, sample_rate: int) -> torch.Tensor:
        if sample_rate == self.target_sample_rate:
            return wav
        return torchaudio.functional.resample(wav, sample_rate, self.target_sample_rate)

    @staticmethod
    def _to_mono(wav: torch.Tensor) -> torch.Tensor:
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)
        return wav

    def _fix_length(self, wav: torch.Tensor, length: int) -> torch.Tensor:
        if length is None or length <= 0:
            return wav
        T = wav.size(-1)
        if T == length:
            return wav
        if T > length:
            start = random.randint(0, T - length) if self.random_crop else 0
            return wav[..., start : start + length]
        # Pad on the right with zeros.
        pad = length - T
        return F.pad(wav, (0, pad))

    @staticmethod
    def _peak_normalize(wav: torch.Tensor, peak: float) -> torch.Tensor:
        max_val = wav.abs().max()
        if max_val < 1e-8:
            return wav
        return wav * (peak / max_val)

    def _volume_perturb(self, wav: torch.Tensor) -> torch.Tensor:
        gain = random.uniform(*self.volume_range)
        return wav * gain

    def _add_noise(self, wav: torch.Tensor) -> torch.Tensor:
        noise_path = random.choice(self.noise_paths)
        try:
            noise, sr = torchaudio.load(noise_path)
        except Exception:
            return wav
        noise = self._resample(noise, sr)
        noise = self._to_mono(noise)
        # Match length.
        if noise.size(-1) < wav.size(-1):
            repeats = wav.size(-1) // noise.size(-1) + 1
            noise = noise.repeat(1, repeats)
        noise = noise[..., : wav.size(-1)]

        snr_db = random.uniform(*self.snr_range)
        signal_power = wav.pow(2).mean().clamp_min(1e-10)
        noise_power = noise.pow(2).mean().clamp_min(1e-10)
        snr_linear = 10 ** (snr_db / 10.0)
        scale = torch.sqrt(signal_power / (snr_linear * noise_power))
        return wav + scale * noise


# ----------------------------------------------------------------------
# Functional helpers (used by the dataset code paths that need transforms
# without owning a full ``AudioTransform`` instance).
# ----------------------------------------------------------------------
def random_crop_or_pad(wav: torch.Tensor, length: int, random_offset: bool = True) -> torch.Tensor:
    """Random-crop or zero-pad ``wav`` to ``length`` samples (last dimension)."""
    T = wav.size(-1)
    if T == length:
        return wav
    if T > length:
        start = random.randint(0, T - length) if random_offset else 0
        return wav[..., start : start + length]
    return F.pad(wav, (0, length - T))


def collect_noise_paths(root: Optional[str], extensions: Tuple[str, ...] = (".wav", ".flac")) -> List[str]:
    """Recursively collect noise audio file paths under ``root``."""
    if root is None:
        return []
    root_path = Path(root)
    if not root_path.exists():
        return []
    paths: List[str] = []
    for ext in extensions:
        paths.extend(str(p) for p in root_path.rglob(f"*{ext}"))
    return paths


__all__ = ["AudioTransform", "random_crop_or_pad", "collect_noise_paths"]
