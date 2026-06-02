"""Dataset classes for UltraCodec.

This module provides three datasets:

* :class:`VCTKDataset` — wraps an existing VCTK split (the user already has the
  data downloaded and split into ``train``/``train_test``/``test`` subdirs).
* :class:`LibriSpeechDataset` — wraps LibriSpeech with optional automatic
  download via :mod:`ultracodec.data.download`.
* :class:`MixedDataset` — concatenates several datasets and optionally samples
  proportionally to user-specified weights.
"""
from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Union

import torch
import torchaudio
from torch.utils.data import Dataset

from ultracodec.data.transforms import AudioTransform

logger = logging.getLogger(__name__)

AUDIO_EXTS = (".wav", ".flac", ".mp3", ".ogg")


# ----------------------------------------------------------------------
# Internal helpers
# ----------------------------------------------------------------------
def _list_audio_files(root: Path, extensions: Sequence[str] = AUDIO_EXTS) -> List[Path]:
    """Recursively list audio files under ``root``."""
    files: List[Path] = []
    for ext in extensions:
        files.extend(root.rglob(f"*{ext}"))
    return sorted(files)


def _safe_load(path: Union[str, Path]) -> tuple[torch.Tensor, int]:
    """Load an audio file, returning ``(wav[1, T], sample_rate)``.

    Falls back to ``soundfile`` if torchaudio cannot decode the file.
    """
    try:
        wav, sr = torchaudio.load(str(path))
    except Exception:  # pragma: no cover - codec edge cases
        import soundfile as sf

        data, sr = sf.read(str(path), dtype="float32", always_2d=False)
        wav = torch.from_numpy(data)
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        else:
            wav = wav.transpose(0, 1)
    return wav, sr


# ----------------------------------------------------------------------
# VCTKDataset
# ----------------------------------------------------------------------
class VCTKDataset(Dataset):
    """VCTK dataset wrapper.

    Expects ``root_dir`` to contain pre-split subdirectories
    (``train``/``train_test``/``test``). Audio files are discovered
    recursively under ``root_dir/<split>``.

    Parameters
    ----------
    root_dir:
        Root directory of the VCTK corpus
        (e.g. ``/data01/audio_group/m24_yuanjiajun/AP-BWE/VCTK-Corpus-0.92/wav_test``).
    split:
        One of ``train``, ``train_test``, ``test``.
    transform:
        Optional :class:`AudioTransform`. If ``None`` a default transform
        with the requested ``sample_rate`` and ``segment_length`` is used.
    sample_rate, segment_length:
        Used only when ``transform`` is ``None``.
    augment:
        Whether to enable augmentation in the default transform.
    """

    DEFAULT_ROOT = "/data01/audio_group/m24_yuanjiajun/AP-BWE/VCTK-Corpus-0.92/wav_test"
    SUPPORTED_SPLITS = ("train", "train_test", "test")

    def __init__(
        self,
        root_dir: Union[str, Path] = DEFAULT_ROOT,
        split: str = "train",
        transform: Optional[AudioTransform] = None,
        sample_rate: int = 16000,
        segment_length: int = 48000,
        augment: bool = False,
        extensions: Sequence[str] = AUDIO_EXTS,
    ) -> None:
        if split not in self.SUPPORTED_SPLITS:
            raise ValueError(
                f"Invalid split '{split}'. Expected one of {self.SUPPORTED_SPLITS}."
            )
        self.root_dir = Path(root_dir)
        self.split = split
        self.split_dir = self.root_dir / split
        if not self.split_dir.exists():
            logger.warning(
                "VCTK split directory %s does not exist. The dataset will be empty.",
                self.split_dir,
            )
            self.files: List[Path] = []
        else:
            self.files = _list_audio_files(self.split_dir, extensions)
        if transform is None:
            transform = AudioTransform(
                target_sample_rate=sample_rate,
                segment_length=segment_length,
                augment=augment,
                random_crop=split.startswith("train"),
            )
        self.transform = transform

    # PyTorch Dataset API ------------------------------------------------
    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict:
        path = self.files[idx]
        wav, sr = _safe_load(path)
        wav = self.transform(wav, sr)
        return {
            "wav": wav,
            "path": str(path),
            "dataset": "vctk",
            "split": self.split,
        }


# ----------------------------------------------------------------------
# LibriSpeechDataset
# ----------------------------------------------------------------------
class LibriSpeechDataset(Dataset):
    """LibriSpeech dataset wrapper.

    The data is expected to live under
    ``root_dir/LibriSpeech/<split>``. When ``download=True`` and the split
    is missing, :func:`ultracodec.data.download.download_librispeech` is
    invoked to fetch it.

    Parameters
    ----------
    root_dir:
        Directory holding the ``LibriSpeech`` tree.
    split:
        Any of LibriSpeech's official splits, e.g. ``train-clean-100``,
        ``train-clean-360``, ``dev-clean``, ``test-clean`` …
    download:
        Whether to attempt automatic download when the split is absent.
    """

    def __init__(
        self,
        root_dir: Union[str, Path],
        split: str = "train-clean-360",
        download: bool = True,
        transform: Optional[AudioTransform] = None,
        sample_rate: int = 16000,
        segment_length: int = 48000,
        augment: bool = False,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.split = split
        self.download = download
        self._maybe_download()

        split_dir = self.root_dir / "LibriSpeech" / split
        if not split_dir.exists():
            logger.warning(
                "LibriSpeech split %s not found at %s; dataset will be empty.",
                split,
                split_dir,
            )
            self.files: List[Path] = []
        else:
            self.files = _list_audio_files(split_dir)
        if transform is None:
            transform = AudioTransform(
                target_sample_rate=sample_rate,
                segment_length=segment_length,
                augment=augment,
                random_crop=split.startswith("train"),
            )
        self.transform = transform

    def _maybe_download(self) -> None:
        if not self.download:
            return
        target = self.root_dir / "LibriSpeech" / self.split
        if target.exists() and any(target.rglob("*.flac")):
            return
        try:
            from ultracodec.data.download import download_librispeech

            logger.info("Downloading LibriSpeech split %s -> %s", self.split, self.root_dir)
            download_librispeech(self.root_dir, splits=[self.split])
        except Exception as exc:  # pragma: no cover - network optional
            logger.warning("LibriSpeech automatic download failed: %s", exc)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict:
        path = self.files[idx]
        wav, sr = _safe_load(path)
        wav = self.transform(wav, sr)
        return {
            "wav": wav,
            "path": str(path),
            "dataset": "librispeech",
            "split": self.split,
        }


# ----------------------------------------------------------------------
# MixedDataset
# ----------------------------------------------------------------------
class MixedDataset(Dataset):
    """Combine several datasets, optionally with weighted random sampling.

    When ``weights`` is ``None`` the datasets are concatenated and indexed
    deterministically (``len = sum(len(d) for d in datasets)``). When
    ``weights`` is provided, sampling is multinomial: ``__len__`` is
    user-controlled via ``length`` (defaults to the sum of dataset sizes).
    """

    def __init__(
        self,
        datasets: Iterable[Dataset],
        weights: Optional[Sequence[float]] = None,
        length: Optional[int] = None,
        seed: int = 0,
    ) -> None:
        self.datasets: List[Dataset] = list(datasets)
        if not self.datasets:
            raise ValueError("MixedDataset requires at least one dataset.")
        self.weights = list(weights) if weights is not None else None
        if self.weights is not None:
            if len(self.weights) != len(self.datasets):
                raise ValueError("weights must match the number of datasets.")
            total = float(sum(self.weights))
            if total <= 0:
                raise ValueError("weights must sum to a positive value.")
            self.weights = [w / total for w in self.weights]
        self._sizes = [len(d) for d in self.datasets]
        self._cum_sizes: List[int] = []
        cumulative = 0
        for size in self._sizes:
            cumulative += size
            self._cum_sizes.append(cumulative)
        self._length = length if length is not None else self._cum_sizes[-1]
        self._rng = random.Random(seed)

    def __len__(self) -> int:
        return self._length

    def _resolve(self, idx: int) -> tuple[int, int]:
        if self.weights is None:
            for i, cum in enumerate(self._cum_sizes):
                if idx < cum:
                    prev = self._cum_sizes[i - 1] if i > 0 else 0
                    return i, idx - prev
            raise IndexError(idx)
        # Weighted sampling – ``idx`` is treated as an opaque counter.
        ds_idx = self._rng.choices(range(len(self.datasets)), weights=self.weights, k=1)[0]
        size = self._sizes[ds_idx]
        if size == 0:
            raise RuntimeError(
                f"Sub-dataset at index {ds_idx} has length 0; cannot sample from it."
            )
        item_idx = self._rng.randint(0, size - 1)
        return ds_idx, item_idx

    def __getitem__(self, idx: int) -> dict:
        ds_idx, item_idx = self._resolve(idx)
        return self.datasets[ds_idx][item_idx]


__all__ = ["VCTKDataset", "LibriSpeechDataset", "MixedDataset"]
