"""General-purpose utilities used across UltraCodec."""
from __future__ import annotations

import logging
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Union

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def set_seed(seed: int, deterministic: bool = False) -> None:
    """Set RNG seeds across Python, NumPy and PyTorch for reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(prefer: str = "cuda") -> torch.device:
    """Return the best available torch device (cuda > mps > cpu)."""
    if prefer == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if prefer in {"mps", "auto"} and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def init_logger(
    name: str = "ultracodec",
    level: int = logging.INFO,
    log_file: Optional[Union[str, Path]] = None,
    fmt: str = "[%(asctime)s] %(levelname)s %(name)s: %(message)s",
) -> logging.Logger:
    """Initialise a logger that writes to stdout and optionally to ``log_file``."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    if logger.handlers:
        return logger

    formatter = logging.Formatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------
def save_checkpoint(
    path: Union[str, Path],
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    step: int = 0,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    """Persist a checkpoint to ``path``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "step": step,
        "model": model.state_dict(),
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scheduler is not None and hasattr(scheduler, "state_dict"):
        payload["scheduler"] = scheduler.state_dict()
    if extra is not None:
        payload["extra"] = extra
    torch.save(payload, path)
    return path


def load_checkpoint(
    path: Union[str, Path],
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    map_location: Optional[Union[str, torch.device]] = "cpu",
    strict: bool = True,
) -> Dict[str, Any]:
    """Load a checkpoint produced by :func:`save_checkpoint`.

    Returns the full payload (including ``step`` and ``extra`` if present).
    """
    payload = torch.load(str(path), map_location=map_location)
    state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    model.load_state_dict(state, strict=strict)
    if optimizer is not None and isinstance(payload, dict) and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and isinstance(payload, dict) and "scheduler" in payload \
            and hasattr(scheduler, "load_state_dict"):
        scheduler.load_state_dict(payload["scheduler"])
    return payload if isinstance(payload, dict) else {"model": payload}


def keep_last_checkpoints(directory: Union[str, Path], pattern: str = "*.pt", keep: int = 5) -> None:
    """Delete older checkpoints, keeping only the ``keep`` most recent."""
    directory = Path(directory)
    files = sorted(directory.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    for old in files[keep:]:
        try:
            old.unlink()
        except OSError:  # pragma: no cover
            pass


# ---------------------------------------------------------------------------
# Exponential moving average
# ---------------------------------------------------------------------------
class EMA:
    """Lightweight EMA over model parameters."""

    def __init__(self, model: torch.nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.shadow: Dict[str, torch.Tensor] = {}
        self.backup: Dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.detach().clone()

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            assert name in self.shadow, f"Parameter '{name}' missing from EMA shadow."
            self.shadow[name].mul_(self.decay).add_(param.detach(), alpha=1.0 - self.decay)

    def apply_shadow(self, model: torch.nn.Module) -> None:
        """Swap the live model parameters with the EMA shadow (use during eval)."""
        for name, param in model.named_parameters():
            if name in self.shadow:
                self.backup[name] = param.detach().clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model: torch.nn.Module) -> None:
        """Restore the parameters that were swapped by :meth:`apply_shadow`."""
        for name, param in model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup.clear()


# ---------------------------------------------------------------------------
# Schedulers
# ---------------------------------------------------------------------------
def cosine_warmup_lambda(warmup_steps: int, total_steps: int, min_ratio: float = 0.1):
    """Return a function suitable for :class:`torch.optim.lr_scheduler.LambdaLR`."""
    import math

    def fn(step: int) -> float:
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(1.0, max(0.0, progress))
        cos = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_ratio + (1.0 - min_ratio) * cos

    return fn


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    name: str,
    *,
    warmup_steps: int = 0,
    total_steps: int = 0,
    min_ratio: float = 0.1,
):
    """Construct a common LR scheduler by name."""
    if name in {"cosine", "cosine_warmup"}:
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            cosine_warmup_lambda(warmup_steps, total_steps, min_ratio),
        )
    if name == "constant":
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    raise ValueError(f"Unknown scheduler: {name}")


# ---------------------------------------------------------------------------
# Audio I/O
# ---------------------------------------------------------------------------
def load_audio(path: Union[str, Path], target_sr: Optional[int] = None) -> tuple[torch.Tensor, int]:
    """Load an audio file and (optionally) resample to ``target_sr``.

    Returns ``(wav[1, T], sample_rate)``.
    """
    import torchaudio

    wav, sr = torchaudio.load(str(path))
    if wav.dim() > 1 and wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if target_sr is not None and sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
        sr = target_sr
    return wav, sr


def save_audio(path: Union[str, Path], wav: torch.Tensor, sample_rate: int) -> Path:
    """Save ``wav`` (``[C, T]`` or ``[T]``) to ``path`` as 16-bit PCM."""
    import torchaudio

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    torchaudio.save(str(path), wav.detach().cpu(), sample_rate)
    return path


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------
@dataclass
class AverageMeter:
    """Online average tracker."""

    name: str = ""
    sum: float = 0.0
    count: int = 0

    def update(self, value: float, n: int = 1) -> None:
        self.sum += float(value) * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / self.count if self.count else 0.0

    def reset(self) -> None:
        self.sum = 0.0
        self.count = 0


def count_parameters(model: torch.nn.Module, only_trainable: bool = True) -> int:
    """Count parameters in ``model``."""
    if only_trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def flatten_dict(d: Dict[str, Any], parent: str = "", sep: str = ".") -> Dict[str, Any]:
    """Flatten nested dicts (helper for logging hyperparameters)."""
    items: Dict[str, Any] = {}
    for key, value in d.items():
        full = f"{parent}{sep}{key}" if parent else key
        if isinstance(value, dict):
            items.update(flatten_dict(value, full, sep))
        else:
            items[full] = value
    return items


def to_device(batch: Any, device: torch.device) -> Any:
    """Recursively move tensors in a (possibly nested) batch to ``device``."""
    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=True)
    if isinstance(batch, dict):
        return {k: to_device(v, device) for k, v in batch.items()}
    if isinstance(batch, (list, tuple)):
        moved = [to_device(v, device) for v in batch]
        return type(batch)(moved) if isinstance(batch, tuple) else moved
    return batch


__all__ = [
    "AverageMeter",
    "EMA",
    "build_scheduler",
    "cosine_warmup_lambda",
    "count_parameters",
    "flatten_dict",
    "get_device",
    "init_logger",
    "keep_last_checkpoints",
    "load_audio",
    "load_checkpoint",
    "save_audio",
    "save_checkpoint",
    "set_seed",
    "to_device",
]
