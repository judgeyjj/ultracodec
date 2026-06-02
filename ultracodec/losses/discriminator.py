"""HiFi-GAN-style discriminators for UltraCodec adversarial training.

Implements:
    - :class:`PeriodDiscriminator` / :class:`MultiPeriodDiscriminator`
    - :class:`ScaleDiscriminator`  / :class:`MultiScaleDiscriminator`
    - :class:`CombinedDiscriminator`  (MPD + MSD wrapper)

All discriminators return ``(logits, features)`` tuples per sub-discriminator,
where ``features`` is a list of intermediate activations used by the feature
matching loss.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm, weight_norm


# ---------------------------------------------------------------------------
# Period discriminator (HiFi-GAN MPD)
# ---------------------------------------------------------------------------
class PeriodDiscriminator(nn.Module):
    """Single-period 2-D discriminator from HiFi-GAN.

    Reshapes the 1-D waveform to ``[B, 1, T // p, p]`` and applies a stack of
    2-D convolutions, exposing periodic structure of period ``p``.

    Args:
        period: Sampling period of the discriminator.
        channels: Initial number of channels.
        kernel_size: 2-D conv kernel size on the time axis.
        stride: Stride along the time axis.
        use_spectral_norm: If ``True``, use spectral norm instead of weight norm.
    """

    def __init__(
        self,
        period: int,
        channels: int = 32,
        kernel_size: int = 5,
        stride: int = 3,
        use_spectral_norm: bool = False,
    ) -> None:
        super().__init__()
        self.period = period
        norm_f = spectral_norm if use_spectral_norm else weight_norm
        chs = [1, channels, channels * 4, channels * 16, channels * 32, channels * 32]
        self.convs = nn.ModuleList()
        for in_c, out_c in zip(chs[:-1], chs[1:]):
            self.convs.append(
                norm_f(
                    nn.Conv2d(
                        in_c,
                        out_c,
                        kernel_size=(kernel_size, 1),
                        stride=(stride, 1),
                        padding=(kernel_size // 2, 0),
                    )
                )
            )
        self.conv_post = norm_f(
            nn.Conv2d(chs[-1], 1, kernel_size=(3, 1), stride=1, padding=(1, 0))
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Run the period discriminator.

        Args:
            x: Waveform ``[B, 1, T]`` or ``[B, T]``.

        Returns:
            ``(logits[B, -1], features)`` where ``features`` lists the
            activations after each conv layer.
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)
        b, c, t = x.shape
        # 1-D → 2-D periodic reshape.
        if t % self.period != 0:
            pad = self.period - (t % self.period)
            x = F.pad(x, (0, pad), mode="reflect")
            t = t + pad
        x = x.view(b, c, t // self.period, self.period)

        features: List[torch.Tensor] = []
        h = x
        for conv in self.convs:
            h = F.leaky_relu(conv(h), 0.1)
            features.append(h)
        h = self.conv_post(h)
        features.append(h)
        logits = h.flatten(1)
        return logits, features


class MultiPeriodDiscriminator(nn.Module):
    """Stack of :class:`PeriodDiscriminator` operating on different periods.

    Args:
        periods: Sequence of integer periods (default ``[2, 3, 5, 7, 11]``).
    """

    def __init__(self, periods: Sequence[int] = (2, 3, 5, 7, 11)) -> None:
        super().__init__()
        self.discriminators = nn.ModuleList(
            [PeriodDiscriminator(p) for p in periods]
        )

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[List[torch.Tensor]]]:
        """Return per-discriminator ``(logits, features)`` lists."""
        all_logits: List[torch.Tensor] = []
        all_feats: List[List[torch.Tensor]] = []
        for d in self.discriminators:
            logit, feats = d(x)
            all_logits.append(logit)
            all_feats.append(feats)
        return all_logits, all_feats


# ---------------------------------------------------------------------------
# Scale discriminator (HiFi-GAN MSD)
# ---------------------------------------------------------------------------
class ScaleDiscriminator(nn.Module):
    """Single-scale 1-D discriminator from HiFi-GAN.

    Args:
        use_spectral_norm: Use spectral norm (recommended for the first
            scale in the MSD stack).
    """

    def __init__(self, use_spectral_norm: bool = False) -> None:
        super().__init__()
        norm_f = spectral_norm if use_spectral_norm else weight_norm
        self.convs = nn.ModuleList(
            [
                norm_f(nn.Conv1d(1, 16, 15, 1, padding=7)),
                norm_f(nn.Conv1d(16, 64, 41, 4, groups=4, padding=20)),
                norm_f(nn.Conv1d(64, 256, 41, 4, groups=16, padding=20)),
                norm_f(nn.Conv1d(256, 1024, 41, 4, groups=64, padding=20)),
                norm_f(nn.Conv1d(1024, 1024, 41, 4, groups=256, padding=20)),
                norm_f(nn.Conv1d(1024, 1024, 5, 1, padding=2)),
            ]
        )
        self.conv_post = norm_f(nn.Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Forward pass returning logits and intermediate features."""
        if x.dim() == 2:
            x = x.unsqueeze(1)
        features: List[torch.Tensor] = []
        h = x
        for conv in self.convs:
            h = F.leaky_relu(conv(h), 0.1)
            features.append(h)
        h = self.conv_post(h)
        features.append(h)
        return h.flatten(1), features


class MultiScaleDiscriminator(nn.Module):
    """Multi-scale discriminator: progressively downsampled audio inputs.

    Args:
        scales: Number of scales (default 3 → ``1×, 2×, 4×`` average pooling).
    """

    def __init__(self, scales: int = 3) -> None:
        super().__init__()
        self.discriminators = nn.ModuleList(
            [ScaleDiscriminator(use_spectral_norm=(i == 0)) for i in range(scales)]
        )
        self.pool = nn.AvgPool1d(kernel_size=4, stride=2, padding=2)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[List[torch.Tensor]]]:
        """Return per-scale ``(logits, features)`` lists."""
        if x.dim() == 2:
            x = x.unsqueeze(1)
        all_logits: List[torch.Tensor] = []
        all_feats: List[List[torch.Tensor]] = []
        h = x
        for i, d in enumerate(self.discriminators):
            if i > 0:
                h = self.pool(h)
            logit, feats = d(h)
            all_logits.append(logit)
            all_feats.append(feats)
        return all_logits, all_feats


# ---------------------------------------------------------------------------
# Combined MPD + MSD
# ---------------------------------------------------------------------------
class CombinedDiscriminator(nn.Module):
    """Wrapper bundling :class:`MultiPeriodDiscriminator` + :class:`MultiScaleDiscriminator`.

    Args:
        periods: Periods for the MPD branch.
        scales: Number of scales for the MSD branch.
    """

    def __init__(
        self,
        periods: Sequence[int] = (2, 3, 5, 7, 11),
        scales: int = 3,
    ) -> None:
        super().__init__()
        self.mpd = MultiPeriodDiscriminator(periods=periods)
        self.msd = MultiScaleDiscriminator(scales=scales)

    def forward(self, x: torch.Tensor) -> Dict[str, List]:
        """Run both branches and return concatenated outputs.

        Args:
            x: Waveform ``[B, 1, T]`` or ``[B, T]``.

        Returns:
            Dictionary with keys ``logits`` (list of per-sub-discriminator
            logits) and ``features`` (list of per-sub-discriminator feature
            lists). The order is ``MPD ... MSD ...``.
        """
        mpd_logits, mpd_feats = self.mpd(x)
        msd_logits, msd_feats = self.msd(x)
        return {
            "logits": list(mpd_logits) + list(msd_logits),
            "features": list(mpd_feats) + list(msd_feats),
        }


__all__ = [
    "PeriodDiscriminator",
    "MultiPeriodDiscriminator",
    "ScaleDiscriminator",
    "MultiScaleDiscriminator",
    "CombinedDiscriminator",
]
