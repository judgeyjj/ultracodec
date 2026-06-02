"""Adaptive Frame Rate (AFR) gating module for UltraCodec.

Implements information-density-driven dynamic frame selection:
    - Silent/stationary segments → drastically reduce frame rate
    - High-density segments (consonants, transitions) → maintain high frame rate
    - Steady segments (sustained vowels) → moderately reduce frame rate

Uses Gumbel-Sigmoid for differentiable discrete gate training.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class AdaptiveFrameRateGate(nn.Module):
    """Adaptive Frame Rate gating module.

    Dynamically decides whether to keep or drop frames based on
    information density estimated from multiple signals:
        1. VAD (Voice Activity Detection) score
        2. Inter-frame entropy change
        3. Residual norm (post-quantization residual magnitude)

    Uses Gumbel-Sigmoid for differentiable discrete gating during training.

    Args:
        config: OmegaConf dict with AFR configuration.
        input_dim: Feature dimension of input.
    """

    def __init__(self, config: dict, input_dim: int = 512):
        super().__init__()
        self.enabled = config.get('enabled', True)
        self.target_keep = config.get('target_keep', 0.0625)
        self.gumbel_temperature = config.get('gumbel_temperature', 1.0)
        self.entropy_weight = config.get('entropy_weight', 0.1)
        self.min_frame_rate = config.get('min_frame_rate', 1.5625)
        self.max_frame_rate = config.get('max_frame_rate', 6.25)

        # VAD head: predicts voice activity from features
        self.vad_head = nn.Sequential(
            nn.Linear(input_dim, input_dim // 4),
            nn.GELU(),
            nn.Linear(input_dim // 4, 1),
        )

        # Entropy estimator: estimates local information density
        self.entropy_head = nn.Sequential(
            nn.Linear(input_dim, input_dim // 4),
            nn.GELU(),
            nn.Linear(input_dim // 4, 1),
        )

        # Gate network: combines all signals into keep/drop decision
        # Input: VAD score + entropy score + residual norm = 3 features
        self.gate_network = nn.Sequential(
            nn.Linear(3, 64),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

        # Learnable weights for combining signals
        self.beta = nn.Parameter(torch.tensor([1.0, 1.0, 1.0]))

    def _gumbel_sigmoid(
        self, logits: torch.Tensor, temperature: float, hard: bool = False
    ) -> torch.Tensor:
        """Gumbel-Sigmoid for differentiable binary gating.

        Args:
            logits: Gate logits [B, T].
            temperature: Gumbel temperature (lower = more discrete).
            hard: If True, use straight-through hard gating.

        Returns:
            Soft or hard gate values [B, T] in [0, 1].
        """
        if self.training:
            # Add Gumbel noise for exploration
            u = torch.rand_like(logits).clamp(1e-8, 1 - 1e-8)
            gumbel_noise = -torch.log(-torch.log(u))
            y = torch.sigmoid((logits + gumbel_noise) / temperature)
        else:
            y = torch.sigmoid(logits / temperature)

        if hard:
            # Straight-through: hard in forward, soft in backward
            y_hard = (y > 0.5).float()
            y = y_hard - y.detach() + y
        return y

    def forward(
        self,
        x: torch.Tensor,
        residual_norms: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass - compute gate decisions and apply masking.

        Args:
            x: Encoded features [B, D, T].
            residual_norms: Per-frame residual norms from SPQ [B, T].
                           If None, uses zero as placeholder.

        Returns:
            Dictionary with:
                - 'output': Gated features [B, D, T] (masked with gate values)
                - 'gate_decisions': Gate values [B, T] (0=drop, 1=keep)
                - 'keep_ratio': Actual proportion of kept frames
                - 'rate_loss': Regularization loss for target keep ratio
                - 'kept_indices': Boolean mask of kept frames [B, T]
        """
        if not self.enabled:
            B, D, T = x.shape
            return {
                'output': x,
                'gate_decisions': torch.ones(B, T, device=x.device),
                'keep_ratio': torch.tensor(1.0, device=x.device),
                'rate_loss': torch.tensor(0.0, device=x.device),
                'kept_indices': torch.ones(B, T, device=x.device, dtype=torch.bool),
            }

        B, D, T = x.shape

        # Transpose for per-frame processing
        x_t = rearrange(x, 'b d t -> b t d')  # [B, T, D]

        # 1. VAD score
        vad_score = self.vad_head(x_t).squeeze(-1)  # [B, T]
        vad_score = torch.sigmoid(vad_score)

        # 2. Entropy estimation (local information density)
        entropy_score = self.entropy_head(x_t).squeeze(-1)  # [B, T]
        entropy_score = torch.sigmoid(entropy_score)

        # 3. Residual norm (normalized)
        if residual_norms is not None:
            # Normalize residual norms to [0, 1] range
            r_min = residual_norms.min(dim=-1, keepdim=True).values
            r_max = residual_norms.max(dim=-1, keepdim=True).values
            r_range = (r_max - r_min).clamp(min=1e-8)
            norm_score = (residual_norms - r_min) / r_range
        else:
            norm_score = torch.zeros(B, T, device=x.device)

        # Combine signals with learnable weights
        beta_softmax = F.softmax(self.beta, dim=0)
        gate_features = torch.stack([
            vad_score * beta_softmax[0],
            entropy_score * beta_softmax[1],
            norm_score * beta_softmax[2],
        ], dim=-1)  # [B, T, 3]

        # Gate logits
        gate_logits = self.gate_network(gate_features).squeeze(-1)  # [B, T]

        # Apply Gumbel-Sigmoid
        temperature = self.gumbel_temperature
        hard = not self.training  # Hard gating at inference
        gate_values = self._gumbel_sigmoid(gate_logits, temperature, hard=hard)

        # Compute keep ratio
        keep_ratio = gate_values.mean()

        # Rate regularization loss: encourage keep_ratio to match target
        rate_loss = (keep_ratio - self.target_keep) ** 2

        # Apply gate to features (soft masking during training)
        gate_expanded = gate_values.unsqueeze(1)  # [B, 1, T]
        output = x * gate_expanded  # [B, D, T]

        # Binary mask for inference
        kept_indices = gate_values > 0.5

        return {
            'output': output,
            'gate_decisions': gate_values,
            'keep_ratio': keep_ratio,
            'rate_loss': rate_loss,
            'kept_indices': kept_indices,
        }

    def compress(self, x: torch.Tensor, gate_decisions: torch.Tensor) -> torch.Tensor:
        """Remove dropped frames for actual compression (inference).

        Args:
            x: Features [B, D, T].
            gate_decisions: Binary gate values [B, T].

        Returns:
            Compressed features with dropped frames removed.
            Note: Returns a list of tensors since each batch item
            may have different number of kept frames.
        """
        B, D, T = x.shape
        kept_mask = gate_decisions > 0.5  # [B, T]

        # For uniform processing, pad to max kept length
        max_kept = kept_mask.sum(dim=-1).max().item()
        compressed = torch.zeros(B, D, max_kept, device=x.device)

        for b in range(B):
            mask_b = kept_mask[b]  # [T]
            kept_frames = x[b, :, mask_b]  # [D, T_kept]
            compressed[b, :, :kept_frames.size(1)] = kept_frames

        return compressed

    def decompress(
        self, x_compressed: torch.Tensor, gate_decisions: torch.Tensor, target_length: int
    ) -> torch.Tensor:
        """Restore dropped frames via interpolation (inference).

        Args:
            x_compressed: Compressed features [B, D, T_compressed].
            gate_decisions: Original gate decisions [B, T_original].
            target_length: Target output length T_original.

        Returns:
            Restored features [B, D, T_original] with interpolated dropped frames.
        """
        B, D, _ = x_compressed.shape
        device = x_compressed.device
        output = torch.zeros(B, D, target_length, device=device)

        kept_mask = gate_decisions > 0.5  # [B, T]

        for b in range(B):
            mask_b = kept_mask[b]  # [T_original]
            kept_positions = mask_b.nonzero(as_tuple=True)[0]  # positions of kept frames
            num_kept = kept_positions.size(0)

            if num_kept == 0:
                continue

            # Place kept frames
            kept_frames = x_compressed[b, :, :num_kept]  # [D, num_kept]

            if num_kept == target_length:
                output[b] = kept_frames
            else:
                # Interpolate to fill gaps
                # Use linear interpolation between kept frames
                kept_float = kept_positions.float()
                all_positions = torch.arange(target_length, device=device).float()

                # For each output position, interpolate from nearest kept frames
                for d in range(D):
                    output[b, d] = self._interpolate_1d(
                        kept_float, kept_frames[d], all_positions
                    )

        return output

    @staticmethod
    def _interpolate_1d(
        x_known: torch.Tensor, y_known: torch.Tensor, x_query: torch.Tensor
    ) -> torch.Tensor:
        """1D linear interpolation.

        Args:
            x_known: Known x positions [N].
            y_known: Known y values [N].
            x_query: Query positions [M].

        Returns:
            Interpolated values [M].
        """
        # Clamp query positions to known range
        indices = torch.searchsorted(x_known, x_query).clamp(1, len(x_known) - 1)
        x0 = x_known[indices - 1]
        x1 = x_known[indices]
        y0 = y_known[indices - 1]
        y1 = y_known[indices]

        # Linear interpolation
        t = ((x_query - x0) / (x1 - x0 + 1e-8)).clamp(0, 1)
        return y0 + t * (y1 - y0)

    def set_temperature(self, temperature: float) -> None:
        """Update Gumbel temperature (for annealing during training).

        Args:
            temperature: New temperature value (lower = more discrete).
        """
        self.gumbel_temperature = temperature
