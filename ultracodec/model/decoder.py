"""Cascaded Coarse-to-Fine Decoder for UltraCodec.

Implements progressive upsampling from 3.125Hz compressed representation
back to 16kHz waveform through multiple stages:
    Stage 1: 3.125Hz → 6.25Hz (Transformer + transposed conv)
    Stage 2: 6.25Hz → 12.5Hz (Transformer + transposed conv)
    Stage 3: 12.5Hz → 25Hz (Transformer + transposed conv)
    Stage 4: 25Hz → 50Hz (Transformer + transposed conv)
    Final: 50Hz → 16kHz waveform (large-stride transposed conv)

Uses skip connections from encoder intermediate features.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .encoder import CausalConv1d, CausalTransformerBlock


class CausalTransposeConv1d(nn.Module):
    """Causal transposed convolution for upsampling.

    Removes right-side padding to maintain causal property.

    Args:
        in_channels: Input channels.
        out_channels: Output channels.
        kernel_size: Kernel size.
        stride: Upsampling factor.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
    ):
        super().__init__()
        self.stride = stride
        # Padding to remove from right side for causal output
        self.right_pad = kernel_size - stride

        self.conv_transpose = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size, stride=stride, padding=0
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input [B, C, T].

        Returns:
            Upsampled output [B, C_out, T * stride].
        """
        y = self.conv_transpose(x)
        # Remove right padding for causal
        if self.right_pad > 0:
            y = y[:, :, :-self.right_pad]
        return y


class UpsampleBlock(nn.Module):
    """Upsample block: transposed convolution + Transformer refinement.

    Args:
        in_dim: Input dimension.
        out_dim: Output dimension.
        upsample_ratio: Temporal upsampling factor.
        num_transformer_layers: Number of Transformer blocks for refinement.
        num_heads: Number of attention heads.
        use_skip: Whether to use skip connection from encoder.
        skip_dim: Dimension of incoming skip features (if different from in_dim).
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        upsample_ratio: int = 2,
        num_transformer_layers: int = 2,
        num_heads: int = 8,
        use_skip: bool = True,
        skip_dim: Optional[int] = None,
    ):
        super().__init__()
        self.use_skip = use_skip

        # Skip connection projection (encoder feature might have different dim)
        actual_skip_dim = skip_dim if skip_dim is not None else in_dim
        self.skip_proj = nn.Conv1d(actual_skip_dim, in_dim, 1) if use_skip else None

        # Upsampling via transposed convolution
        self.upsample = CausalTransposeConv1d(
            in_channels=in_dim + (in_dim if use_skip else 0),
            out_channels=out_dim,
            kernel_size=2 * upsample_ratio,
            stride=upsample_ratio,
        )

        # Post-upsample refinement
        self.norm = nn.GroupNorm(1, out_dim)
        self.activation = nn.GELU()

        # Transformer refinement blocks
        self.transformer_blocks = nn.ModuleList([
            CausalTransformerBlock(dim=out_dim, num_heads=num_heads)
            for _ in range(num_transformer_layers)
        ])

        # Final conv for smoothing
        self.smooth = CausalConv1d(out_dim, out_dim, kernel_size=3, stride=1)

    def forward(
        self, x: torch.Tensor, skip: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input features [B, D_in, T].
            skip: Skip connection features from encoder [B, D_in, T_skip].
                  T_skip should match T (or be aligned).

        Returns:
            Upsampled and refined features [B, D_out, T * upsample_ratio].
        """
        if self.use_skip and skip is not None:
            # Align skip connection length to input
            if skip.size(2) != x.size(2):
                skip = F.interpolate(skip, size=x.size(2), mode='linear', align_corners=False)
            skip_proj = self.skip_proj(skip)
            x = torch.cat([x, skip_proj], dim=1)  # [B, 2*D_in, T]

        # Upsample
        x = self.upsample(x)
        x = self.norm(x)
        x = self.activation(x)

        # Transformer refinement
        for block in self.transformer_blocks:
            x = block(x)

        # Smooth
        x = self.smooth(x)
        return x


class CoarseDecoder(nn.Module):
    """Coarse decoder - from 3.125Hz to intermediate representation.

    Handles the initial upsampling stages using Transformer + transposed conv.

    Args:
        in_dim: Input dimension (encoder output dim).
        hidden_dims: Hidden dimensions for each stage (reversed from encoder).
        upsample_ratios: Upsampling factors for each stage.
        num_transformer_layers: Transformer layers per stage.
        skip_dims: Dimensions of skip features from encoder (one per stage).
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dims: List[int],
        upsample_ratios: List[int],
        num_transformer_layers: int = 2,
        skip_dims: Optional[List[int]] = None,
    ):
        super().__init__()
        self.num_stages = len(hidden_dims) - 1  # stages between consecutive dims

        self.stages = nn.ModuleList()
        for i in range(self.num_stages):
            s_dim = skip_dims[i] if skip_dims is not None and i < len(skip_dims) else None
            self.stages.append(
                UpsampleBlock(
                    in_dim=hidden_dims[i],
                    out_dim=hidden_dims[i + 1],
                    upsample_ratio=upsample_ratios[i],
                    num_transformer_layers=num_transformer_layers,
                    num_heads=min(8, hidden_dims[i + 1] // 32),
                    use_skip=True,
                    skip_dim=s_dim,
                )
            )

    def forward(
        self, x: torch.Tensor, skip_features: Optional[List[torch.Tensor]] = None
    ) -> torch.Tensor:
        """Forward pass through coarse decoder.

        Args:
            x: Input [B, D, T] at lowest frame rate.
            skip_features: List of encoder intermediate features (reversed order).

        Returns:
            Upsampled features at higher frame rate.
        """
        for i, stage in enumerate(self.stages):
            skip = skip_features[i] if skip_features is not None and i < len(skip_features) else None
            x = stage(x, skip=skip)
        return x


class FineDecoder(nn.Module):
    """Fine decoder - from 50Hz features to 16kHz waveform.

    Uses large-stride transposed convolutions to reconstruct the waveform
    from the 50Hz intermediate representation.

    Args:
        in_dim: Input feature dimension.
        output_channels: Output audio channels (1 for mono).
    """

    def __init__(self, in_dim: int = 64, output_channels: int = 1):
        super().__init__()

        # Multi-stage upsampling: 50Hz → 16kHz
        # Total upsample = 320x, factored as 4 × 4 × 4 × 5
        self.upsample_layers = nn.ModuleList([
            nn.Sequential(
                CausalTransposeConv1d(in_dim, in_dim, kernel_size=8, stride=4),
                nn.GELU(),
                CausalConv1d(in_dim, in_dim, kernel_size=7, stride=1),
                nn.GELU(),
            ),
            nn.Sequential(
                CausalTransposeConv1d(in_dim, in_dim // 2, kernel_size=8, stride=4),
                nn.GELU(),
                CausalConv1d(in_dim // 2, in_dim // 2, kernel_size=7, stride=1),
                nn.GELU(),
            ),
            nn.Sequential(
                CausalTransposeConv1d(in_dim // 2, in_dim // 4, kernel_size=8, stride=4),
                nn.GELU(),
                CausalConv1d(in_dim // 4, in_dim // 4, kernel_size=7, stride=1),
                nn.GELU(),
            ),
            nn.Sequential(
                CausalTransposeConv1d(in_dim // 4, in_dim // 4, kernel_size=10, stride=5),
                nn.GELU(),
                CausalConv1d(in_dim // 4, in_dim // 4, kernel_size=7, stride=1),
                nn.GELU(),
            ),
        ])
        # Total: 4 * 4 * 4 * 5 = 320

        # Final output projection
        self.output_conv = CausalConv1d(in_dim // 4, output_channels, kernel_size=7, stride=1)
        self.tanh = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass to generate waveform.

        Args:
            x: Input features [B, D, T] at 50Hz.

        Returns:
            Waveform [B, 1, T * 320] at 16kHz.
        """
        for layer in self.upsample_layers:
            x = layer(x)

        x = self.output_conv(x)
        x = self.tanh(x)
        return x


class CascadedDecoder(nn.Module):
    """Cascaded Coarse-to-Fine Decoder for UltraCodec.

    Progressive upsampling pipeline:
        Stage 1-4: 3.125Hz → 6.25Hz → 12.5Hz → 25Hz → 50Hz (coarse stages)
        Stage 5: 50Hz → 16kHz waveform (fine vocoder stage)

    Uses skip connections from encoder for multi-scale information recovery.

    Args:
        config: OmegaConf dict with decoder configuration.
        encoder_dims: List of encoder hidden dimensions (for skip connections).
    """

    def __init__(self, config: dict, encoder_dims: Optional[List[int]] = None):
        super().__init__()
        hidden_dims: List[int] = list(config.get('hidden_dims', [512, 256, 128, 64]))
        upsample_ratios: List[int] = list(config.get('upsample_ratios', [2, 2, 2, 2]))
        num_transformer_layers: int = config.get('num_transformer_layers', 2)

        assert len(hidden_dims) == len(upsample_ratios) + 1 or \
               len(hidden_dims) == len(upsample_ratios), \
            "hidden_dims length should match upsample_ratios (or +1)"

        # If hidden_dims has same length as upsample_ratios, add output dim
        if len(hidden_dims) == len(upsample_ratios):
            dims = hidden_dims
        else:
            dims = hidden_dims

        # Compute skip dims from encoder_dims.
        # Encoder features = [frontend_out, stage0_out, stage1_out, ..., stageN_out]
        # Decoder gets reversed(features[:-1]) as skips.
        # With encoder hidden_dims=[64,128,256,512]:
        #   features = [64-dim, 64-dim, 128-dim, 256-dim, 512-dim]
        #   skip_features = reversed(features[:-1]) = [256, 128, 64, 64]
        skip_dims = None
        if encoder_dims is not None:
            # Encoder features list: [frontend_dim, stage0_out, stage1_out, ...]
            # frontend_dim = encoder_dims[0]
            # stage i out = encoder_dims[i] (for i=0 it's same as frontend)
            # Full features list dims: [enc[0]] + [enc[0], enc[1], enc[2], enc[3]]
            # Actually: features = [frontend=enc[0], stage0_out=enc[0], stage1_out=enc[1], stage2_out=enc[2], stage3_out=enc[3]]
            # Skip = reversed(features[:-1]) = [enc[2], enc[1], enc[0], enc[0]]
            enc_feature_dims = [encoder_dims[0]]  # frontend output
            for i in range(len(encoder_dims)):
                enc_feature_dims.append(encoder_dims[i])
            # features[:-1] then reversed
            skip_dims = list(reversed(enc_feature_dims[:-1]))

        # Coarse decoder stages (3.125Hz → 50Hz)
        self.coarse = CoarseDecoder(
            in_dim=dims[0],
            hidden_dims=dims,
            upsample_ratios=upsample_ratios,
            num_transformer_layers=num_transformer_layers,
            skip_dims=skip_dims,
        )

        # Fine decoder (50Hz → 16kHz waveform)
        self.fine = FineDecoder(
            in_dim=dims[-1],
            output_channels=1,
        )

    def forward(
        self,
        x: torch.Tensor,
        encoder_features: Optional[List[torch.Tensor]] = None,
        target_length: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through cascaded decoder.

        Args:
            x: Quantized/gated features [B, D, T] at 3.125Hz.
            encoder_features: List of encoder intermediate features for skips.
                            Should be in reverse order (deepest first).
            target_length: Target waveform length for trimming.

        Returns:
            Dictionary with:
                - 'audio': Reconstructed waveform [B, 1, T_audio].
                - 'coarse_features': Intermediate 50Hz features.
        """
        # Prepare skip connections (reverse encoder features for decoder)
        skip_features = None
        if encoder_features is not None and len(encoder_features) > 1:
            # Encoder features are [frontend_out, stage1_out, ..., stageN_out]
            # Decoder needs them in reverse order (excluding the final encoder output)
            skip_features = list(reversed(encoder_features[:-1]))

        # Coarse upsampling: 3.125Hz → 50Hz
        coarse_out = self.coarse(x, skip_features=skip_features)

        # Fine vocoder: 50Hz → 16kHz
        audio = self.fine(coarse_out)

        # Trim to target length if specified
        if target_length is not None and audio.size(2) > target_length:
            audio = audio[:, :, :target_length]

        return {
            'audio': audio,
            'coarse_features': coarse_out,
        }
