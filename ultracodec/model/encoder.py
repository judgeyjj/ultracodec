"""Hierarchical Temporal Compression Encoder (HTCE) for UltraCodec.

Implements a 4-stage progressive downsampling encoder that compresses
16kHz waveform from 50Hz features down to 3.125Hz representations.

Architecture:
    Frontend (stride=320): waveform → 50Hz features
    Stage 1: 50Hz → 25Hz (stride-2 causal conv + Transformer)
    Stage 2: 25Hz → 12.5Hz (TimeAttnPool + Transformer)
    Stage 3: 12.5Hz → 6.25Hz (TimeAttnPool + Transformer)
    Stage 4: 6.25Hz → 3.125Hz (TimeAttnPool + Transformer)
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


class CausalConv1d(nn.Module):
    """Causal 1D convolution with left-padding to prevent future information leakage.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Convolution kernel size.
        stride: Convolution stride.
        dilation: Convolution dilation.
        groups: Number of groups for grouped convolution.
        bias: Whether to include bias.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.stride = stride
        self.dilation = dilation
        self.kernel_size = kernel_size
        # Causal padding: (kernel_size - 1) * dilation on the left
        self.padding = (kernel_size - 1) * dilation

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding=0,  # We handle padding manually
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor [B, C, T].

        Returns:
            Output tensor [B, C_out, T_out] where T_out = ceil(T / stride).
        """
        # Left pad for causal convolution
        x = F.pad(x, (self.padding, 0))
        return self.conv(x)


class TimeAttnPool(nn.Module):
    """Time-Attention Pooling - learned attention-weighted temporal downsampling.

    Instead of naive strided convolution or average pooling, this module uses
    attention weights to selectively aggregate adjacent frames.

    Args:
        dim: Feature dimension.
        pool_size: Number of adjacent frames to merge into one (downsampling factor).
    """

    def __init__(self, dim: int, pool_size: int = 2):
        super().__init__()
        self.pool_size = pool_size
        self.query_proj = nn.Linear(dim, dim)
        self.key_proj = nn.Linear(dim, dim)
        self.scale = dim ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with attention-weighted temporal pooling.

        Args:
            x: Input tensor [B, D, T].

        Returns:
            Downsampled tensor [B, D, T // pool_size].
        """
        B, D, T = x.shape
        k = self.pool_size

        # Pad to make T divisible by pool_size
        pad_len = (k - T % k) % k
        if pad_len > 0:
            x = F.pad(x, (0, pad_len))
            T = T + pad_len

        # Reshape: [B, D, T] -> [B, T//k, k, D]
        x_grouped = rearrange(x, 'b d (n k) -> b n k d', k=k)

        # Generate queries from mean of each group (positional context)
        queries = x_grouped.mean(dim=2)  # [B, T//k, D]
        queries = self.query_proj(queries)  # [B, T//k, D]

        # Keys from each frame in the group
        keys = self.key_proj(x_grouped)  # [B, T//k, k, D]

        # Attention scores: [B, T//k, k]
        attn = torch.einsum('b n d, b n k d -> b n k', queries, keys) * self.scale
        attn = F.softmax(attn, dim=-1)

        # Weighted sum: [B, T//k, D]
        out = torch.einsum('b n k, b n k d -> b n d', attn, x_grouped)

        # Back to [B, D, T//k]
        return rearrange(out, 'b n d -> b d n')


class CausalTransformerBlock(nn.Module):
    """Causal Transformer block with causal self-attention and FFN.

    Uses causal mask to ensure no future information leakage,
    enabling streaming inference compatibility.

    Args:
        dim: Model dimension.
        num_heads: Number of attention heads.
        ffn_mult: FFN hidden dimension multiplier.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        ffn_mult: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(dim)
        ffn_dim = int(dim * ffn_mult)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, dim),
            nn.Dropout(dropout),
        )

    def _get_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Generate causal attention mask."""
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
            diagonal=1,
        )
        return mask  # True = masked positions

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor [B, D, T] (channel-first format).

        Returns:
            Output tensor [B, D, T].
        """
        # Transpose to [B, T, D] for attention
        x = rearrange(x, 'b d t -> b t d')

        # Self-attention with causal mask
        residual = x
        x_norm = self.norm1(x)
        causal_mask = self._get_causal_mask(x_norm.size(1), x_norm.device)
        attn_out, _ = self.attn(
            x_norm, x_norm, x_norm, attn_mask=causal_mask
        )
        x = residual + attn_out

        # FFN
        residual = x
        x = residual + self.ffn(self.norm2(x))

        # Back to [B, D, T]
        return rearrange(x, 'b t d -> b d t')


class EncoderStage(nn.Module):
    """Single encoder stage: downsampling + Transformer blocks.

    Args:
        in_dim: Input feature dimension.
        out_dim: Output feature dimension.
        downsample_ratio: Temporal downsampling factor.
        num_transformer_layers: Number of Transformer blocks.
        num_heads: Number of attention heads.
        use_attn_pool: If True, use TimeAttnPool; else use strided CausalConv.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        downsample_ratio: int = 2,
        num_transformer_layers: int = 4,
        num_heads: int = 8,
        use_attn_pool: bool = True,
    ):
        super().__init__()
        self.use_attn_pool = use_attn_pool

        # Dimension projection
        self.proj = CausalConv1d(in_dim, out_dim, kernel_size=3, stride=1)

        # Downsampling
        if use_attn_pool:
            self.downsample = TimeAttnPool(out_dim, pool_size=downsample_ratio)
        else:
            self.downsample = CausalConv1d(
                out_dim, out_dim, kernel_size=2 * downsample_ratio,
                stride=downsample_ratio
            )

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            CausalTransformerBlock(dim=out_dim, num_heads=num_heads)
            for _ in range(num_transformer_layers)
        ])

        self.norm = nn.GroupNorm(1, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor [B, D_in, T].

        Returns:
            Output tensor [B, D_out, T // downsample_ratio].
        """
        # Project dimension
        x = self.proj(x)
        x = F.gelu(x)

        # Downsample
        x = self.downsample(x)

        # Transformer refinement
        for block in self.transformer_blocks:
            x = block(x)

        x = self.norm(x)
        return x


class HTCE(nn.Module):
    """Hierarchical Temporal Compression Encoder.

    4-stage progressive downsampling encoder:
        - Frontend: waveform → 50Hz features (stride=320 causal conv)
        - Stage 1: 50Hz → 25Hz (downsample 2x)
        - Stage 2: 25Hz → 12.5Hz (downsample 2x)
        - Stage 3: 12.5Hz → 6.25Hz (downsample 2x)
        - Stage 4: 6.25Hz → 3.125Hz (downsample 2x)

    Input: [B, 1, T] raw waveform (16kHz)
    Output: [B, D, T'] compressed representation (T' ≈ T/5120, i.e. 3.125Hz)

    Also returns intermediate features for skip connections to decoder.

    Args:
        config: OmegaConf dict with encoder configuration.
    """

    def __init__(self, config: dict):
        super().__init__()
        input_channels = config.get('input_channels', 1)
        hidden_dims: List[int] = list(config.get('hidden_dims', [64, 128, 256, 512]))
        downsample_ratios: List[int] = list(config.get('downsample_ratios', [2, 2, 2, 2]))
        num_transformer_layers: int = config.get('num_transformer_layers', 4)
        num_heads: int = config.get('num_heads', 8)
        causal: bool = config.get('causal', True)

        assert len(hidden_dims) == len(downsample_ratios), \
            "hidden_dims and downsample_ratios must have same length"

        self.num_stages = len(hidden_dims)

        # Frontend: large stride convolution to convert waveform to 50Hz features
        # stride=320 maps 16kHz → 50Hz (16000/320 = 50)
        self.frontend = nn.Sequential(
            CausalConv1d(input_channels, hidden_dims[0], kernel_size=7, stride=5),
            nn.GELU(),
            CausalConv1d(hidden_dims[0], hidden_dims[0], kernel_size=7, stride=4),
            nn.GELU(),
            CausalConv1d(hidden_dims[0], hidden_dims[0], kernel_size=7, stride=4),
            nn.GELU(),
            CausalConv1d(hidden_dims[0], hidden_dims[0], kernel_size=7, stride=4),
            nn.GELU(),
        )
        # Total frontend stride: 5 * 4 * 4 * 4 = 320

        # Encoder stages
        self.stages = nn.ModuleList()
        for i in range(self.num_stages):
            in_dim = hidden_dims[i - 1] if i > 0 else hidden_dims[0]
            out_dim = hidden_dims[i]
            # Stage 1 uses strided conv, stages 2-4 use attention pooling
            use_attn_pool = (i > 0)

            self.stages.append(
                EncoderStage(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    downsample_ratio=downsample_ratios[i],
                    num_transformer_layers=num_transformer_layers,
                    num_heads=min(num_heads, out_dim // 32),
                    use_attn_pool=use_attn_pool,
                )
            )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through HTCE encoder.

        Args:
            x: Raw waveform input [B, 1, T] at 16kHz.

        Returns:
            Dictionary with:
                - 'z': Final compressed representation [B, D, T'].
                - 'features': List of intermediate features from each stage
                              for skip connections to decoder.
        """
        # Frontend: waveform → 50Hz features
        h = self.frontend(x)  # [B, hidden_dims[0], T//320]

        # Multi-stage compression
        features = [h]  # Store intermediate for skip connections
        for stage in self.stages:
            h = stage(h)
            features.append(h)

        return {
            'z': h,  # Final: [B, hidden_dims[-1], T // (320 * prod(downsample_ratios))]
            'features': features,  # For decoder skip connections
        }

    def get_output_length(self, input_length: int) -> int:
        """Calculate output sequence length given input waveform length."""
        # Frontend: 4 conv layers with strides [5, 4, 4, 4]
        length = input_length
        for stride in [5, 4, 4, 4]:
            length = (length + stride - 1) // stride  # Causal conv output length

        # Each stage downsamples by its ratio
        for stage in self.stages:
            if stage.use_attn_pool:
                pool_size = stage.downsample.pool_size
                length = (length + pool_size - 1) // pool_size
            else:
                length = (length + 1) // 2  # Strided conv
        return length
