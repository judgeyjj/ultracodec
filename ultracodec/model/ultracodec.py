"""UltraCodec main model - Ultra-low frame rate speech codec for LLMs.

Integrates all components:
    - HTCE: Hierarchical Temporal Compression Encoder
    - SPQ: Semantic Predictive Quantizer
    - AFR: Adaptive Frame Rate Gate
    - CascadedDecoder: Coarse-to-fine waveform reconstruction

Target: Compress 16kHz speech to ~3.125Hz discrete tokens (< 5 tokens/sec),
enabling efficient LLM processing with minimal context consumption.
"""

from __future__ import annotations

import io
import struct
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .adaptive_framerate import AdaptiveFrameRateGate
from .decoder import CascadedDecoder
from .encoder import HTCE
from .quantizer import SPQ


class UltraCodec(nn.Module):
    """UltraCodec - Ultra-low frame rate neural speech codec.

    Architecture: HTCE Encoder → SPQ Quantizer → AFR Gate → Cascaded Decoder

    Provides interfaces for:
        - forward(x): Full forward pass (training)
        - encode(x): Encode to discrete tokens
        - decode(codes): Decode from tokens
        - compress(audio): Full compression pipeline
        - decompress(data): Full decompression pipeline

    Args:
        config: OmegaConf dict with full model configuration (from base.yaml 'model' section).
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.sample_rate = config.get('sample_rate', 16000)
        self.channels = config.get('channels', 1)

        # Extract sub-configs
        encoder_config = config.get('encoder', {})
        quantizer_config = config.get('quantizer', {})
        afr_config = config.get('afr', {})
        decoder_config = config.get('decoder', {})

        # Encoder: waveform → compressed features
        self.encoder = HTCE(encoder_config)

        # Get encoder output dimension
        encoder_dims = list(encoder_config.get('hidden_dims', [64, 128, 256, 512]))
        encoder_output_dim = encoder_dims[-1]

        # Quantizer: continuous features → discrete codes
        self.quantizer = SPQ(quantizer_config, input_dim=encoder_output_dim)

        # Adaptive Frame Rate gate
        self.afr = AdaptiveFrameRateGate(afr_config, input_dim=encoder_output_dim)

        # Decoder: quantized features → waveform
        self.decoder = CascadedDecoder(decoder_config, encoder_dims=encoder_dims)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Full forward pass for training.

        Args:
            x: Raw waveform [B, 1, T] at 16kHz.

        Returns:
            Dictionary with:
                - 'x_hat': Reconstructed waveform [B, 1, T]
                - 'codes': Discrete codebook indices [B, num_codebooks, T']
                - 'commitment_loss': VQ commitment loss
                - 'codebook_loss': Codebook/prediction loss
                - 'gate_decisions': AFR gate values [B, T']
                - 'keep_ratio': Proportion of frames kept by AFR
                - 'rate_loss': AFR rate regularization loss
                - 'encoder_features': Intermediate features for auxiliary losses
        """
        target_length = x.size(2)

        # 1. Encode
        encoder_out = self.encoder(x)
        z = encoder_out['z']  # [B, D, T']
        features = encoder_out['features']

        # 2. Quantize (SPQ: predict + residual quantize)
        quant_out = self.quantizer(z)
        quantized = quant_out['quantized']  # [B, D, T']
        codes = quant_out['codes']
        commitment_loss = quant_out['commitment_loss']
        prediction_loss = quant_out['prediction_loss']
        diversity_loss = quant_out.get('diversity_loss', None)
        residual_norms = quant_out['residual_norms']

        # 3. Adaptive Frame Rate gating
        afr_out = self.afr(quantized, residual_norms=residual_norms)
        gated_features = afr_out['output']  # [B, D, T']
        gate_decisions = afr_out['gate_decisions']
        keep_ratio = afr_out['keep_ratio']
        rate_loss = afr_out['rate_loss']

        # 4. Decode (cascaded coarse-to-fine)
        decoder_out = self.decoder(
            gated_features,
            encoder_features=features,
            target_length=target_length,
        )
        x_hat = decoder_out['audio']

        # Ensure output matches input length
        if x_hat.size(2) < target_length:
            x_hat = F.pad(x_hat, (0, target_length - x_hat.size(2)))
        elif x_hat.size(2) > target_length:
            x_hat = x_hat[:, :, :target_length]

        result = {
            'x_hat': x_hat,
            'codes': codes,
            'commitment_loss': commitment_loss,
            'codebook_loss': prediction_loss,
            'gate_decisions': gate_decisions,
            'keep_ratio': keep_ratio,
            'rate_loss': rate_loss,
            'encoder_features': features,
        }
        if diversity_loss is not None:
            result['diversity_loss'] = diversity_loss
        return result

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Encode audio to discrete codes (inference).

        Args:
            x: Raw waveform [B, 1, T] at 16kHz.

        Returns:
            Dictionary with:
                - 'codes': Discrete codebook indices [B, num_codebooks, T']
                - 'gate': Gate decisions [B, T']
        """
        self.eval()

        # Encode
        encoder_out = self.encoder(x)
        z = encoder_out['z']

        # Quantize
        codes = self.quantizer.encode(z)

        # AFR gate
        afr_out = self.afr(z)
        gate = afr_out['gate_decisions']

        return {
            'codes': codes,
            'gate': gate,
        }

    @torch.no_grad()
    def decode(self, codes: torch.Tensor, gate: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Decode from discrete codes back to waveform.

        Args:
            codes: Codebook indices [B, num_codebooks, T'].
            gate: Optional gate decisions [B, T'] for frame selection.

        Returns:
            Reconstructed waveform [B, 1, T].
        """
        self.eval()

        # Decode quantized features from codes
        quantized = self.quantizer.decode(codes)

        # Apply gate if provided
        if gate is not None:
            gate_expanded = gate.unsqueeze(1)
            quantized = quantized * gate_expanded

        # Decode to waveform
        decoder_out = self.decoder(quantized)
        return decoder_out['audio']

    @torch.no_grad()
    def compress(self, x: torch.Tensor) -> bytes:
        """Complete compression pipeline - encode audio to bytes.

        Useful for computing actual bitrate.

        Args:
            x: Raw waveform [B, 1, T] at 16kHz. Only B=1 supported.

        Returns:
            Compressed bytes representation.
        """
        self.eval()
        assert x.size(0) == 1, "Compress only supports batch_size=1"

        # Encode to codes
        result = self.encode(x)
        codes = result['codes']  # [1, num_codebooks, T']
        gate = result['gate']  # [1, T']

        # Pack to bytes
        buffer = io.BytesIO()

        # Header: num_codebooks, T', codebook_size (for bit width calculation)
        num_cb = codes.size(1)
        seq_len = codes.size(2)
        buffer.write(struct.pack('III', num_cb, seq_len, self.sample_rate))

        # Write gate decisions as bit-packed
        gate_binary = (gate[0] > 0.5).cpu().numpy().astype('uint8')
        buffer.write(gate_binary.tobytes())

        # Write codes (16-bit per code index)
        codes_np = codes[0].cpu().numpy().astype('uint16')
        buffer.write(codes_np.tobytes())

        return buffer.getvalue()

    @torch.no_grad()
    def decompress(self, compressed: bytes, device: torch.device = None) -> torch.Tensor:
        """Decompress bytes back to waveform.

        Args:
            compressed: Bytes from compress().
            device: Target device.

        Returns:
            Reconstructed waveform [1, 1, T].
        """
        self.eval()
        if device is None:
            device = next(self.parameters()).device

        buffer = io.BytesIO(compressed)

        # Read header
        num_cb, seq_len, sr = struct.unpack('III', buffer.read(12))

        # Read gate
        gate_bytes = buffer.read(seq_len)
        gate = torch.tensor(
            list(gate_bytes), dtype=torch.float32, device=device
        ).unsqueeze(0)  # [1, T']

        # Read codes
        import numpy as np
        codes_flat = np.frombuffer(buffer.read(), dtype=np.uint16)
        codes = torch.tensor(
            codes_flat.reshape(num_cb, seq_len),
            dtype=torch.long, device=device
        ).unsqueeze(0)  # [1, num_cb, T']

        # Decode
        return self.decode(codes, gate=gate)

    def stream_encode(self, chunk: torch.Tensor) -> Optional[Dict[str, torch.Tensor]]:
        """Stream-compatible encoding for real-time processing.

        Processes one chunk at a time. Due to causal architecture,
        each chunk can be processed independently.

        Args:
            chunk: Audio chunk [1, 1, chunk_size].

        Returns:
            Dictionary with codes and gate for this chunk, or None if
            chunk is too short to produce output.
        """
        return self.encode(chunk)

    def stream_decode(self, codes: torch.Tensor, gate: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Stream-compatible decoding.

        Args:
            codes: Codes for one chunk [1, num_codebooks, T'].
            gate: Gate decisions [1, T'].

        Returns:
            Decoded audio chunk [1, 1, T_audio].
        """
        return self.decode(codes, gate=gate)

    def get_num_params(self, non_embedding: bool = True) -> int:
        """Get total number of parameters.

        Args:
            non_embedding: If True, exclude embedding parameters.

        Returns:
            Total parameter count.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            # Subtract codebook embeddings
            for module in self.modules():
                if hasattr(module, 'embedding') and isinstance(module.embedding, nn.Embedding):
                    n_params -= module.embedding.weight.numel()
        return n_params

    def get_bitrate(self, audio_length_sec: float, codes: torch.Tensor, gate: Optional[torch.Tensor] = None) -> float:
        """Calculate actual bitrate.

        Args:
            audio_length_sec: Duration of audio in seconds.
            codes: Encoded codes [B, num_codebooks, T'].
            gate: Gate decisions [B, T'].

        Returns:
            Bitrate in kbps.
        """
        import math
        num_codebooks = codes.size(1)
        seq_len = codes.size(2)

        # Bits per code = log2(codebook_size)
        codebook_size = self.config.get('quantizer', {}).get('codebook_size', 1024)
        bits_per_code = math.log2(codebook_size)

        # If gate is provided, only count kept frames
        if gate is not None:
            effective_frames = (gate > 0.5).float().sum(dim=-1).mean().item()
        else:
            effective_frames = seq_len

        total_bits = effective_frames * num_codebooks * bits_per_code
        bitrate_kbps = total_bits / (audio_length_sec * 1000)
        return bitrate_kbps
