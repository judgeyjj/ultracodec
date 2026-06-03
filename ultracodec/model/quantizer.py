"""Semantic Predictive Quantization (SPQ) for UltraCodec.

Implements the SPQ module which uses DPCM-style predictive coding to
reduce quantization redundancy. Instead of quantizing raw features,
we quantize prediction residuals, leveraging temporal correlation.

Key components:
    - VectorQuantize: Anti-collapse VQ with EMA codebook updates
    - ResidualVQ: Multi-codebook residual vector quantization
    - SemanticPredictor: Autoregressive predictor using Transformer
    - SPQ: Full semantic predictive quantization pipeline

Anti-collapse mechanisms:
    - Aggressive dead code reset (every 20 steps, dynamic threshold)
    - Codebook diversity loss (maximize entropy of code distribution)
    - Lower EMA decay (0.95) for faster adaptation
    - Gumbel noise for early-stage exploration
    - Periodic K-means reinit every 1000 steps
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


class VectorQuantize(nn.Module):
    """Anti-collapse Vector Quantization with EMA codebook updates.

    Uses exponential moving average to update codebook entries during
    training, with straight-through estimator for gradient propagation.
    Includes aggressive codebook reset, diversity loss, exploration noise,
    and periodic K-means reinit to prevent codebook collapse.

    Args:
        dim: Input/codebook entry dimension.
        codebook_size: Number of codebook entries.
        commitment_weight: Weight for commitment loss.
        decay: EMA decay rate for codebook update (default 0.95).
        epsilon: Epsilon for numerical stability in EMA.
        codebook_reset: Whether to reset dead codebook entries.
        reset_every: How often (in steps) to check and reset dead codes.
        reset_threshold: Dynamic threshold factor (fraction of mean usage).
        normalize: Whether to L2 normalize inputs and codebook before distance.
        diversity_weight: Weight for codebook diversity (entropy) loss.
        noise_scale: Initial scale for Gumbel exploration noise.
        kmeans_reinit_every: Steps between full K-means reinit (0=disabled).
    """

    def __init__(
        self,
        dim: int,
        codebook_size: int = 1024,
        commitment_weight: float = 0.25,
        decay: float = 0.95,
        epsilon: float = 1e-5,
        codebook_reset: bool = True,
        reset_every: int = 20,
        reset_threshold: float = 0.1,
        normalize: bool = True,
        diversity_weight: float = 0.1,
        noise_scale: float = 0.1,
        kmeans_reinit_every: int = 1000,
    ):
        super().__init__()
        self.dim = dim
        self.codebook_size = codebook_size
        self.commitment_weight = commitment_weight
        self.decay = decay
        self.epsilon = epsilon
        self.codebook_reset = codebook_reset
        self.reset_every = reset_every
        self.reset_threshold = reset_threshold
        self.normalize = normalize
        self.diversity_weight = diversity_weight
        self.noise_scale = noise_scale
        self.kmeans_reinit_every = kmeans_reinit_every

        # Codebook
        self.embedding = nn.Embedding(codebook_size, dim)
        nn.init.uniform_(self.embedding.weight, -1.0 / codebook_size, 1.0 / codebook_size)

        # EMA tracking
        self.register_buffer('cluster_size', torch.zeros(codebook_size))
        self.register_buffer('embed_avg', self.embedding.weight.data.clone())
        self.register_buffer('inited', torch.tensor(False))
        self.register_buffer('steps', torch.tensor(0, dtype=torch.long))

    def _init_codebook(self, data: torch.Tensor) -> None:
        """Initialize codebook from first batch of data."""
        if self.inited:
            return
        # Use random subset of data to initialize
        n = data.shape[0]
        if n >= self.codebook_size:
            indices = torch.randperm(n, device=data.device)[:self.codebook_size]
            self.embedding.weight.data.copy_(data[indices])
        else:
            # Repeat data to fill codebook
            repeats = (self.codebook_size + n - 1) // n
            expanded = data.repeat(repeats, 1)[:self.codebook_size]
            noise = torch.randn_like(expanded) * 0.01
            self.embedding.weight.data.copy_(expanded + noise)
        self.embed_avg.data.copy_(self.embedding.weight.data)
        self.cluster_size.data.fill_(1.0)
        self.inited.fill_(True)

    def _maybe_reset_codes(self, flat_inputs: torch.Tensor) -> None:
        """Reset dead codes with dynamic threshold.

        Dead codes are those with cluster_size < mean_usage * reset_threshold.
        They are replaced with randomly sampled encoder outputs plus small
        noise to avoid multiple dead codes collapsing to the same vector.
        """
        if not self.codebook_reset:
            return
        if self.steps % self.reset_every != 0:
            return

        # Dynamic threshold: codes used less than 10% of mean are dead
        mean_usage = self.cluster_size.mean()
        dead_mask = self.cluster_size < (mean_usage * self.reset_threshold)
        n_dead = dead_mask.sum().item()
        if n_dead == 0:
            return

        # Sample random encoder outputs as replacements
        n_samples = flat_inputs.size(0)
        random_indices = torch.randint(0, n_samples, (n_dead,), device=flat_inputs.device)
        new_codes = flat_inputs[random_indices].detach().to(self.embedding.weight.dtype)

        # Add small noise to each replacement to avoid duplicates
        noise = torch.randn_like(new_codes) * 0.02
        new_codes = new_codes + noise

        self.embedding.weight.data[dead_mask] = new_codes
        # Reset EMA stats for revived codes to mean level
        self.cluster_size[dead_mask] = mean_usage
        self.embed_avg.data[dead_mask] = new_codes * mean_usage

    def _compute_diversity_loss(self) -> torch.Tensor:
        """Compute diversity loss: encourages uniform codebook usage.

        Returns normalized negative entropy of code distribution.
        diversity_loss = (max_entropy - entropy) / max_entropy in [0, 1].
        """
        # Use cluster_size as proxy for code usage distribution
        probs = self.cluster_size / (self.cluster_size.sum() + 1e-8)
        # Clamp for numerical stability
        probs = probs.clamp(min=1e-8)
        entropy = -(probs * torch.log(probs)).sum()
        max_entropy = math.log(self.codebook_size)
        # Normalized so that 0 = perfect uniform, 1 = fully collapsed
        diversity_loss = (max_entropy - entropy) / max_entropy
        return diversity_loss

    def _kmeans_reinit(self, data: torch.Tensor) -> None:
        """Run mini K-means on encoder outputs to reinitialize codebook.

        Uses 10 iterations of K-means on the provided data to find
        better codebook centroids. Only updates codes that converge.
        """
        n = data.shape[0]
        if n < self.codebook_size:
            return  # Not enough data for K-means

        # Subsample if too much data
        max_samples = min(n, self.codebook_size * 32)
        if n > max_samples:
            indices = torch.randperm(n, device=data.device)[:max_samples]
            data = data[indices]
            n = max_samples

        # Initialize centroids from data
        init_indices = torch.randperm(n, device=data.device)[:self.codebook_size]
        centroids = data[init_indices].clone()

        # Run K-means iterations
        for _ in range(10):
            if self.normalize:
                data_norm = F.normalize(data, p=2, dim=-1)
                cent_norm = F.normalize(centroids, p=2, dim=-1)
                dists = 2.0 - 2.0 * (data_norm @ cent_norm.t())
            else:
                dists = (
                    data.pow(2).sum(dim=-1, keepdim=True)
                    - 2 * data @ centroids.t()
                    + centroids.pow(2).sum(dim=-1, keepdim=True).t()
                )
            assignments = dists.argmin(dim=-1)
            one_hot = F.one_hot(assignments, self.codebook_size).float()
            counts = one_hot.sum(dim=0)
            new_centroids = one_hot.t() @ data

            # Update only non-empty clusters
            mask = counts > 0
            if mask.any():
                centroids[mask] = new_centroids[mask] / counts[mask].unsqueeze(1)

        # Update codebook
        self.embedding.weight.data.copy_(centroids.to(self.embedding.weight.dtype))
        self.embed_avg.data.copy_(centroids.to(self.embed_avg.dtype))
        self.cluster_size.data.fill_(1.0)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass with quantization.

        Args:
            x: Input tensor [B, D, T] or [B*T, D].

        Returns:
            Tuple of:
                - quantized: Quantized output (same shape as input)
                - codes: Codebook indices [B*T] or [B, T]
                - loss: Commitment + diversity loss
        """
        need_reshape = x.dim() == 3
        if need_reshape:
            B, D, T = x.shape
            x_flat = rearrange(x, 'b d t -> (b t) d')
        else:
            x_flat = x

        # Initialize codebook on first forward
        if self.training and not self.inited:
            self._init_codebook(x_flat.detach())

        # Optionally reset dead codes before distance computation
        if self.training:
            self._maybe_reset_codes(x_flat.detach())
            self.steps += 1

        # Periodic K-means reinit
        if self.training and self.kmeans_reinit_every > 0:
            if self.steps > 0 and self.steps % self.kmeans_reinit_every == 0:
                self._kmeans_reinit(x_flat.detach())

        # Compute distances with optional L2 normalization
        if self.normalize:
            x_norm = F.normalize(x_flat, p=2, dim=-1)
            cb_norm = F.normalize(self.embedding.weight, p=2, dim=-1)
            # ||x_norm - cb_norm||^2 = 2 - 2 * x_norm @ cb_norm^T
            dists = 2.0 - 2.0 * (x_norm @ cb_norm.t())
        else:
            # Compute distances: ||x - e||^2 = ||x||^2 - 2*x*e^T + ||e||^2
            dists = (
                x_flat.pow(2).sum(dim=-1, keepdim=True)
                - 2 * x_flat @ self.embedding.weight.t()
                + self.embedding.weight.pow(2).sum(dim=-1, keepdim=True).t()
            )

        # Add Gumbel noise for exploration (decays over training)
        if self.training and self.noise_scale > 0:
            # Noise decays linearly over the first 1000 steps
            decay_factor = max(0.0, 1.0 - float(self.steps) / 1000.0)
            if decay_factor > 0:
                noise = torch.randn_like(dists) * self.noise_scale * decay_factor
                dists = dists + noise

        # Find nearest codebook entry
        codes = dists.argmin(dim=-1)  # [B*T]
        quantized_flat = self.embedding(codes)  # [B*T, D]

        # EMA update (training only)
        if self.training:
            with torch.no_grad():
                one_hot = F.one_hot(codes, self.codebook_size).float()  # [B*T, K]
                new_cluster_size = one_hot.sum(dim=0)  # [K]

                # Use normalized vectors for EMA if normalize mode
                if self.normalize:
                    ema_input = F.normalize(x_flat.detach(), p=2, dim=-1).float()
                else:
                    ema_input = x_flat.detach().float()
                new_embed_sum = one_hot.t() @ ema_input  # [K, D] in fp32

                self.cluster_size.data.mul_(self.decay).add_(
                    new_cluster_size, alpha=1 - self.decay
                )
                self.embed_avg.data.mul_(self.decay).add_(
                    new_embed_sum, alpha=1 - self.decay
                )

                # Laplace smoothing
                n = self.cluster_size.sum()
                cluster_size = (
                    (self.cluster_size + self.epsilon)
                    / (n + self.codebook_size * self.epsilon) * n
                )
                embed_normalized = self.embed_avg / cluster_size.unsqueeze(1)
                self.embedding.weight.data.copy_(embed_normalized)

        # Losses
        commitment_loss = F.mse_loss(x_flat.detach(), quantized_flat) \
            + self.commitment_weight * F.mse_loss(x_flat, quantized_flat.detach())

        # Diversity loss (maximize codebook entropy)
        if self.training and self.diversity_weight > 0:
            diversity_loss = self._compute_diversity_loss()
            total_loss = commitment_loss + self.diversity_weight * diversity_loss
        else:
            total_loss = commitment_loss

        # Straight-through estimator
        quantized_flat = x_flat + (quantized_flat - x_flat).detach()

        if need_reshape:
            quantized = rearrange(quantized_flat, '(b t) d -> b d t', b=B, t=T)
            codes = codes.view(B, T)
        else:
            quantized = quantized_flat

        return quantized, codes, total_loss

    @torch.no_grad()
    def get_diversity_loss(self) -> torch.Tensor:
        """Get current diversity loss value (for logging)."""
        return self._compute_diversity_loss()

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """Decode codebook indices back to vectors.

        Args:
            codes: Codebook indices [B, T] or [N].

        Returns:
            Decoded vectors [B, D, T] or [N, D].
        """
        need_reshape = codes.dim() == 2
        if need_reshape:
            B, T = codes.shape
            codes_flat = codes.view(-1)
        else:
            codes_flat = codes

        quantized = self.embedding(codes_flat)

        if need_reshape:
            quantized = rearrange(quantized, '(b t) d -> b d t', b=B, t=T)

        return quantized


class ResidualVQ(nn.Module):
    """Residual Vector Quantization with multiple codebook layers.

    Each layer quantizes the residual from the previous layer,
    progressively reducing quantization error.

    Args:
        dim: Input dimension.
        codebook_size: Number of entries per codebook.
        num_codebooks: Number of residual quantization layers.
        codebook_dim: Dimension of codebook entries (projects if != dim).
        commitment_weight: Commitment loss weight.
        decay: EMA decay rate for codebook update.
        codebook_reset: Whether to reset dead codebook entries.
        reset_every: How often (in steps) to check and reset dead codes.
        reset_threshold: Dynamic threshold factor for dead code detection.
        normalize: Whether to L2 normalize before distance computation.
        diversity_weight: Weight for codebook diversity loss.
        noise_scale: Initial scale for exploration noise.
        kmeans_reinit_every: Steps between full K-means reinit.
    """

    def __init__(
        self,
        dim: int,
        codebook_size: int = 1024,
        num_codebooks: int = 8,
        codebook_dim: Optional[int] = None,
        commitment_weight: float = 0.25,
        decay: float = 0.95,
        codebook_reset: bool = True,
        reset_every: int = 20,
        reset_threshold: float = 0.1,
        normalize: bool = True,
        diversity_weight: float = 0.1,
        noise_scale: float = 0.1,
        kmeans_reinit_every: int = 1000,
    ):
        super().__init__()
        self.dim = dim
        self.codebook_dim = codebook_dim or dim
        self.num_codebooks = num_codebooks

        # Project to/from codebook dimension if needed
        self.project_in = nn.Linear(dim, self.codebook_dim) if self.codebook_dim != dim else nn.Identity()
        self.project_out = nn.Linear(self.codebook_dim, dim) if self.codebook_dim != dim else nn.Identity()

        # VQ layers
        self.layers = nn.ModuleList([
            VectorQuantize(
                dim=self.codebook_dim,
                codebook_size=codebook_size,
                commitment_weight=commitment_weight,
                decay=decay,
                codebook_reset=codebook_reset,
                reset_every=reset_every,
                reset_threshold=reset_threshold,
                normalize=normalize,
                diversity_weight=diversity_weight,
                noise_scale=noise_scale,
                kmeans_reinit_every=kmeans_reinit_every,
            )
            for _ in range(num_codebooks)
        ])

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass with residual quantization.

        Args:
            x: Input tensor [B, D, T].

        Returns:
            Tuple of:
                - quantized: Sum of all quantized layers [B, D, T]
                - codes: Codebook indices [B, num_codebooks, T]
                - total_loss: Sum of all layer losses (commitment + diversity)
                - diversity_loss: Sum of diversity losses across layers
        """
        B, D, T = x.shape

        # Project to codebook dim
        x_proj = rearrange(x, 'b d t -> (b t) d')
        x_proj = self.project_in(x_proj)
        x_proj = rearrange(x_proj, '(b t) d -> b d t', b=B, t=T)

        residual = x_proj
        quantized_sum = torch.zeros_like(x_proj)
        all_codes = []
        total_loss = torch.tensor(0.0, device=x.device)
        diversity_loss_total = torch.tensor(0.0, device=x.device)

        for layer in self.layers:
            quantized, codes, loss = layer(residual)
            residual = residual - quantized.detach()
            quantized_sum = quantized_sum + quantized
            all_codes.append(codes)
            total_loss = total_loss + loss
            # Accumulate diversity loss for logging
            if self.training:
                diversity_loss_total = diversity_loss_total + layer.get_diversity_loss()

        # Project back to original dim
        quantized_out = rearrange(quantized_sum, 'b d t -> (b t) d')
        quantized_out = self.project_out(quantized_out)
        quantized_out = rearrange(quantized_out, '(b t) d -> b d t', b=B, t=T)

        codes = torch.stack(all_codes, dim=1)  # [B, num_codebooks, T]
        return quantized_out, codes, total_loss, diversity_loss_total

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """Decode from codebook indices.

        Args:
            codes: Indices [B, num_codebooks, T].

        Returns:
            Decoded tensor [B, D, T].
        """
        B, num_cb, T = codes.shape
        quantized_sum = torch.zeros(B, self.codebook_dim, T, device=codes.device)

        for i, layer in enumerate(self.layers):
            quantized = layer.decode(codes[:, i])  # [B, codebook_dim, T]
            quantized_sum = quantized_sum + quantized

        # Project back
        out = rearrange(quantized_sum, 'b d t -> (b t) d')
        out = self.project_out(out)
        out = rearrange(out, '(b t) d -> b d t', b=B, t=T)
        return out


class SemanticPredictor(nn.Module):
    """Semantic Predictor - Autoregressive Transformer for next-frame prediction.

    Implements DPCM-style prediction: uses past frames to predict the current
    frame, so only the prediction residual needs to be quantized.

    Args:
        dim: Feature dimension.
        prediction_order: Number of past frames used for prediction.
        num_layers: Number of Transformer layers.
        num_heads: Number of attention heads.
    """

    def __init__(
        self,
        dim: int,
        prediction_order: int = 2,
        num_layers: int = 4,
        num_heads: int = 8,
    ):
        super().__init__()
        self.dim = dim
        self.prediction_order = prediction_order

        # Positional embedding for the context window
        self.pos_embed = nn.Parameter(torch.randn(1, prediction_order, dim) * 0.02)

        # Transformer layers for prediction
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Output projection
        self.output_proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict current frame from past frames.

        Uses a sliding window of `prediction_order` past frames to predict
        each frame autoregressively.

        Args:
            x: Input sequence [B, D, T].

        Returns:
            Predictions [B, D, T] where prediction[t] is based on x[t-order:t].
        """
        B, D, T = x.shape
        order = self.prediction_order

        # Pad with zeros at the beginning for the first `order` frames
        x_padded = F.pad(x, (order, 0))  # [B, D, T + order]

        # Extract sliding windows: for each position t, get [t, t+1, ..., t+order-1]
        # These represent the `order` frames *before* position t (after padding)
        predictions = []

        # Efficient batched processing
        x_transposed = rearrange(x_padded, 'b d t -> b t d')  # [B, T+order, D]

        # Create context windows for all positions at once
        # For position t in output, context is x_padded[:, :, t:t+order]
        contexts = x_transposed.unfold(1, order, 1)  # [B, T+1, D, order]
        contexts = contexts[:, :T]  # [B, T, D, order]
        contexts = rearrange(contexts, 'b t d o -> (b t) o d')  # [B*T, order, D]

        # Add positional embeddings
        contexts = contexts + self.pos_embed

        # Pass through transformer
        preds = self.transformer(contexts)  # [B*T, order, D]

        # Take the last position's output as prediction
        preds = preds[:, -1, :]  # [B*T, D]
        preds = self.output_proj(preds)

        # Reshape back
        preds = rearrange(preds, '(b t) d -> b d t', b=B, t=T)
        return preds


class SPQ(nn.Module):
    """Semantic Predictive Quantization module.

    Pipeline:
        1. Use SemanticPredictor to predict current frame from history
        2. Compute residual = actual - predicted
        3. Quantize residual with ResidualVQ (not raw features)
        4. Reconstruct = predicted + quantized_residual

    Advantage: Residual distribution is concentrated near zero,
    requiring fewer codebook entries for accurate quantization.

    Args:
        config: OmegaConf dict with quantizer configuration.
        input_dim: Dimension of encoder output features.
    """

    def __init__(self, config: dict, input_dim: int = 512):
        super().__init__()
        codebook_size: int = config.get('codebook_size', 1024)
        codebook_dim: int = config.get('codebook_dim', 128)
        num_codebooks: int = config.get('num_codebooks', 8)
        commitment_weight: float = config.get('commitment_weight', 0.25)
        prediction_order: int = config.get('prediction_order', 2)
        predictor_layers: int = config.get('predictor_layers', 4)
        predictor_heads: int = config.get('predictor_heads', 8)
        ema_decay: float = config.get('ema_decay', 0.95)
        codebook_reset: bool = config.get('codebook_reset', True)
        reset_every: int = config.get('reset_every', 20)
        reset_threshold: float = config.get('reset_threshold', 0.1)
        normalize: bool = config.get('normalize', True)
        diversity_weight: float = config.get('diversity_weight', 0.1)
        noise_scale: float = config.get('noise_scale', 0.1)
        kmeans_reinit_every: int = config.get('kmeans_reinit_every', 1000)

        self.input_dim = input_dim
        self.codebook_dim = codebook_dim
        self.num_codebooks = num_codebooks

        # Semantic predictor (operates in codebook_dim space)
        self.pre_proj = nn.Linear(input_dim, codebook_dim) if input_dim != codebook_dim else nn.Identity()
        self.predictor = SemanticPredictor(
            dim=codebook_dim,
            prediction_order=prediction_order,
            num_layers=predictor_layers,
            num_heads=predictor_heads,
        )

        # Residual VQ (quantizes prediction residuals)
        self.rvq = ResidualVQ(
            dim=codebook_dim,
            codebook_size=codebook_size,
            num_codebooks=num_codebooks,
            codebook_dim=codebook_dim,
            commitment_weight=commitment_weight,
            decay=ema_decay,
            codebook_reset=codebook_reset,
            reset_every=reset_every,
            reset_threshold=reset_threshold,
            normalize=normalize,
            diversity_weight=diversity_weight,
            noise_scale=noise_scale,
            kmeans_reinit_every=kmeans_reinit_every,
        )

        # Post projection back to input_dim
        self.post_proj = nn.Linear(codebook_dim, input_dim) if input_dim != codebook_dim else nn.Identity()

    def forward(
        self, x: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with predictive quantization.

        Args:
            x: Input features [B, D, T] from encoder.

        Returns:
            Dictionary with:
                - 'quantized': Reconstructed features [B, D, T]
                - 'codes': Codebook indices [B, num_codebooks, T]
                - 'commitment_loss': VQ commitment + diversity loss
                - 'prediction_loss': Prediction accuracy loss
                - 'diversity_loss': Codebook diversity loss (for logging)
                - 'residuals': Raw residuals before quantization [B, codebook_dim, T]
                - 'residual_norms': Per-frame residual norms [B, T]
        """
        B, D, T = x.shape

        # Project to codebook dimension
        x_proj = rearrange(x, 'b d t -> (b t) d')
        x_proj = self.pre_proj(x_proj)
        x_proj = rearrange(x_proj, '(b t) d -> b d t', b=B, t=T)

        # Predict current frame from past frames
        predictions = self.predictor(x_proj)  # [B, codebook_dim, T]

        # Compute residual
        residuals = x_proj - predictions  # [B, codebook_dim, T]

        # Compute residual norms (useful for AFR)
        residual_norms = residuals.norm(dim=1)  # [B, T]

        # Quantize residuals (now returns 4 values)
        quantized_residuals, codes, commitment_loss, diversity_loss = self.rvq(residuals)

        # Reconstruct: prediction + quantized residual
        reconstructed = predictions + quantized_residuals  # [B, codebook_dim, T]

        # Project back to input dimension
        recon_out = rearrange(reconstructed, 'b d t -> (b t) d')
        recon_out = self.post_proj(recon_out)
        recon_out = rearrange(recon_out, '(b t) d -> b d t', b=B, t=T)

        # Prediction loss (how well predictor predicts)
        prediction_loss = F.mse_loss(predictions, x_proj.detach())

        return {
            'quantized': recon_out,
            'codes': codes,
            'commitment_loss': commitment_loss,
            'prediction_loss': prediction_loss,
            'diversity_loss': diversity_loss,
            'residuals': residuals,
            'residual_norms': residual_norms,
        }

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode features to discrete codes (inference).

        Args:
            x: Input features [B, D, T].

        Returns:
            Codebook indices [B, num_codebooks, T].
        """
        B, D, T = x.shape

        # Project
        x_proj = rearrange(x, 'b d t -> (b t) d')
        x_proj = self.pre_proj(x_proj)
        x_proj = rearrange(x_proj, '(b t) d -> b d t', b=B, t=T)

        # Predict and compute residual
        predictions = self.predictor(x_proj)
        residuals = x_proj - predictions

        # Quantize residuals
        _, codes, _, _ = self.rvq(residuals)
        return codes

    def decode(self, codes: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Decode from discrete codes back to features.

        For proper decoding, we need to autoregressively reconstruct since
        predictions depend on previously decoded frames.

        Args:
            codes: Codebook indices [B, num_codebooks, T].
            context: Optional prior context for prediction [B, D, order].

        Returns:
            Reconstructed features [B, D, T].
        """
        B, num_cb, T = codes.shape

        # Decode quantized residuals from codes
        quantized_residuals = self.rvq.decode(codes)  # [B, codebook_dim, T]

        # Autoregressive reconstruction
        order = self.predictor.prediction_order
        reconstructed = torch.zeros(B, self.codebook_dim, T, device=codes.device)

        # Initialize context buffer
        if context is not None:
            ctx = rearrange(context, 'b d t -> (b t) d')
            ctx = self.pre_proj(ctx)
            ctx = rearrange(ctx, '(b t) d -> b d t', b=B)
            buffer = ctx
        else:
            buffer = torch.zeros(B, self.codebook_dim, order, device=codes.device)

        # Decode frame by frame
        for t in range(T):
            # Get context for prediction (last `order` frames)
            if t == 0:
                ctx_frames = buffer[:, :, -order:]
            else:
                available = reconstructed[:, :, max(0, t - order):t]
                if available.size(2) < order:
                    pad_frames = buffer[:, :, -(order - available.size(2)):]
                    ctx_frames = torch.cat([pad_frames, available], dim=2)
                else:
                    ctx_frames = available

            # Predict using context
            ctx_input = rearrange(ctx_frames, 'b d o -> b o d')
            ctx_input = ctx_input + self.predictor.pos_embed[:, :ctx_input.size(1)]
            pred = self.predictor.transformer(ctx_input)
            pred = pred[:, -1, :]  # [B, D]
            pred = self.predictor.output_proj(pred)

            # Reconstruct: prediction + quantized residual
            reconstructed[:, :, t] = pred + quantized_residuals[:, :, t]

        # Project back to input dim
        out = rearrange(reconstructed, 'b d t -> (b t) d')
        out = self.post_proj(out)
        out = rearrange(out, '(b t) d -> b d t', b=B, t=T)
        return out
