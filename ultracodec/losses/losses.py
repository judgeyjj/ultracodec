"""UltraCodec loss functions.

Implements all reconstruction, quantization, adversarial, and regularization
losses described in Section 3.7 of the paper:

    L_total = L_recon + λ_q * L_vq + λ_p * L_pred + λ_adv * L_adv
              + λ_r * L_rate + λ_lm * L_lm + λ_sem * L_sem

The combined :class:`UltraCodecLoss` reads loss weights and stage-aware flags
from the OmegaConf training config.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Reconstruction losses
# ---------------------------------------------------------------------------
def _stft(
    wav: torch.Tensor,
    n_fft: int,
    hop_length: int,
    win_length: int,
) -> torch.Tensor:
    """Compute the magnitude spectrogram of ``wav`` with a Hann window.

    Args:
        wav: Waveform tensor of shape ``[B, T]`` or ``[B, 1, T]``.
        n_fft: FFT size.
        hop_length: Hop length between frames.
        win_length: Window length (must be ``<= n_fft``).

    Returns:
        Magnitude spectrogram tensor of shape ``[B, F, N]``.
    """
    if wav.dim() == 3:
        wav = wav.squeeze(1)
    # cuFFT doesn't support BFloat16, cast to float32 for STFT
    input_dtype = wav.dtype
    if wav.dtype == torch.bfloat16 or wav.dtype == torch.float16:
        wav = wav.float()
    window = torch.hann_window(win_length, device=wav.device, dtype=wav.dtype)
    spec = torch.stft(
        wav,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=True,
        pad_mode="reflect",
        normalized=False,
        return_complex=True,
    )
    mag = spec.abs()
    # Cast back to original dtype
    if input_dtype != mag.dtype:
        mag = mag.to(input_dtype)
    return mag


class MultiResolutionSTFTLoss(nn.Module):
    """Multi-resolution STFT loss combining spectral convergence and log-magnitude L1.

    For each FFT size ``n_fft`` in ``fft_sizes``:
        * spectral convergence:  ``||S - S_hat||_F / ||S||_F``
        * log magnitude L1:       ``mean |log(S+eps) - log(S_hat+eps)|``

    Args:
        fft_sizes: List of FFT sizes (default ``[512, 1024, 2048]``).
        hop_sizes: Optional list of hop sizes (default ``n_fft // 4``).
        win_sizes: Optional list of window sizes (default ``n_fft``).
        eps: Numerical floor for the logarithm.
    """

    def __init__(
        self,
        fft_sizes: Sequence[int] = (512, 1024, 2048),
        hop_sizes: Optional[Sequence[int]] = None,
        win_sizes: Optional[Sequence[int]] = None,
        eps: float = 1e-7,
    ) -> None:
        super().__init__()
        if hop_sizes is None:
            hop_sizes = [n // 4 for n in fft_sizes]
        if win_sizes is None:
            win_sizes = list(fft_sizes)
        assert len(fft_sizes) == len(hop_sizes) == len(win_sizes)
        self.fft_sizes = list(fft_sizes)
        self.hop_sizes = list(hop_sizes)
        self.win_sizes = list(win_sizes)
        self.eps = eps

    def forward(self, x_hat: torch.Tensor, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute the multi-resolution STFT loss.

        Args:
            x_hat: Reconstructed waveform ``[B, 1, T]`` or ``[B, T]``.
            x:     Reference waveform with the same shape.

        Returns:
            Dictionary with keys ``sc`` (spectral convergence), ``mag``
            (log-magnitude L1) and ``total`` (their sum).
        """
        sc_loss = x.new_zeros(())
        mag_loss = x.new_zeros(())
        for n_fft, hop, win in zip(self.fft_sizes, self.hop_sizes, self.win_sizes):
            s = _stft(x, n_fft, hop, win)
            s_hat = _stft(x_hat, n_fft, hop, win)
            denom = torch.linalg.norm(s, ord="fro") + self.eps
            sc = torch.linalg.norm(s - s_hat, ord="fro") / denom
            mag = F.l1_loss(torch.log(s_hat + self.eps), torch.log(s + self.eps))
            sc_loss = sc_loss + sc
            mag_loss = mag_loss + mag
        n = float(len(self.fft_sizes))
        sc_loss = sc_loss / n
        mag_loss = mag_loss / n
        return {"sc": sc_loss, "mag": mag_loss, "total": sc_loss + mag_loss}


class MelReconstructionLoss(nn.Module):
    """L1 loss on a log-mel spectrogram (80-band by default).

    Args:
        sample_rate: Audio sample rate (Hz).
        n_fft: FFT size.
        hop_length: Hop length.
        win_length: Window length (defaults to ``n_fft``).
        n_mels: Number of mel bands (default 80).
        f_min: Minimum mel frequency.
        f_max: Maximum mel frequency (defaults to ``sample_rate / 2``).
        eps: Numerical floor for the logarithm.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: Optional[int] = None,
        n_mels: int = 80,
        f_min: float = 0.0,
        f_max: Optional[float] = None,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        # Lazy import to avoid hard dependency at module import time.
        import torchaudio

        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length or n_fft,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max or sample_rate / 2,
            power=1.0,
        )
        self.eps = eps

    def forward(self, x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Return the L1 loss between log-mel spectrograms of ``x_hat`` and ``x``."""
        if x_hat.dim() == 3:
            x_hat = x_hat.squeeze(1)
        if x.dim() == 3:
            x = x.squeeze(1)
        m_hat = torch.log(self.mel(x_hat) + self.eps)
        m = torch.log(self.mel(x) + self.eps)
        return F.l1_loss(m_hat, m)


class TimeDomainLoss(nn.Module):
    """Time-domain L1 + L2 reconstruction loss.

    Args:
        l1_weight: Weight on the L1 component.
        l2_weight: Weight on the L2 component.
    """

    def __init__(self, l1_weight: float = 1.0, l2_weight: float = 1.0) -> None:
        super().__init__()
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight

    def forward(self, x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Combine weighted L1 and L2 losses on the raw waveform."""
        l1 = F.l1_loss(x_hat, x)
        l2 = F.mse_loss(x_hat, x)
        return self.l1_weight * l1 + self.l2_weight * l2


# ---------------------------------------------------------------------------
# Quantization losses
# ---------------------------------------------------------------------------
class CommitmentLoss(nn.Module):
    """VQ commitment loss (passthrough wrapper).

    The actual commitment loss tensor is produced by the SPQ quantizer; this
    module simply scales it so that it can be plugged into the same loss
    pipeline as the others.
    """

    def __init__(self, weight: float = 0.25) -> None:
        super().__init__()
        self.weight = weight

    def forward(self, commitment: torch.Tensor) -> torch.Tensor:
        return self.weight * commitment


class CodebookLoss(nn.Module):
    """Codebook diversity / utilisation regulariser.

    Encourages a uniform distribution over codebook indices to avoid
    collapse. Computed as the negative entropy of the soft code-usage
    distribution, summed over each codebook.

    Args:
        codebook_size: Number of entries per codebook.
        eps: Numerical floor inside the log.
    """

    def __init__(self, codebook_size: int = 1024, eps: float = 1e-8) -> None:
        super().__init__()
        self.codebook_size = codebook_size
        self.eps = eps

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        """Compute the diversity loss.

        Args:
            codes: Long tensor of code indices ``[B, num_codebooks, T]``.

        Returns:
            Scalar diversity loss (lower → more uniform usage).
        """
        if codes.dim() == 2:
            codes = codes.unsqueeze(1)
        b, nq, t = codes.shape
        loss = codes.new_zeros((), dtype=torch.float32)
        max_ent = torch.log(torch.tensor(float(self.codebook_size)))
        for q in range(nq):
            flat = codes[:, q].reshape(-1)
            counts = torch.bincount(flat, minlength=self.codebook_size).float()
            probs = counts / (counts.sum() + self.eps)
            ent = -(probs * torch.log(probs + self.eps)).sum()
            # Loss is the gap between achieved and uniform entropy.
            loss = loss + (max_ent - ent)
        return loss / max(1, nq)


# ---------------------------------------------------------------------------
# Adversarial losses
# ---------------------------------------------------------------------------
class AdversarialLoss(nn.Module):
    """GAN adversarial loss with hinge or least-squares variants.

    Use :meth:`generator_loss` to compute the generator-side loss from the
    discriminator's logits on fake samples, and :meth:`discriminator_loss`
    for the discriminator update.

    Args:
        loss_type: Either ``"hinge"`` or ``"lsgan"``.
    """

    def __init__(self, loss_type: str = "hinge") -> None:
        super().__init__()
        if loss_type not in {"hinge", "lsgan"}:
            raise ValueError(f"Unknown adversarial loss type: {loss_type}")
        self.loss_type = loss_type

    def generator_loss(self, fake_logits: List[torch.Tensor]) -> torch.Tensor:
        """Generator update: pull fake logits towards the real distribution."""
        loss = fake_logits[0].new_zeros(())
        for logit in fake_logits:
            if self.loss_type == "hinge":
                loss = loss + (-logit).mean()
            else:  # lsgan
                loss = loss + F.mse_loss(logit, torch.ones_like(logit))
        return loss / max(1, len(fake_logits))

    def discriminator_loss(
        self,
        real_logits: List[torch.Tensor],
        fake_logits: List[torch.Tensor],
    ) -> torch.Tensor:
        """Discriminator update: classify real vs. fake."""
        loss = real_logits[0].new_zeros(())
        for r, f in zip(real_logits, fake_logits):
            if self.loss_type == "hinge":
                loss = loss + F.relu(1.0 - r).mean() + F.relu(1.0 + f).mean()
            else:  # lsgan
                loss = loss + F.mse_loss(r, torch.ones_like(r)) + F.mse_loss(
                    f, torch.zeros_like(f)
                )
        return loss / max(1, len(real_logits))

    def forward(self, fake_logits: List[torch.Tensor]) -> torch.Tensor:
        """Default ``forward`` returns the generator-side loss."""
        return self.generator_loss(fake_logits)


class FeatureMatchingLoss(nn.Module):
    """Feature-matching loss: L1 between intermediate discriminator features."""

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        fake_features: List[List[torch.Tensor]],
        real_features: List[List[torch.Tensor]],
    ) -> torch.Tensor:
        """Compute feature-matching distance.

        Args:
            fake_features: For each sub-discriminator, a list of intermediate
                feature tensors produced from the generator output.
            real_features: Same shape as ``fake_features`` but from the real
                signal.

        Returns:
            Scalar L1 distance averaged across sub-discriminators and layers.
        """
        if len(fake_features) == 0:
            return torch.zeros((), device=next(iter(())) if False else "cpu")
        device = fake_features[0][0].device
        loss = torch.zeros((), device=device)
        n_terms = 0
        for fake_d, real_d in zip(fake_features, real_features):
            for f, r in zip(fake_d, real_d):
                loss = loss + F.l1_loss(f, r.detach())
                n_terms += 1
        return loss / max(1, n_terms)


# ---------------------------------------------------------------------------
# AFR regularisation
# ---------------------------------------------------------------------------
class AFRRegularizationLoss(nn.Module):
    """Adaptive frame-rate regulariser.

    Combines a quadratic rate-tracking term that drags ``keep_ratio`` towards
    a target value, with a per-frame entropy penalty that prevents the gate
    distribution from collapsing to {0, 1}.

    Args:
        target_keep: Desired fraction of frames retained.
        rate_weight: Weight on the rate-tracking term.
        entropy_weight: Weight on the entropy regulariser (encourages a
            healthy soft-gate distribution).
        eps: Numerical floor inside the log.
    """

    def __init__(
        self,
        target_keep: float = 0.0625,
        rate_weight: float = 1.0,
        entropy_weight: float = 0.1,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.target_keep = target_keep
        self.rate_weight = rate_weight
        self.entropy_weight = entropy_weight
        self.eps = eps

    def forward(
        self,
        gate: torch.Tensor,
        keep_ratio: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute the AFR regularisation losses.

        Args:
            gate: Soft gate values ``[B, T']`` in ``(0, 1)``.
            keep_ratio: Optional pre-computed mean keep ratio. If ``None``,
                computed as ``gate.mean()``.

        Returns:
            Dictionary with ``rate``, ``entropy`` and ``total`` keys.
        """
        if keep_ratio is None:
            keep_ratio = gate.mean()
        rate = (keep_ratio - self.target_keep).pow(2)
        # Bernoulli-style entropy of the soft gate; max at gate==0.5.
        p = gate.clamp(self.eps, 1.0 - self.eps)
        bern_ent = -(p * torch.log(p) + (1 - p) * torch.log(1 - p)).mean()
        # We *subtract* entropy because we want to penalise low entropy.
        entropy_term = -bern_ent
        total = self.rate_weight * rate + self.entropy_weight * entropy_term
        return {"rate": rate, "entropy": entropy_term, "total": total}


# ---------------------------------------------------------------------------
# Combined loss
# ---------------------------------------------------------------------------
def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    """Safely fetch ``key`` from a dict-like or OmegaConf object."""
    if cfg is None:
        return default
    if hasattr(cfg, "get"):
        try:
            val = cfg.get(key, default)
        except Exception:
            val = default
        return val
    return getattr(cfg, key, default)


class UltraCodecLoss(nn.Module):
    """Stage-aware combined loss for UltraCodec training.

    Reads loss weights from ``config.training.losses`` and stage information
    from ``config.training.stage`` (``1``, ``2`` or ``3``). The forward pass
    consumes the dictionary returned by :class:`UltraCodec.forward` plus the
    raw target waveform and (optionally) discriminator outputs.

    Args:
        config: OmegaConf-style mapping with ``model`` and ``training``
            sub-sections.
    """

    def __init__(self, config: Any) -> None:
        super().__init__()
        self.config = config
        training_cfg = _cfg_get(config, "training", {})
        model_cfg = _cfg_get(config, "model", {})
        loss_cfg = _cfg_get(training_cfg, "losses", {})

        self.stage = int(_cfg_get(training_cfg, "stage", 1))
        # Loss weights (paper Section 3.7).
        self.w_recon = float(_cfg_get(loss_cfg, "reconstruction_weight", 1.0))
        self.w_adv = float(_cfg_get(loss_cfg, "adversarial_weight", 0.0))
        self.w_fm = float(_cfg_get(loss_cfg, "feature_matching_weight", 2.0))
        self.w_codebook = float(_cfg_get(loss_cfg, "codebook_weight", 0.25))
        self.w_pred = float(_cfg_get(loss_cfg, "prediction_weight", 0.5))
        self.w_semantic = float(_cfg_get(loss_cfg, "semantic_weight", 0.0))
        self.w_rate = float(_cfg_get(loss_cfg, "rate_weight", 10.0))
        self.w_llm = float(_cfg_get(loss_cfg, "llm_weight", 0.0))
        self.w_diversity = float(_cfg_get(loss_cfg, "diversity_weight", 0.0))

        # AFR regularisation flag (disabled in stage 1 by config).
        afr_train_cfg = _cfg_get(training_cfg, "afr", {})
        self.afr_enabled = bool(_cfg_get(afr_train_cfg, "enabled", self.stage >= 2))

        # Sub-modules.
        sample_rate = int(_cfg_get(model_cfg, "sample_rate", 16000))
        fft_sizes = list(_cfg_get(loss_cfg, "stft_fft_sizes", [512, 1024, 2048]))
        self.stft_loss = MultiResolutionSTFTLoss(fft_sizes=fft_sizes)
        self.mel_loss = MelReconstructionLoss(
            sample_rate=sample_rate,
            n_fft=1024,
            hop_length=256,
            n_mels=int(_cfg_get(loss_cfg, "n_mels", 80)),
        )
        self.time_loss = TimeDomainLoss()

        codebook_size = int(_cfg_get(_cfg_get(model_cfg, "quantizer", {}),
                                      "codebook_size", 1024))
        self.codebook_div = CodebookLoss(codebook_size=codebook_size)

        adv_type = str(_cfg_get(loss_cfg, "adversarial_type", "hinge"))
        self.adv_loss = AdversarialLoss(loss_type=adv_type)
        self.fm_loss = FeatureMatchingLoss()

        afr_model_cfg = _cfg_get(model_cfg, "afr", {})
        self.afr_loss = AFRRegularizationLoss(
            target_keep=float(_cfg_get(afr_model_cfg, "target_keep", 0.0625)),
            rate_weight=1.0,
            entropy_weight=float(_cfg_get(afr_model_cfg, "entropy_weight", 0.1)),
        )

    # ------------------------------------------------------------------
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        discriminator_outputs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute the total generator loss.

        Args:
            outputs: Output dict from :class:`UltraCodec.forward`. Required
                keys: ``x_hat``, ``commitment_loss``, ``codebook_loss``,
                ``rate_loss``. Optional: ``codes``, ``gate_decisions``,
                ``keep_ratio``.
            targets: Ground-truth waveform ``[B, 1, T]``.
            discriminator_outputs: Optional dict with keys ``fake_logits``,
                ``real_logits``, ``fake_features``, ``real_features``. Only
                consumed when ``adversarial_weight > 0``.

        Returns:
            ``(total_loss, partial_losses)`` where ``partial_losses`` is a
            mapping from loss name to scalar tensor (useful for logging).
        """
        x_hat = outputs["x_hat"]
        if x_hat.dim() == 2:
            x_hat = x_hat.unsqueeze(1)
        if targets.dim() == 2:
            targets = targets.unsqueeze(1)

        partial: Dict[str, torch.Tensor] = {}

        # ---- Reconstruction ------------------------------------------------
        stft_terms = self.stft_loss(x_hat, targets)
        mel = self.mel_loss(x_hat, targets)
        time = self.time_loss(x_hat, targets)
        recon = stft_terms["total"] + mel + time
        partial["loss/stft_sc"] = stft_terms["sc"].detach()
        partial["loss/stft_mag"] = stft_terms["mag"].detach()
        partial["loss/mel"] = mel.detach()
        partial["loss/time"] = time.detach()
        partial["loss/recon"] = recon.detach()

        total = self.w_recon * recon

        # ---- Quantization --------------------------------------------------
        commit = outputs.get("commitment_loss")
        if commit is not None:
            total = total + self.w_codebook * commit
            partial["loss/commitment"] = commit.detach()
        prediction = outputs.get("codebook_loss")
        if prediction is not None:
            total = total + self.w_pred * prediction
            partial["loss/prediction"] = prediction.detach()

        if self.w_diversity > 0 and "codes" in outputs and outputs["codes"] is not None:
            try:
                div = self.codebook_div(outputs["codes"])
                total = total + self.w_diversity * div
                partial["loss/codebook_div"] = div.detach()
            except Exception:
                pass

        # ---- AFR regularisation -------------------------------------------
        if self.afr_enabled:
            gate = outputs.get("gate_decisions")
            keep = outputs.get("keep_ratio")
            if gate is not None:
                afr_terms = self.afr_loss(gate, keep_ratio=keep)
                total = total + self.w_rate * afr_terms["total"]
                partial["loss/afr_rate"] = afr_terms["rate"].detach()
                partial["loss/afr_entropy"] = afr_terms["entropy"].detach()
            elif "rate_loss" in outputs and outputs["rate_loss"] is not None:
                rl = outputs["rate_loss"]
                total = total + self.w_rate * rl
                partial["loss/afr_rate"] = rl.detach()

        # ---- Adversarial + feature matching -------------------------------
        if self.w_adv > 0 and discriminator_outputs is not None:
            fake_logits = discriminator_outputs.get("fake_logits", [])
            if fake_logits:
                g_adv = self.adv_loss.generator_loss(fake_logits)
                total = total + self.w_adv * g_adv
                partial["loss/adv_g"] = g_adv.detach()
            real_feats = discriminator_outputs.get("real_features")
            fake_feats = discriminator_outputs.get("fake_features")
            if real_feats and fake_feats:
                fm = self.fm_loss(fake_feats, real_feats)
                total = total + self.w_fm * fm
                partial["loss/feature_match"] = fm.detach()

        partial["loss/total"] = total.detach()
        if "keep_ratio" in outputs and outputs["keep_ratio"] is not None:
            try:
                partial["stat/keep_ratio"] = outputs["keep_ratio"].detach().mean()
            except Exception:
                pass
        return total, partial


__all__ = [
    "MultiResolutionSTFTLoss",
    "MelReconstructionLoss",
    "TimeDomainLoss",
    "CommitmentLoss",
    "CodebookLoss",
    "AdversarialLoss",
    "FeatureMatchingLoss",
    "AFRRegularizationLoss",
    "UltraCodecLoss",
]
