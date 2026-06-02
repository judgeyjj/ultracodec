#!/usr/bin/env python3
"""UltraCodec training script.

Implements the full multi-stage training pipeline described in
``papers/paper3_ultracodec.md`` Section 5:

* **Stage 1** (config: ``configs/train_stage1.yaml``)
  Pure reconstruction. No adversarial loss, AFR disabled.

* **Stage 2** (``configs/train_stage2.yaml``)
  Adds discriminator + adversarial / feature-matching losses, full SPQ
  optimisation. AFR enabled.

* **Stage 3** (``configs/train_stage3.yaml``)
  Joint fine-tuning with AFR + LLM-aware losses (LoRA on codec/LLM).

Features
--------
* PyTorch native ``DistributedDataParallel`` (``torchrun``)
* Automatic mixed precision (``bf16``/``fp16`` per config)
* Gradient accumulation
* Generator + discriminator optimisers with alternating GAN updates
* EMA over generator weights
* Periodic validation
* Checkpoint save / resume (model, optimiser, scheduler, EMA, scaler)
* Wandb + TensorBoard logging (silently skipped if libs missing)
* CLI overrides for output dir, max_steps, batch_size

Usage
-----
::

    # Single GPU
    python scripts/train.py --config configs/train_stage1.yaml

    # Multi-GPU (4 ranks on one node)
    torchrun --nproc_per_node=4 scripts/train.py --config configs/train_stage2.yaml

    # Resume from checkpoint
    python scripts/train.py --config configs/train_stage2.yaml \\
        --resume runs/uc_stage1/best.pt
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf, open_dict
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

# Make the package importable when running the script directly.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultracodec.data import LibriSpeechDataset, MixedDataset, VCTKDataset  # noqa: E402
from ultracodec.losses import CombinedDiscriminator, UltraCodecLoss  # noqa: E402
from ultracodec.losses.losses import AdversarialLoss  # noqa: E402
from ultracodec.utils import (  # noqa: E402
    EMA,
    init_logger,
    load_checkpoint,
    save_checkpoint,
    set_seed,
)
from ultracodec.utils.utils import (  # noqa: E402
    AverageMeter,
    build_scheduler,
    count_parameters,
    keep_last_checkpoints,
    to_device,
)

logger = logging.getLogger("ultracodec.train")


# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------
def is_dist_run() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def get_rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return int(os.environ.get("RANK", "0"))


def get_world_size() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return int(os.environ.get("WORLD_SIZE", "1"))


def is_main_process() -> bool:
    return get_rank() == 0


def setup_distributed(backend: str = "nccl", cfg=None) -> Tuple[int, int, torch.device]:
    """Initialise ``torch.distributed`` if running under ``torchrun``.

    Returns ``(rank, world_size, device)`` regardless of distributed mode.
    If cfg contains hardware.gpu_id, sets CUDA_VISIBLE_DEVICES accordingly
    (only in non-distributed single-GPU mode).
    """
    if is_dist_run():
        if not dist.is_initialized():
            dist.init_process_group(backend=backend)
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
        else:
            device = torch.device("cpu")
        return rank, world_size, device
    # Single-GPU mode: respect hardware.gpu_id from config
    if cfg is not None and not os.environ.get("CUDA_VISIBLE_DEVICES"):
        gpu_id = cfg.get("hardware", {}).get("gpu_id", None)
        if gpu_id is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            logging.info(f"Set CUDA_VISIBLE_DEVICES={gpu_id} from config")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return 0, 1, device


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
def load_config(path: str) -> DictConfig:
    """Load a YAML config and merge it with its ``defaults: [base]`` parents."""
    cfg = OmegaConf.load(path)
    parents = []
    defaults = cfg.get("defaults", None)
    if defaults is not None:
        cfg_dir = Path(path).parent
        # OmegaConf ListConfig may not pass isinstance(..., list)
        defaults_list = OmegaConf.to_container(defaults) if hasattr(defaults, '__iter__') and not isinstance(defaults, str) else []
        # Remove defaults key from cfg before merge
        with open_dict(cfg):
            cfg.pop("defaults", None)
        for entry in defaults_list:
            parent_path = cfg_dir / f"{entry}.yaml"
            if parent_path.exists():
                parents.append(OmegaConf.load(parent_path))
            else:
                logging.warning(f"Default config not found: {parent_path}")
    merged = OmegaConf.merge(*parents, cfg) if parents else cfg
    OmegaConf.resolve(merged)
    return merged  # type: ignore[return-value]


def apply_cli_overrides(cfg: DictConfig, args: argparse.Namespace) -> DictConfig:
    if args.output_dir is not None:
        cfg.training.checkpoint.output_dir = args.output_dir
    if args.max_steps is not None:
        cfg.training.max_steps = int(args.max_steps)
    if args.batch_size is not None:
        cfg.training.batch_size = int(args.batch_size)
    if args.seed is not None:
        cfg.seed = int(args.seed)
    return cfg


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def build_train_dataset(cfg: DictConfig):
    parts = []
    data_cfg = cfg.data
    aug_enabled = bool(getattr(data_cfg.get("augmentation", {}), "enabled", False))

    vctk_root = Path(data_cfg.vctk.root)
    if vctk_root.exists():
        parts.append(
            VCTKDataset(
                root_dir=vctk_root,
                split="train",
                sample_rate=data_cfg.sample_rate,
                segment_length=data_cfg.segment_length,
                augment=aug_enabled,
            )
        )
    else:
        logger.warning("VCTK root %s missing; skipping VCTK in training set.", vctk_root)

    ls_cfg = data_cfg.librispeech
    for split in ls_cfg.train_splits:
        parts.append(
            LibriSpeechDataset(
                root_dir=ls_cfg.root,
                split=split,
                download=bool(ls_cfg.get("download", True)),
                sample_rate=data_cfg.sample_rate,
                segment_length=data_cfg.segment_length,
                augment=aug_enabled,
            )
        )

    if not parts:
        raise RuntimeError(
            "No training datasets available. Verify `data.vctk.root` and "
            "`data.librispeech.root` in your config."
        )
    return MixedDataset(parts)


def build_val_dataset(cfg: DictConfig):
    data_cfg = cfg.data
    ls_cfg = data_cfg.librispeech
    splits = ls_cfg.val_splits
    if not splits:
        return None
    return LibriSpeechDataset(
        root_dir=ls_cfg.root,
        split=splits[0],
        download=bool(ls_cfg.get("download", True)),
        sample_rate=data_cfg.sample_rate,
        segment_length=data_cfg.segment_length,
        augment=False,
    )


def build_dataloader(
    dataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    distributed: bool,
    drop_last: bool = True,
):
    sampler = None
    if distributed and dataset is not None:
        sampler = DistributedSampler(
            dataset,
            num_replicas=get_world_size(),
            rank=get_rank(),
            shuffle=shuffle,
            drop_last=drop_last,
        )
        shuffle = False
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
        persistent_workers=num_workers > 0,
    )


# ---------------------------------------------------------------------------
# Model / optimiser construction
# ---------------------------------------------------------------------------
def build_generator(cfg: DictConfig) -> nn.Module:
    """Construct the UltraCodec generator from config."""
    from ultracodec.model import UltraCodec

    return UltraCodec(cfg.model)


def build_discriminator(cfg: DictConfig) -> Optional[nn.Module]:
    """Build the combined discriminator (None if adversarial weight is 0)."""
    loss_cfg = cfg.training.get("losses", {})
    adv_w = float(loss_cfg.get("adversarial_weight", 0.0))
    if adv_w <= 0.0:
        return None
    disc_cfg = cfg.training.get("discriminator", {}) or {}
    periods = list(disc_cfg.get("periods", [2, 3, 5, 7, 11]))
    scales = int(disc_cfg.get("scales", 3))
    return CombinedDiscriminator(periods=periods, scales=scales)


def build_optimizers(
    generator: nn.Module,
    discriminator: Optional[nn.Module],
    cfg: DictConfig,
) -> Tuple[torch.optim.Optimizer, Optional[torch.optim.Optimizer]]:
    train_cfg = cfg.training
    g_opt = torch.optim.AdamW(
        [p for p in generator.parameters() if p.requires_grad],
        lr=float(train_cfg.learning_rate),
        weight_decay=float(train_cfg.weight_decay),
        betas=(0.9, 0.98),
    )
    d_opt: Optional[torch.optim.Optimizer] = None
    if discriminator is not None:
        d_lr = float(train_cfg.get("discriminator_lr", train_cfg.learning_rate))
        d_opt = torch.optim.AdamW(
            [p for p in discriminator.parameters() if p.requires_grad],
            lr=d_lr,
            weight_decay=float(train_cfg.weight_decay),
            betas=(0.5, 0.9),
        )
    return g_opt, d_opt


# ---------------------------------------------------------------------------
# Precision helpers
# ---------------------------------------------------------------------------
def resolve_amp(precision: str, device: torch.device) -> Tuple[bool, torch.dtype]:
    """Return ``(use_amp, dtype)`` from a config string."""
    p = (precision or "").lower()
    if device.type != "cuda" or p in {"fp32", "32", ""}:
        return False, torch.float32
    if p in {"bf16", "bfloat16"}:
        return True, torch.bfloat16
    if p in {"fp16", "float16", "16"}:
        return True, torch.float16
    return False, torch.float32


# ---------------------------------------------------------------------------
# Logging utilities
# ---------------------------------------------------------------------------
class TrainLogger:
    """Thin wrapper around wandb + TensorBoard. Silent if libs unavailable."""

    def __init__(self, cfg: DictConfig, output_dir: Path) -> None:
        self.cfg = cfg
        self.output_dir = output_dir
        self.use_wandb = False
        self.tb = None
        if not is_main_process():
            return
        log_cfg = cfg.training.get("logging", {})
        if bool(log_cfg.get("use_wandb", False)):
            try:
                import wandb

                wandb.init(
                    project=log_cfg.get("project", "ultracodec"),
                    name=log_cfg.get("run_name", None),
                    config=OmegaConf.to_container(cfg, resolve=True),
                    dir=str(output_dir),
                )
                self.use_wandb = True
                self._wandb = wandb
            except Exception as e:  # pragma: no cover
                logger.warning("wandb unavailable: %s", e)
        try:
            from torch.utils.tensorboard import SummaryWriter

            self.tb = SummaryWriter(log_dir=str(output_dir / "tb"))
        except Exception as e:  # pragma: no cover
            logger.warning("TensorBoard unavailable: %s", e)

    def log(self, metrics: Dict[str, float], step: int) -> None:
        if not is_main_process():
            return
        if self.use_wandb:
            try:
                self._wandb.log(metrics, step=step)
            except Exception:
                pass
        if self.tb is not None:
            for k, v in metrics.items():
                try:
                    self.tb.add_scalar(k, float(v), step)
                except Exception:
                    pass

    def close(self) -> None:
        if not is_main_process():
            return
        if self.use_wandb:
            try:
                self._wandb.finish()
            except Exception:
                pass
        if self.tb is not None:
            try:
                self.tb.close()
            except Exception:
                pass


def reduce_scalar(value: torch.Tensor) -> torch.Tensor:
    """All-reduce a scalar tensor (mean) across DDP ranks."""
    if not (dist.is_available() and dist.is_initialized()):
        return value
    v = value.detach().clone()
    dist.all_reduce(v, op=dist.ReduceOp.SUM)
    v = v / dist.get_world_size()
    return v


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def run_validation(
    generator: nn.Module,
    val_loader: Optional[DataLoader],
    loss_fn: UltraCodecLoss,
    device: torch.device,
    amp_dtype: torch.dtype,
    use_amp: bool,
    max_batches: int = 50,
) -> Dict[str, float]:
    if val_loader is None:
        return {}
    generator.eval()
    meter_total = AverageMeter("val/total")
    meter_recon = AverageMeter("val/recon")
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= max_batches:
                break
            batch = to_device(batch, device)
            wav = batch["wav"]
            if wav.dim() == 2:
                wav = wav.unsqueeze(1)
            ctx = (
                torch.autocast(device_type=device.type, dtype=amp_dtype)
                if use_amp
                else nullcontext()
            )
            with ctx:
                outputs = generator(wav)
                loss, partial = loss_fn(outputs, wav, discriminator_outputs=None)
            meter_total.update(float(loss.detach()))
            recon = partial.get("loss/recon")
            if recon is not None:
                meter_recon.update(float(recon))
    generator.train()
    return {"val/total": meter_total.avg, "val/recon": meter_recon.avg}


def train(cfg: DictConfig, resume: Optional[str] = None) -> None:
    """Main training entry."""
    rank, world_size, device = setup_distributed(cfg=cfg)
    set_seed(int(cfg.get("seed", 42)) + rank)

    if is_main_process():
        logger.info("World size: %d, device: %s", world_size, device)
        logger.info("Stage: %s", cfg.training.get("stage", "?"))

    # Build datasets / loaders.
    train_ds = build_train_dataset(cfg)
    val_ds = build_val_dataset(cfg) if is_main_process() else None
    if is_main_process():
        logger.info("Train dataset size: %d", len(train_ds))
        if val_ds is not None:
            logger.info("Val dataset size: %d", len(val_ds))

    train_loader = build_dataloader(
        train_ds,
        batch_size=int(cfg.training.batch_size),
        shuffle=True,
        num_workers=int(cfg.data.num_workers),
        distributed=world_size > 1,
        drop_last=True,
    )
    val_loader = (
        build_dataloader(
            val_ds,
            batch_size=int(cfg.training.batch_size),
            shuffle=False,
            num_workers=max(2, int(cfg.data.num_workers) // 2),
            distributed=False,
            drop_last=False,
        )
        if val_ds is not None
        else None
    )

    # Build models.
    generator = build_generator(cfg).to(device)
    discriminator = build_discriminator(cfg)
    if discriminator is not None:
        discriminator = discriminator.to(device)

    if is_main_process():
        logger.info(
            "Generator params: %.2f M", count_parameters(generator) / 1e6
        )
        if discriminator is not None:
            logger.info(
                "Discriminator params: %.2f M",
                count_parameters(discriminator) / 1e6,
            )

    # Optimisers + schedulers.
    g_opt, d_opt = build_optimizers(generator, discriminator, cfg)
    g_sched = build_scheduler(
        g_opt,
        str(cfg.training.scheduler),
        warmup_steps=int(cfg.training.warmup_steps),
        total_steps=int(cfg.training.max_steps),
    )
    d_sched = None
    if d_opt is not None:
        d_sched = build_scheduler(
            d_opt,
            str(cfg.training.scheduler),
            warmup_steps=int(cfg.training.warmup_steps),
            total_steps=int(cfg.training.max_steps),
        )

    # AMP setup.
    use_amp, amp_dtype = resolve_amp(str(cfg.training.get("precision", "fp32")), device)
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and amp_dtype == torch.float16))
    d_scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and amp_dtype == torch.float16))

    # Loss.
    loss_fn = UltraCodecLoss(cfg).to(device)
    adv_loss_module = AdversarialLoss(
        loss_type=str(cfg.training.get("losses", {}).get("adversarial_type", "hinge"))
    )

    # Resume.
    start_step = 0
    if resume:
        payload = load_checkpoint(
            resume, generator, g_opt, g_sched, map_location=device
        )
        start_step = int(payload.get("step", 0))
        if discriminator is not None and "discriminator" in payload:
            discriminator.load_state_dict(payload["discriminator"])
            if d_opt is not None and "d_optimizer" in payload:
                d_opt.load_state_dict(payload["d_optimizer"])
            if d_sched is not None and "d_scheduler" in payload:
                d_sched.load_state_dict(payload["d_scheduler"])
        if "scaler" in payload and scaler is not None:
            try:
                scaler.load_state_dict(payload["scaler"])
            except Exception:
                pass
        if is_main_process():
            logger.info("Resumed from %s at step %d.", resume, start_step)

    # EMA over the live (un-wrapped) generator.
    ema = EMA(generator, decay=float(cfg.training.get("ema_decay", 0.999)))

    # Wrap with DDP after EMA snapshot.
    if world_size > 1:
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        ddp_kwargs = dict(
            device_ids=[local_rank] if device.type == "cuda" else None,
            output_device=local_rank if device.type == "cuda" else None,
            find_unused_parameters=True,
        )
        generator = DDP(generator, **ddp_kwargs)
        if discriminator is not None:
            discriminator = DDP(discriminator, **ddp_kwargs)

    def _gen_module() -> nn.Module:
        return generator.module if isinstance(generator, DDP) else generator

    def _disc_module() -> Optional[nn.Module]:
        if discriminator is None:
            return None
        return discriminator.module if isinstance(discriminator, DDP) else discriminator

    output_dir = Path(cfg.training.checkpoint.output_dir)
    if is_main_process():
        output_dir.mkdir(parents=True, exist_ok=True)

    train_logger = TrainLogger(cfg, output_dir)

    max_steps = int(cfg.training.max_steps)
    save_every = int(cfg.training.checkpoint.save_every)
    keep_last = int(cfg.training.checkpoint.get("keep_last", 5))
    log_every = int(cfg.training.logging.log_every)
    val_every = int(cfg.training.get("val_every", max(save_every, 1)))
    accum = max(1, int(cfg.training.get("accumulate_grad", 1)))
    grad_clip = float(cfg.training.get("gradient_clip", 1.0))
    adv_weight = float(cfg.training.get("losses", {}).get("adversarial_weight", 0.0))

    best_val = float("inf")
    step = start_step
    epoch = 0
    g_opt.zero_grad(set_to_none=True)
    if d_opt is not None:
        d_opt.zero_grad(set_to_none=True)

    if is_main_process():
        logger.info("=== Starting training ===")

    generator.train()
    if discriminator is not None:
        discriminator.train()

    while step < max_steps:
        epoch += 1
        sampler = getattr(train_loader, "sampler", None)
        if isinstance(sampler, DistributedSampler):
            sampler.set_epoch(epoch)
        epoch_t0 = time.time()
        for accum_idx, batch in enumerate(train_loader):
            if step >= max_steps:
                break
            batch = to_device(batch, device)
            wav = batch["wav"]
            if wav.dim() == 2:
                wav = wav.unsqueeze(1)

            # ----- Generator forward (kept for both D and G updates) -----
            ctx = (
                torch.autocast(device_type=device.type, dtype=amp_dtype)
                if use_amp
                else nullcontext()
            )
            with ctx:
                outputs = generator(wav)
                x_hat = outputs["x_hat"]

            # ============ Discriminator update (every micro-step) ============
            d_loss_value = 0.0
            if discriminator is not None and adv_weight > 0:
                with ctx:
                    real_out = discriminator(wav)
                    fake_out = discriminator(x_hat.detach())
                    d_loss = adv_loss_module.discriminator_loss(
                        real_out["logits"], fake_out["logits"]
                    )
                if amp_dtype == torch.float16:
                    d_scaler.scale(d_loss / accum).backward()
                else:
                    (d_loss / accum).backward()
                d_loss_value = float(d_loss.detach())

            # ============ Generator update ============
            disc_outputs: Optional[Dict[str, Any]] = None
            if discriminator is not None and adv_weight > 0:
                with ctx:
                    fake_out_g = discriminator(x_hat)
                    real_out_g = discriminator(wav)
                disc_outputs = {
                    "fake_logits": fake_out_g["logits"],
                    "real_logits": real_out_g["logits"],
                    "fake_features": fake_out_g["features"],
                    "real_features": real_out_g["features"],
                }
            with ctx:
                g_loss, partial = loss_fn(outputs, wav, discriminator_outputs=disc_outputs)
            if amp_dtype == torch.float16:
                scaler.scale(g_loss / accum).backward()
            else:
                (g_loss / accum).backward()

            do_step = ((accum_idx + 1) % accum == 0)
            if do_step:
                # Unscale for clipping if needed.
                if amp_dtype == torch.float16:
                    scaler.unscale_(g_opt)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in generator.parameters() if p.requires_grad],
                    grad_clip,
                )
                if amp_dtype == torch.float16:
                    scaler.step(g_opt)
                    scaler.update()
                else:
                    g_opt.step()
                g_opt.zero_grad(set_to_none=True)
                g_sched.step()

                if d_opt is not None:
                    if amp_dtype == torch.float16:
                        d_scaler.unscale_(d_opt)
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in discriminator.parameters() if p.requires_grad],
                        grad_clip,
                    )
                    if amp_dtype == torch.float16:
                        d_scaler.step(d_opt)
                        d_scaler.update()
                    else:
                        d_opt.step()
                    d_opt.zero_grad(set_to_none=True)
                    if d_sched is not None:
                        d_sched.step()

                # Update EMA on the un-wrapped generator.
                ema.update(_gen_module())

                step += 1

                # ----- Logging -----
                if is_main_process() and step % log_every == 0:
                    metrics = {
                        "train/total": float(g_loss.detach()),
                        "train/d_loss": d_loss_value,
                        "train/lr_g": g_opt.param_groups[0]["lr"],
                    }
                    if d_opt is not None:
                        metrics["train/lr_d"] = d_opt.param_groups[0]["lr"]
                    for k, v in partial.items():
                        try:
                            metrics[f"train/{k.split('/', 1)[-1]}"] = float(v)
                        except Exception:
                            pass
                    train_logger.log(metrics, step)
                    logger.info(
                        "epoch=%d step=%d/%d g=%.4f d=%.4f lr=%.2e",
                        epoch,
                        step,
                        max_steps,
                        metrics["train/total"],
                        d_loss_value,
                        metrics["train/lr_g"],
                    )

                # ----- Validation -----
                if val_every > 0 and step > 0 and step % val_every == 0 and is_main_process():
                    ema.apply_shadow(_gen_module())
                    val_metrics = run_validation(
                        _gen_module(),
                        val_loader,
                        loss_fn,
                        device,
                        amp_dtype,
                        use_amp,
                    )
                    ema.restore(_gen_module())
                    if val_metrics:
                        train_logger.log(val_metrics, step)
                        logger.info(
                            "VAL step=%d total=%.4f recon=%.4f",
                            step,
                            val_metrics.get("val/total", float("nan")),
                            val_metrics.get("val/recon", float("nan")),
                        )
                        cur = val_metrics.get("val/total", float("inf"))
                        if cur < best_val:
                            best_val = cur
                            best_path = output_dir / "best.pt"
                            _save(
                                best_path,
                                _gen_module(),
                                g_opt,
                                g_sched,
                                _disc_module(),
                                d_opt,
                                d_sched,
                                ema,
                                scaler,
                                step,
                                cfg,
                            )

                # ----- Checkpointing -----
                if (
                    save_every > 0
                    and step > 0
                    and step % save_every == 0
                    and is_main_process()
                ):
                    ckpt_path = output_dir / f"step_{step:08d}.pt"
                    _save(
                        ckpt_path,
                        _gen_module(),
                        g_opt,
                        g_sched,
                        _disc_module(),
                        d_opt,
                        d_sched,
                        ema,
                        scaler,
                        step,
                        cfg,
                    )
                    keep_last_checkpoints(output_dir, "step_*.pt", keep=keep_last)

        if is_main_process():
            logger.info("Epoch %d finished in %.1fs", epoch, time.time() - epoch_t0)

    # Final checkpoint.
    if is_main_process():
        final_path = output_dir / "last.pt"
        _save(
            final_path,
            _gen_module(),
            g_opt,
            g_sched,
            _disc_module(),
            d_opt,
            d_sched,
            ema,
            scaler,
            step,
            cfg,
        )
        logger.info("Training finished at step %d. Final ckpt -> %s", step, final_path)

    train_logger.close()
    cleanup_distributed()


def _save(
    path: Path,
    generator: nn.Module,
    g_opt: torch.optim.Optimizer,
    g_sched: Any,
    discriminator: Optional[nn.Module],
    d_opt: Optional[torch.optim.Optimizer],
    d_sched: Any,
    ema: EMA,
    scaler: torch.cuda.amp.GradScaler,
    step: int,
    cfg: DictConfig,
) -> None:
    """Persist a full training checkpoint."""
    extra: Dict[str, Any] = {
        "ema_shadow": {k: v.detach().cpu() for k, v in ema.shadow.items()},
        "config": OmegaConf.to_container(cfg, resolve=True),
    }
    if scaler is not None and scaler.is_enabled():
        try:
            extra["scaler"] = scaler.state_dict()
        except Exception:
            pass
    save_checkpoint(path, generator, g_opt, g_sched, step=step, extra=extra)
    # Append discriminator state on top of saved payload.
    if discriminator is not None:
        payload = torch.load(str(path), map_location="cpu")
        payload["discriminator"] = discriminator.state_dict()
        if d_opt is not None:
            payload["d_optimizer"] = d_opt.state_dict()
        if d_sched is not None and hasattr(d_sched, "state_dict"):
            payload["d_scheduler"] = d_sched.state_dict()
        torch.save(payload, str(path))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="UltraCodec training")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--resume", type=str, default=None, help="Optional checkpoint to resume from")
    parser.add_argument("--output_dir", type=str, default=None, help="Override training.checkpoint.output_dir")
    parser.add_argument("--max_steps", type=int, default=None, help="Override training.max_steps")
    parser.add_argument("--batch_size", type=int, default=None, help="Override training.batch_size")
    parser.add_argument("--seed", type=int, default=None, help="Override seed")
    return parser.parse_args(argv)


def main(argv: Optional[list] = None) -> None:
    args = parse_args(argv)
    init_logger(
        "ultracodec",
        level=logging.INFO if get_rank() == 0 else logging.WARNING,
    )
    cfg = load_config(args.config)
    cfg = apply_cli_overrides(cfg, args)

    resume = args.resume or cfg.training.get("resume_from", None)
    train(cfg, resume=resume)


if __name__ == "__main__":  # pragma: no cover
    main()
