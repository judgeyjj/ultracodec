#!/usr/bin/env python3
"""UltraCodec evaluation entry point.

Loads a trained checkpoint, runs encode → decode on every clip in the
configured datasets and computes the metric battery described in
``configs/eval.yaml``. Supports CSV / JSON output, per-clip detail logs and
optional comparison against a baseline checkpoint.

Examples
--------
Evaluate a checkpoint with the default config::

    python scripts/evaluate.py --config configs/eval.yaml \\
        --checkpoint runs/uc_stage3/best.pt

Evaluate only on the VCTK split with a custom metric subset::

    python scripts/evaluate.py --checkpoint runs/uc_stage3/best.pt \\
        --dataset vctk-test --metrics pesq,stoi,utmos

Compare against a baseline (e.g. EnCodec checkpoint)::

    python scripts/evaluate.py --checkpoint runs/uc_stage3/best.pt \\
        --baseline_checkpoint runs/baselines/encodec.pt
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultracodec.data import LibriSpeechDataset, VCTKDataset
from ultracodec.metrics import MetricCalculator
from ultracodec.utils import get_device, init_logger, load_checkpoint
from ultracodec.utils.utils import to_device

logger = logging.getLogger("ultracodec.evaluate")


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------
def load_config(path: str) -> DictConfig:
    """Load a YAML config and merge ``defaults:`` parents (OmegaConf style)."""
    cfg = OmegaConf.load(path)
    parents = []
    if isinstance(cfg.get("defaults", None), list):
        cfg_dir = Path(path).parent
        for entry in cfg.pop("defaults"):
            parent_path = cfg_dir / f"{entry}.yaml"
            if parent_path.exists():
                parents.append(OmegaConf.load(parent_path))
    return OmegaConf.merge(*parents, cfg) if parents else cfg  # type: ignore[return-value]


def build_dataset(spec, cfg: DictConfig):
    """Instantiate an evaluation dataset from a config spec dict."""
    name = str(spec.name)
    if name.startswith("librispeech"):
        return LibriSpeechDataset(
            root_dir=spec.root,
            split=spec.split,
            download=False,
            sample_rate=cfg.data.sample_rate,
            segment_length=-1,
            augment=False,
        )
    if name.startswith("vctk"):
        return VCTKDataset(
            root_dir=spec.root,
            split=spec.split,
            sample_rate=cfg.data.sample_rate,
            segment_length=-1,
            augment=False,
        )
    raise ValueError(f"Unknown evaluation dataset: {name}")


def build_model(cfg: DictConfig) -> torch.nn.Module:
    """Build an UltraCodec instance, falling back to identity if unimported."""
    try:
        from ultracodec.model import UltraCodec  # type: ignore[attr-defined]

        return UltraCodec(cfg.model)
    except (ImportError, AttributeError):
        logger.info("Model not implemented yet; returning identity placeholder.")

        class _Identity(torch.nn.Module):
            sample_rate = 16000

            def forward(self, wav: torch.Tensor):  # type: ignore[override]
                return {"x_hat": wav}

            @torch.no_grad()
            def encode(self, wav: torch.Tensor):  # type: ignore[override]
                return {"codes": torch.zeros(1, 1, 1, dtype=torch.long), "gate": None}

            @torch.no_grad()
            def decode(self, codes, gate=None):  # type: ignore[override]
                return torch.zeros_like(codes, dtype=torch.float32)

        return _Identity()


# ---------------------------------------------------------------------------
# Reconstruction helper
# ---------------------------------------------------------------------------
@torch.no_grad()
def reconstruct(model: torch.nn.Module, wav: torch.Tensor) -> torch.Tensor:
    """Run ``wav`` through the codec end-to-end and return the reconstruction.

    Tries (in order) ``model.forward``, then explicit encode→decode, so it
    works whether the model exposes a single ``forward`` or separate
    ``encode``/``decode`` methods.
    """
    try:
        out = model(wav)
        if isinstance(out, dict):
            for key in ("x_hat", "wav_hat", "audio"):
                if key in out:
                    return out[key]
        if isinstance(out, torch.Tensor):
            return out
    except Exception as exc:
        logger.debug("forward() failed: %s; trying encode/decode.", exc)

    enc = model.encode(wav)
    if isinstance(enc, dict):
        codes = enc["codes"]
        gate = enc.get("gate", None)
        return model.decode(codes, gate=gate)
    if isinstance(enc, tuple):
        return model.decode(*enc)
    return model.decode(enc)


# ---------------------------------------------------------------------------
# Metric aggregation
# ---------------------------------------------------------------------------
def _flatten_metrics(cfg_metrics: Any) -> List[str]:
    """Flatten the nested ``evaluation.metrics`` block into a list of names."""
    if cfg_metrics is None:
        return []
    if isinstance(cfg_metrics, (list, tuple)):
        return [str(m) for m in cfg_metrics]
    out: List[str] = []
    for _, group in cfg_metrics.items():
        if isinstance(group, (list, tuple)):
            out.extend(str(m) for m in group)
    return out


def _summarise(detail: List[Dict[str, float]]) -> Dict[str, float]:
    """Mean over per-clip detail rows, ignoring NaNs."""
    if not detail:
        return {}
    keys = sorted({k for row in detail for k in row.keys() if k != "id"})
    summary: Dict[str, float] = {}
    for k in keys:
        values = [row[k] for row in detail if k in row and not _is_nan(row[k])]
        if values:
            summary[k] = float(statistics.fmean(values))
        else:
            summary[k] = float("nan")
    return summary


def _is_nan(x: Any) -> bool:
    try:
        return isinstance(x, float) and x != x  # NaN check
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(
    cfg: DictConfig,
    checkpoint: Optional[str] = None,
    dataset_filter: Optional[str] = None,
    metric_filter: Optional[List[str]] = None,
    max_clips: Optional[int] = None,
) -> Dict[str, Dict[str, Any]]:
    """Evaluate ``checkpoint`` on every dataset declared in ``cfg.evaluation``.

    Args:
        cfg: Merged OmegaConf config (must contain ``evaluation.datasets``).
        checkpoint: Optional checkpoint path overriding ``cfg.evaluation.checkpoint``.
        dataset_filter: If set, only datasets whose ``name`` equals this are run.
        metric_filter: Optional whitelist of metric names.
        max_clips: Optional cap on the number of clips per dataset (debugging).

    Returns:
        Mapping ``{dataset_name: {"summary": {...}, "detail": [...]}}``.
    """
    device = get_device()
    model = build_model(cfg).to(device).eval()
    if checkpoint:
        load_checkpoint(checkpoint, model, map_location=device, strict=False)
        logger.info("Loaded checkpoint %s.", checkpoint)

    eval_cfg = cfg.evaluation
    metrics_list = metric_filter or _flatten_metrics(eval_cfg.get("metrics", None))
    whisper_model = str(eval_cfg.get("whisper_model", "openai/whisper-small"))
    sample_rate = int(cfg.data.sample_rate)

    calculator = MetricCalculator(
        sample_rate=sample_rate,
        device=str(device),
        whisper_model=whisper_model,
        enable_whisper=("wer" in metrics_list) if metrics_list else True,
        enable_utmos=("utmos" in metrics_list) if metrics_list else True,
    )

    results: Dict[str, Dict[str, Any]] = {}
    for spec in eval_cfg.datasets:
        if dataset_filter and str(spec.name) != dataset_filter:
            continue
        try:
            dataset = build_dataset(spec, cfg)
        except Exception as exc:
            logger.warning("Failed to build dataset %s: %s", spec.name, exc)
            continue

        loader = DataLoader(
            dataset,
            batch_size=int(eval_cfg.batch_size),
            shuffle=False,
            num_workers=int(eval_cfg.num_workers),
            collate_fn=lambda b: b,  # variable-length samples
        )

        detail: List[Dict[str, float]] = []
        coding_rows: List[Dict[str, float]] = []
        n_clips = 0
        t0 = time.time()
        for batch in loader:
            for sample in batch:
                if max_clips is not None and n_clips >= max_clips:
                    break
                wav = to_device(sample["wav"].unsqueeze(0), device)
                wav_hat = reconstruct(model, wav)

                # 1-D float32 numpy waveforms for metric calculators.
                ref_np = wav.squeeze().detach().cpu().numpy().astype(np.float32)
                deg_np = wav_hat.squeeze().detach().cpu().numpy().astype(np.float32)

                row: Dict[str, float] = {"id": str(sample.get("id", n_clips))}
                row.update(
                    calculator.compute_all(
                        ref_np, deg_np, sr=sample_rate, metrics=metrics_list or None,
                    )
                )

                # Coding-side metrics (need the model + tensor).
                if metrics_list is None or "bitrate" in metrics_list:
                    row["bitrate_kbps"] = calculator.compute_bitrate(model, wav)
                if metrics_list is None or "frame_rate" in metrics_list:
                    row["frame_rate_hz"] = calculator.compute_frame_rate(model, wav)
                if metrics_list is None or "compression_ratio" in metrics_list:
                    bps = row.get("bitrate_kbps", float("nan")) * 1000.0
                    raw_bps = sample_rate * 16  # 16-bit PCM
                    row["compression_ratio"] = float(raw_bps / bps) if bps > 0 else float("nan")
                detail.append(row)
                coding_rows.append(row)
                n_clips += 1
            if max_clips is not None and n_clips >= max_clips:
                break

        elapsed = time.time() - t0
        summary = _summarise(detail)
        summary["n_clips"] = n_clips
        summary["elapsed_sec"] = round(elapsed, 2)
        results[str(spec.name)] = {"summary": summary, "detail": detail}
        _log_summary(spec.name, summary)

    return results


def _log_summary(name: str, summary: Dict[str, float]) -> None:
    """Pretty-print a metric summary line for one dataset."""
    pieces = [f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in summary.items()]
    logger.info("[%s] %s", name, " | ".join(pieces))


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------
def write_outputs(results: Dict[str, Dict[str, Any]], out_dir: Path) -> None:
    """Persist results as ``metrics.json`` (summary + detail) and per-set CSVs."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # JSON: summary + detail.
    (out_dir / "metrics.json").write_text(
        json.dumps(results, indent=2, default=_json_default), encoding="utf-8"
    )

    # CSV: one file per dataset with detail rows.
    for name, payload in results.items():
        detail = payload.get("detail", [])
        if not detail:
            continue
        keys = sorted({k for row in detail for k in row.keys()})
        csv_path = out_dir / f"{name}.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for row in detail:
                writer.writerow(row)

    # Aggregate summary table (TSV, easy to paste into the paper).
    tsv_path = out_dir / "summary.tsv"
    metric_keys = sorted({
        k for payload in results.values() for k in payload.get("summary", {}).keys()
    })
    with tsv_path.open("w", encoding="utf-8") as f:
        f.write("dataset\t" + "\t".join(metric_keys) + "\n")
        for name, payload in results.items():
            summary = payload.get("summary", {})
            row = [name] + [
                f"{summary[k]:.4f}" if isinstance(summary.get(k), float) else str(summary.get(k, ""))
                for k in metric_keys
            ]
            f.write("\t".join(row) + "\n")


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def diff_results(
    main: Dict[str, Dict[str, Any]],
    base: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, float]]:
    """Compute (main - baseline) on summary metrics for shared datasets."""
    out: Dict[str, Dict[str, float]] = {}
    for name, payload in main.items():
        if name not in base:
            continue
        a = payload.get("summary", {})
        b = base[name].get("summary", {})
        diff: Dict[str, float] = {}
        for k in a.keys() & b.keys():
            try:
                diff[k] = float(a[k]) - float(b[k])
            except (TypeError, ValueError):
                continue
        out[name] = diff
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="UltraCodec evaluation")
    parser.add_argument("--config", type=str, default="configs/eval.yaml")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to UltraCodec checkpoint (.pt)")
    parser.add_argument("--baseline_checkpoint", type=str, default=None,
                        help="Optional baseline checkpoint to compare against")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Restrict evaluation to a single dataset name")
    parser.add_argument("--metrics", type=str, default=None,
                        help="Comma-separated metric whitelist (e.g. pesq,stoi,utmos)")
    parser.add_argument("--max_clips", type=int, default=None,
                        help="Optional cap on number of clips per dataset")
    parser.add_argument("--output_dir", type=str, default=None)
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    init_logger("ultracodec", level=logging.INFO)
    args = parse_args(argv)
    cfg = load_config(args.config)
    if args.output_dir is not None:
        cfg.evaluation.output_dir = args.output_dir
    out_dir = Path(cfg.evaluation.output_dir)

    metric_filter = (
        [m.strip() for m in args.metrics.split(",") if m.strip()]
        if args.metrics else None
    )
    checkpoint = args.checkpoint or cfg.evaluation.get("checkpoint", None)

    results = evaluate(
        cfg,
        checkpoint=checkpoint,
        dataset_filter=args.dataset,
        metric_filter=metric_filter,
        max_clips=args.max_clips,
    )
    write_outputs(results, out_dir)
    logger.info("Wrote main results -> %s", out_dir / "metrics.json")

    if args.baseline_checkpoint:
        logger.info("Evaluating baseline checkpoint: %s", args.baseline_checkpoint)
        baseline = evaluate(
            cfg,
            checkpoint=args.baseline_checkpoint,
            dataset_filter=args.dataset,
            metric_filter=metric_filter,
            max_clips=args.max_clips,
        )
        baseline_dir = out_dir / "baseline"
        write_outputs(baseline, baseline_dir)
        diff = diff_results(results, baseline)
        (out_dir / "diff_vs_baseline.json").write_text(
            json.dumps(diff, indent=2), encoding="utf-8",
        )
        for name, dvals in diff.items():
            pieces = [f"{k}={v:+.4f}" for k, v in dvals.items()]
            logger.info("[%s] Δ vs baseline: %s", name, " | ".join(pieces))


if __name__ == "__main__":  # pragma: no cover
    main()
