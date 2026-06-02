#!/usr/bin/env python3
"""UltraCodec encode / decode demo.

Three sub-commands:

* ``encode``    -- waveform → discrete codes blob (``.uc`` binary file).
* ``decode``    -- ``.uc`` blob → reconstructed waveform.
* ``roundtrip`` -- encode → decode → save reconstruction + quality stats.

Examples
--------
::

    # Encode one file
    python scripts/encode_decode.py encode \\
        --input audio.wav --output audio.uc --checkpoint runs/best.pt

    # Decode it back
    python scripts/encode_decode.py decode \\
        --input audio.uc --output audio_recon.wav --checkpoint runs/best.pt

    # Round-trip with metrics + side-by-side wav files
    python scripts/encode_decode.py roundtrip \\
        --input audio.wav --output_dir ./out/ --checkpoint runs/best.pt

    # Batch round-trip on a directory
    python scripts/encode_decode.py roundtrip \\
        --input ./wavs/ --output_dir ./out/ --checkpoint runs/best.pt
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultracodec.metrics import MetricCalculator
from ultracodec.utils import (
    get_device,
    init_logger,
    load_audio,
    load_checkpoint,
    save_audio,
)

logger = logging.getLogger("ultracodec.encode_decode")

AUDIO_SUFFIXES = {".wav", ".flac", ".mp3", ".ogg", ".opus"}


# ---------------------------------------------------------------------------
# Config / model helpers
# ---------------------------------------------------------------------------
def load_config(path: str) -> DictConfig:
    cfg = OmegaConf.load(path)
    parents = []
    if isinstance(cfg.get("defaults", None), list):
        cfg_dir = Path(path).parent
        for entry in cfg.pop("defaults"):
            parent_path = cfg_dir / f"{entry}.yaml"
            if parent_path.exists():
                parents.append(OmegaConf.load(parent_path))
    return OmegaConf.merge(*parents, cfg) if parents else cfg  # type: ignore[return-value]


def build_model(cfg: DictConfig) -> torch.nn.Module:
    """Instantiate UltraCodec, falling back to identity placeholder."""
    try:
        from ultracodec.model import UltraCodec  # type: ignore[attr-defined]

        return UltraCodec(cfg.model)
    except (ImportError, AttributeError):
        logger.info("Model not implemented yet; using identity placeholder.")

        class _Identity(torch.nn.Module):
            sample_rate = 16000

            @torch.no_grad()
            def encode(self, wav: torch.Tensor):  # type: ignore[override]
                return {"codes": torch.zeros(1, 1, 1, dtype=torch.long), "gate": None}

            @torch.no_grad()
            def decode(self, codes, gate=None):  # type: ignore[override]
                return torch.zeros(1, 1, 16000, dtype=torch.float32)

            @torch.no_grad()
            def compress(self, wav: torch.Tensor):  # type: ignore[override]
                return wav.cpu().numpy().astype(np.float32).tobytes()

            @torch.no_grad()
            def decompress(self, blob, device=None):  # type: ignore[override]
                arr = np.frombuffer(blob, dtype=np.float32).copy()
                return torch.tensor(arr).reshape(1, 1, -1)

            def forward(self, wav: torch.Tensor):  # type: ignore[override]
                return {"x_hat": wav}

        return _Identity()


def _list_inputs(path: Path) -> List[Path]:
    """Expand ``path`` into a list of audio files (single file or directory)."""
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(p for p in path.rglob("*") if p.suffix.lower() in AUDIO_SUFFIXES)
    raise FileNotFoundError(f"Input path not found: {path}")


def _load_model(args: argparse.Namespace) -> tuple[torch.nn.Module, DictConfig, torch.device]:
    cfg = load_config(args.config)
    device = get_device()
    model = build_model(cfg).to(device).eval()
    if args.checkpoint:
        load_checkpoint(args.checkpoint, model, map_location=device, strict=False)
        logger.info("Loaded checkpoint %s.", args.checkpoint)
    return model, cfg, device


# ---------------------------------------------------------------------------
# Sub-commands
# ---------------------------------------------------------------------------
@torch.no_grad()
def cmd_encode(args: argparse.Namespace) -> None:
    """Encode an audio file to a binary code blob."""
    model, cfg, device = _load_model(args)
    sample_rate = int(cfg.model.sample_rate)
    wav, _sr = load_audio(args.input, target_sr=sample_rate)
    wav = wav.unsqueeze(0).to(device)  # [1, 1, T]

    if hasattr(model, "compress"):
        blob = model.compress(wav)
    else:
        # Fallback: serialise tensors with torch.save.
        result = model.encode(wav)
        import io

        buf = io.BytesIO()
        torch.save(result, buf)
        blob = buf.getvalue()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(blob)
    seconds = wav.shape[-1] / sample_rate
    bitrate = (len(blob) * 8) / (seconds * 1000.0) if seconds > 0 else float("nan")
    logger.info(
        "Encoded %s -> %s | %.2fs audio | %d bytes | ~%.2f kbps",
        args.input, out_path, seconds, len(blob), bitrate,
    )


@torch.no_grad()
def cmd_decode(args: argparse.Namespace) -> None:
    """Decode a binary code blob produced by ``cmd_encode``."""
    model, cfg, device = _load_model(args)
    sample_rate = int(cfg.model.sample_rate)
    blob = Path(args.input).read_bytes()

    if hasattr(model, "decompress"):
        wav_hat = model.decompress(blob, device=device)
    else:
        import io

        buf = io.BytesIO(blob)
        result = torch.load(buf, map_location=device)
        codes = result["codes"] if isinstance(result, dict) else result
        gate = result.get("gate", None) if isinstance(result, dict) else None
        wav_hat = model.decode(codes, gate=gate)

    wav_hat = wav_hat.squeeze(0).detach().cpu()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_audio(out_path, wav_hat, sample_rate)
    logger.info("Decoded %s -> %s | %d samples", args.input, out_path, wav_hat.shape[-1])


@torch.no_grad()
def cmd_roundtrip(args: argparse.Namespace) -> None:
    """Encode + decode + reconstruction quality report."""
    model, cfg, device = _load_model(args)
    sample_rate = int(cfg.model.sample_rate)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metric_filter: Optional[List[str]] = (
        [m.strip() for m in args.metrics.split(",") if m.strip()]
        if args.metrics else ["pesq", "stoi", "si_sdr", "lsd", "mcd"]
    )
    calculator = MetricCalculator(
        sample_rate=sample_rate,
        device=str(device),
        enable_whisper="wer" in metric_filter,
        enable_utmos="utmos" in metric_filter,
    )

    inputs = _list_inputs(Path(args.input))
    summary: List[dict] = []
    for in_path in inputs:
        stem = in_path.stem
        wav, _sr = load_audio(str(in_path), target_sr=sample_rate)
        wav = wav.unsqueeze(0).to(device)  # [1, 1, T]
        seconds = wav.shape[-1] / sample_rate

        t0 = time.time()
        if hasattr(model, "compress") and hasattr(model, "decompress"):
            blob = model.compress(wav)
            wav_hat = model.decompress(blob, device=device)
            blob_size = len(blob)
        else:
            result = model.encode(wav)
            codes = result["codes"]
            gate = result.get("gate", None)
            wav_hat = model.decode(codes, gate=gate)
            blob_size = int(codes.numel() * 2)  # rough: 16 bits/code
        elapsed = time.time() - t0

        ref_np = wav.squeeze().detach().cpu().numpy().astype(np.float32)
        deg_np = wav_hat.squeeze().detach().cpu().numpy().astype(np.float32)

        # Quality metrics
        scores = calculator.compute_all(
            ref_np, deg_np, sr=sample_rate, metrics=metric_filter,
        )

        # Coding stats
        bitrate = (blob_size * 8) / (seconds * 1000.0) if seconds > 0 else float("nan")
        try:
            frame_rate = calculator.compute_frame_rate(model, wav)
            codec_bitrate = calculator.compute_bitrate(model, wav)
        except Exception:
            frame_rate, codec_bitrate = float("nan"), float("nan")

        # Save outputs
        save_audio(out_dir / f"{stem}_orig.wav", wav.squeeze(0).cpu(), sample_rate)
        save_audio(out_dir / f"{stem}_recon.wav", wav_hat.squeeze(0).detach().cpu(), sample_rate)
        if hasattr(model, "compress"):
            (out_dir / f"{stem}.uc").write_bytes(blob)

        row = {
            "file": str(in_path),
            "duration_sec": round(seconds, 3),
            "encode_decode_time_sec": round(elapsed, 3),
            "blob_bytes": blob_size,
            "bitrate_kbps": round(bitrate, 4),
            "codec_bitrate_kbps": round(codec_bitrate, 4) if codec_bitrate == codec_bitrate else None,
            "frame_rate_hz": round(frame_rate, 4) if frame_rate == frame_rate else None,
            "compression_ratio": round((sample_rate * 16) / (bitrate * 1000.0), 2)
            if bitrate and bitrate > 0 else None,
            **{k: (round(v, 4) if isinstance(v, float) and v == v else None) for k, v in scores.items()},
        }
        summary.append(row)
        _log_row(row)

    report = out_dir / "roundtrip_report.json"
    report.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Roundtrip report -> %s (%d files)", report, len(summary))


def _log_row(row: dict) -> None:
    keys = ["file", "duration_sec", "bitrate_kbps", "frame_rate_hz", "pesq", "stoi", "si_sdr"]
    pieces = []
    for k in keys:
        if k in row and row[k] is not None:
            v = row[k]
            pieces.append(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}")
    logger.info(" | ".join(pieces))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="UltraCodec encode/decode demo")
    parser.add_argument("--config", type=str, default="configs/eval.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)

    sub = parser.add_subparsers(dest="command", required=True)

    p_enc = sub.add_parser("encode", help="Encode audio file -> binary codes")
    p_enc.add_argument("--input", type=str, required=True)
    p_enc.add_argument("--output", type=str, required=True)

    p_dec = sub.add_parser("decode", help="Decode binary codes -> audio file")
    p_dec.add_argument("--input", type=str, required=True)
    p_dec.add_argument("--output", type=str, required=True)

    p_rt = sub.add_parser("roundtrip", help="Encode+decode and report quality")
    p_rt.add_argument("--input", type=str, required=True,
                      help="File or directory of audio files")
    p_rt.add_argument("--output_dir", type=str, required=True)
    p_rt.add_argument("--metrics", type=str, default=None,
                      help="Comma-separated metric whitelist")

    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Optional[Iterable[str]] = None) -> None:
    init_logger("ultracodec", level=logging.INFO)
    args = parse_args(argv)
    if args.command == "encode":
        cmd_encode(args)
    elif args.command == "decode":
        cmd_decode(args)
    elif args.command == "roundtrip":
        cmd_roundtrip(args)
    else:  # pragma: no cover
        raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":  # pragma: no cover
    main()
