"""Automatic dataset download utilities.

The functions here are best-effort: they require network access and
external tools (``wget`` / ``tar`` / ``unzip``). When unavailable, they log
warnings rather than crash so that import-time semantics stay safe.
"""
from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import tarfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Iterable, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Source URLs
# ---------------------------------------------------------------------------
LIBRISPEECH_URLS = {
    "train-clean-100": "https://www.openslr.org/resources/12/train-clean-100.tar.gz",
    "train-clean-360": "https://www.openslr.org/resources/12/train-clean-360.tar.gz",
    "train-other-500": "https://www.openslr.org/resources/12/train-other-500.tar.gz",
    "dev-clean": "https://www.openslr.org/resources/12/dev-clean.tar.gz",
    "dev-other": "https://www.openslr.org/resources/12/dev-other.tar.gz",
    "test-clean": "https://www.openslr.org/resources/12/test-clean.tar.gz",
    "test-other": "https://www.openslr.org/resources/12/test-other.tar.gz",
}

MUSAN_URL = "https://www.openslr.org/resources/17/musan.tar.gz"

DNS_CHALLENGE_URLS = {
    "2020": "https://github.com/microsoft/DNS-Challenge/archive/refs/heads/interspeech2020/master.zip",
    "2021": "https://github.com/microsoft/DNS-Challenge/archive/refs/heads/interspeech2021/master.zip",
    "2022": "https://github.com/microsoft/DNS-Challenge/archive/refs/heads/icassp_2022/master.zip",
}


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------
def _download_url(url: str, dest: Path, chunk_size: int = 1 << 16) -> Path:
    """Download ``url`` to ``dest``. Returns ``dest``."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        logger.info("File %s already exists; skipping download.", dest)
        return dest

    logger.info("Downloading %s -> %s", url, dest)
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with urllib.request.urlopen(url) as response, open(tmp, "wb") as fh:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                fh.write(chunk)
        shutil.move(str(tmp), str(dest))
    except Exception:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise
    return dest


def _extract(archive: Path, target_dir: Path) -> None:
    """Extract ``archive`` (.tar.gz / .tgz / .zip) into ``target_dir``."""
    target_dir.mkdir(parents=True, exist_ok=True)
    suffixes = "".join(archive.suffixes).lower()
    logger.info("Extracting %s -> %s", archive, target_dir)
    if suffixes.endswith((".tar.gz", ".tgz", ".tar")):
        with tarfile.open(archive, "r:*") as tf:
            tf.extractall(target_dir)
    elif suffixes.endswith(".zip"):
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(target_dir)
    else:
        raise ValueError(f"Unsupported archive format: {archive}")


# ---------------------------------------------------------------------------
# LibriSpeech
# ---------------------------------------------------------------------------
def download_librispeech(
    root: Path | str,
    splits: Iterable[str] = ("train-clean-100", "train-clean-360", "dev-clean", "test-clean"),
    keep_archive: bool = False,
) -> Path:
    """Download and extract LibriSpeech splits under ``root``.

    The resulting layout is::

        root/
          LibriSpeech/
            <split>/...

    Parameters
    ----------
    root:
        Output root directory.
    splits:
        Iterable of LibriSpeech split names (see :data:`LIBRISPEECH_URLS`).
    keep_archive:
        Keep the downloaded ``.tar.gz`` file after extraction.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    for split in splits:
        if split not in LIBRISPEECH_URLS:
            logger.warning("Unknown LibriSpeech split '%s'; skipping.", split)
            continue
        target = root / "LibriSpeech" / split
        if target.exists() and any(target.rglob("*.flac")):
            logger.info("LibriSpeech split %s already present at %s.", split, target)
            continue
        archive = root / f"{split}.tar.gz"
        try:
            _download_url(LIBRISPEECH_URLS[split], archive)
            _extract(archive, root)
        except Exception as exc:
            logger.warning("Failed to download/extract LibriSpeech %s: %s", split, exc)
            continue
        if not keep_archive and archive.exists():
            archive.unlink(missing_ok=True)
    return root / "LibriSpeech"


# ---------------------------------------------------------------------------
# MUSAN
# ---------------------------------------------------------------------------
def download_musan(root: Path | str, keep_archive: bool = False) -> Path:
    """Download and extract MUSAN under ``root``."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    target = root / "musan"
    if target.exists() and any(target.rglob("*.wav")):
        logger.info("MUSAN already present at %s.", target)
        return target
    archive = root / "musan.tar.gz"
    try:
        _download_url(MUSAN_URL, archive)
        _extract(archive, root)
    except Exception as exc:
        logger.warning("Failed to download/extract MUSAN: %s", exc)
        return target
    if not keep_archive and archive.exists():
        archive.unlink(missing_ok=True)
    return target


# ---------------------------------------------------------------------------
# DNS Challenge
# ---------------------------------------------------------------------------
def download_dns_challenge(
    root: Path | str,
    version: str = "2020",
    keep_archive: bool = False,
) -> Path:
    """Download a Microsoft DNS-Challenge release.

    The DNS-Challenge corpus is large and distributed via Git LFS; this
    helper only fetches the GitHub repository snapshot. Users requiring
    the full noise data must run the official ``download-dns-challenge.sh``
    script inside the cloned repository.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    if version not in DNS_CHALLENGE_URLS:
        raise ValueError(
            f"Unknown DNS-Challenge version '{version}'. "
            f"Available: {list(DNS_CHALLENGE_URLS)}"
        )
    target = root / f"DNS-Challenge-{version}"
    if target.exists():
        logger.info("DNS-Challenge %s already present at %s.", version, target)
        return target

    archive = root / f"dns-challenge-{version}.zip"
    try:
        _download_url(DNS_CHALLENGE_URLS[version], archive)
        _extract(archive, root)
        # Rename the unpacked top-level directory if necessary.
        candidates = list(root.glob("DNS-Challenge-*"))
        if candidates and not target.exists():
            candidates[0].rename(target)
    except Exception as exc:
        logger.warning("Failed to download/extract DNS-Challenge %s: %s", version, exc)
        return target
    if not keep_archive and archive.exists():
        archive.unlink(missing_ok=True)

    # Best-effort: invoke the upstream LFS download script when present.
    script = target / "download-dns-challenge.sh"
    if script.exists():
        logger.info(
            "DNS-Challenge LFS data must be fetched manually via %s",
            script,
        )
    return target


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def prepare_all_datasets(
    base_dir: Path | str,
    librispeech_splits: Optional[Iterable[str]] = None,
    include_musan: bool = True,
    include_dns: bool = True,
    dns_version: str = "2020",
) -> dict:
    """One-stop helper that downloads all configured datasets.

    Returns a dict mapping dataset name -> on-disk root path.
    """
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    out: dict = {}

    splits = tuple(librispeech_splits) if librispeech_splits else (
        "train-clean-100",
        "train-clean-360",
        "dev-clean",
        "test-clean",
    )
    out["librispeech"] = download_librispeech(base_dir / "librispeech", splits=splits)

    if include_musan:
        out["musan"] = download_musan(base_dir / "musan")
    if include_dns:
        out["dns"] = download_dns_challenge(base_dir / "dns", version=dns_version)
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="UltraCodec dataset downloader")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data",
        help="Root directory under which all datasets are stored.",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="librispeech,musan,dns",
        help="Comma-separated list of datasets to download.",
    )
    parser.add_argument(
        "--librispeech_splits",
        type=str,
        default="train-clean-100,train-clean-360,dev-clean,test-clean",
        help="Comma-separated list of LibriSpeech splits to download.",
    )
    parser.add_argument("--dns_version", type=str, default="2020")
    parser.add_argument("--keep_archives", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[list] = None) -> None:
    """CLI entry point: ``python -m ultracodec.data.download``."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    )
    args = _parse_args(argv)
    base = Path(args.output_dir)
    requested = {x.strip() for x in args.datasets.split(",") if x.strip()}

    if "librispeech" in requested:
        splits = [s.strip() for s in args.librispeech_splits.split(",") if s.strip()]
        download_librispeech(base / "librispeech", splits=splits, keep_archive=args.keep_archives)
    if "musan" in requested:
        download_musan(base / "musan", keep_archive=args.keep_archives)
    if "dns" in requested:
        download_dns_challenge(base / "dns", version=args.dns_version, keep_archive=args.keep_archives)


if __name__ == "__main__":  # pragma: no cover
    main()
