#!/usr/bin/env bash
# LibriSpeech download script with China mirror + progress bar
# Usage: bash scripts/download_librispeech.sh

set -euo pipefail

OUTPUT_DIR="${OUTPUT_DIR:-./data/librispeech}"
MIRROR="https://openslr.magicdatatech.com/resources/12"
FALLBACK="https://www.openslr.org/resources/12"

SPLITS=(train-clean-100 train-clean-360 dev-clean test-clean)

echo "=========================================="
echo " LibriSpeech Downloader (China Mirror)"
echo " Output: ${OUTPUT_DIR}"
echo " Mirror: ${MIRROR}"
echo "=========================================="
echo ""

mkdir -p "${OUTPUT_DIR}"
cd "${OUTPUT_DIR}"

for split in "${SPLITS[@]}"; do
    FILE="${split}.tar.gz"

    if [ -d "LibriSpeech/${split}" ]; then
        echo "[SKIP] ${split} already extracted."
        continue
    fi

    if [ -f "${FILE}" ]; then
        echo "[SKIP] ${FILE} already downloaded, extracting..."
    else
        echo "[DOWN] Downloading ${split}..."
        # Try mirror first, fallback to original
        if ! wget --progress=bar:force:noscroll -c "${MIRROR}/${FILE}" -O "${FILE}" 2>&1; then
            echo "[WARN] Mirror failed, trying original source..."
            wget --progress=bar:force:noscroll -c "${FALLBACK}/${FILE}" -O "${FILE}" 2>&1
        fi
    fi

    echo "[EXTR] Extracting ${FILE}..."
    tar -xzf "${FILE}" --checkpoint=.1000
    echo ""

    echo "[DONE] ${split} ready."
    echo ""
done

echo "=========================================="
echo " All done! Directory structure:"
ls -la LibriSpeech/ 2>/dev/null || echo "  (no LibriSpeech/ dir found)"
echo ""
echo " Total disk usage:"
du -sh . 2>/dev/null
echo "=========================================="
