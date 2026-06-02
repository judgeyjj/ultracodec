#!/usr/bin/env bash
# One-stop dataset download for UltraCodec.
# Downloads LibriSpeech, MUSAN and DNS Challenge into ./data by default.

set -euo pipefail

OUTPUT_DIR="${OUTPUT_DIR:-./data}"
DATASETS="${DATASETS:-librispeech,musan,dns}"
LIBRISPEECH_SPLITS="${LIBRISPEECH_SPLITS:-train-clean-100,train-clean-360,dev-clean,test-clean}"
DNS_VERSION="${DNS_VERSION:-2020}"

echo "== UltraCodec dataset downloader =="
echo "Output dir       : ${OUTPUT_DIR}"
echo "Datasets         : ${DATASETS}"
echo "LibriSpeech split: ${LIBRISPEECH_SPLITS}"
echo "DNS version      : ${DNS_VERSION}"
echo ""

mkdir -p "${OUTPUT_DIR}"

python -m ultracodec.data.download \
    --output_dir "${OUTPUT_DIR}" \
    --datasets "${DATASETS}" \
    --librispeech_splits "${LIBRISPEECH_SPLITS}" \
    --dns_version "${DNS_VERSION}"

echo ""
echo "Done. VCTK is expected to live at:"
echo "  /data01/audio_group/m24_yuanjiajun/AP-BWE/VCTK-Corpus-0.92/wav_test"
echo "(no automatic download; the user already has this data)"
