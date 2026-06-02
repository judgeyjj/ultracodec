#!/bin/bash
# UltraCodec Training Launch Script
# Hardware: Single A100 42GB, GPU index 2

export CUDA_VISIBLE_DEVICES=2

STAGE=${1:-1}
EXTRA_ARGS="${@:2}"

echo "=========================================="
echo " UltraCodec Training - Stage ${STAGE}"
echo " GPU: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "=========================================="

case $STAGE in
  1)
    echo "Stage 1: Basic reconstruction (no discriminator, no AFR)"
    echo "Config: batch=32, accum=4, effective_batch=128"
    python scripts/train.py --config configs/train_stage1.yaml $EXTRA_ARGS
    ;;
  2)
    echo "Stage 2: LLM-aware fine-tuning (+ discriminator + AFR)"
    echo "Config: batch=12, accum=10, effective_batch=120"
    python scripts/train.py --config configs/train_stage2.yaml $EXTRA_ARGS
    ;;
  3)
    echo "Stage 3: Joint codec + LLM LoRA fine-tuning"
    echo "Config: batch=8, accum=16, effective_batch=128"
    python scripts/train.py --config configs/train_stage3.yaml $EXTRA_ARGS
    ;;
  *)
    echo "Usage: bash scripts/run_train.sh [1|2|3] [extra args]"
    echo "Examples:"
    echo "  bash scripts/run_train.sh 1                  # Start stage 1"
    echo "  bash scripts/run_train.sh 2 --resume ckpt.pt # Resume stage 2"
    echo "  bash scripts/run_train.sh 3 --max_steps 10   # Quick test"
    exit 1
    ;;
esac
