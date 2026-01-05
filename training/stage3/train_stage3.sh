#!/usr/bin/env bash

# Get script directory and config path
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_CONFIG="${SCRIPT_DIR}/config_stage3.yaml"
CONFIG_PATH="${1:-$DEFAULT_CONFIG}"

echo "============================================================"
echo "Stage 3 Training: Local (RoI) Feature Alignment"
echo "============================================================"
echo "[Stage3] Script directory: ${SCRIPT_DIR}"
echo "[Stage3] Using config: ${CONFIG_PATH}"
echo "[Stage3] CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "[Stage3] PYTORCH_CUDA_ALLOC_CONF: ${PYTORCH_CUDA_ALLOC_CONF}"
echo "============================================================"
echo ""

# Run training
python -u "${SCRIPT_DIR}/train.py" --config "${CONFIG_PATH}"

echo ""
echo "============================================================"
echo "[Stage3] Training completed!"
echo "============================================================"
