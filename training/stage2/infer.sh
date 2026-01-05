#!/bin/bash
# Inference script for Stage 2
# Usage: bash infer.sh [config_path] [checkpoint_path] [output_csv_path] [num_samples]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_CONFIG="${SCRIPT_DIR}/config_stage2.yaml"
DEFAULT_CHECKPOINT="${SCRIPT_DIR}/outputs/checkpoints/best.pt"
DEFAULT_OUTPUT="${SCRIPT_DIR}/inference_predictions.csv"
DEFAULT_NUM_SAMPLES=20

CONFIG_PATH="${1:-$DEFAULT_CONFIG}"
CHECKPOINT_PATH="${2:-$DEFAULT_CHECKPOINT}"
OUTPUT_PATH="${3:-$DEFAULT_OUTPUT}"
NUM_SAMPLES="${4:-$DEFAULT_NUM_SAMPLES}"

echo "============================================"
echo "[Stage2 Inference]"
echo "============================================"
echo "Config: ${CONFIG_PATH}"
echo "Checkpoint: ${CHECKPOINT_PATH}"
echo "Output: ${OUTPUT_PATH}"
echo "Number of samples: ${NUM_SAMPLES}"
echo "============================================"

python -u "${SCRIPT_DIR}/infer.py" \
    --config "${CONFIG_PATH}" \
    --checkpoint "${CHECKPOINT_PATH}" \
    --output "${OUTPUT_PATH}" \
    --num_samples "${NUM_SAMPLES}" \
    --max_gen_tokens 1024

echo "============================================"
echo "Inference completed!"
echo "Results saved to: ${OUTPUT_PATH}"
echo "============================================"
