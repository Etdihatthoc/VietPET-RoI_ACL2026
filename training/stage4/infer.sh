#!/bin/bash
# Stage 4 Inference Script
# Usage: bash infer.sh [config] [checkpoint] [output_csv] [num_samples]
#   num_samples: Number of samples to generate (default: empty = ALL validation samples)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DEFAULT_CONFIG="${SCRIPT_DIR}/config_stage4.yaml"
DEFAULT_CHECKPOINT="${SCRIPT_DIR}/outputs_ROI/checkpoints/best.pt"
DEFAULT_OUTPUT="${SCRIPT_DIR}/inference_predictions.csv"

CONFIG_PATH="${1:-$DEFAULT_CONFIG}"
CHECKPOINT_PATH="${2:-$DEFAULT_CHECKPOINT}"
OUTPUT_PATH="${3:-$DEFAULT_OUTPUT}"
NUM_SAMPLES="${4:-}"  # Empty = ALL samples

echo "============================================"
echo "[Stage 4 Inference - ROI Reasoning]"
echo "============================================"
echo "Config: ${CONFIG_PATH}"
echo "Checkpoint: ${CHECKPOINT_PATH}"
echo "Output: ${OUTPUT_PATH}"
if [ -z "$NUM_SAMPLES" ]; then
    echo "Number of samples: ALL (entire validation set)"
else
    echo "Number of samples: ${NUM_SAMPLES}"
fi
echo "============================================"

# Build command with optional num_samples
python infer.py \
    --config ${CONFIG_PATH} \
    --checkpoint ${CHECKPOINT_PATH} \
    --output ${OUTPUT_PATH} \
    --max_gen_tokens 1024 \
    --device cuda

echo "============================================"
echo "Inference completed!"
echo "Results saved to: ${OUTPUT_PATH}"
echo "============================================"
