#!/bin/bash
#SBATCH --job-name=dice4
#SBATCH --output=result.txt
#SBATCH --error=error.txt
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G

# Activate conda environment
conda activate dinhson

# Set working directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Run inference on entire validation set
# Arguments: [config] [checkpoint] [output_csv] [num_samples]
# Leave num_samples empty to process ALL validation samples
bash infer.sh \
    config_stage4.yaml \
    outputs_lora_stage2_noGraph/checkpoints/best.pt \
    outputs_lora_stage2_noGraph/predictions/inference_predictions.csv

echo "SBATCH job completed!"
