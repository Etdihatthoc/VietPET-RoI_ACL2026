#!/bin/bash
#SBATCH --job-name=dice3
#SBATCH --output=result.txt
#SBATCH --error=error.txt
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

conda init
conda activate dinhson

bash train_stage3.sh

echo "Training completed."
