#!/bin/bash
#SBATCH --job-name=dice2
#SBATCH --output=result.txt
#SBATCH --error=error.txt
#SBATCH --ntasks=1
#SBATCH --gpus=1
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

conda init
conda activate dinhson

bash train.sh

echo "Training completed."
