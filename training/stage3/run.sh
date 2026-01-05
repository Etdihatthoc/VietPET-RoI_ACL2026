#!/bin/bash
#SBATCH --job-name=dice3
#SBATCH --output=result.txt
#SBATCH --error=error.txt
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --nodelist=dgx02
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
cd "/home/user01/aiotlab/sondinh/ACL 2026/Hirra_model/training/stage3"

conda init
conda activate dinhson

bash train_stage3.sh

echo "Training completed."
