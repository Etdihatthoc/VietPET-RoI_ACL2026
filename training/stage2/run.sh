#!/bin/bash
#SBATCH --job-name=dice2
#SBATCH --output=result.txt
#SBATCH --error=error.txt
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --nodelist=dgx02
cd '/home/user01/aiotlab/sondinh/ACL 2026/Hirra_model/training/stage2'

conda init
conda activate dinhson

bash train.sh

echo "Training completed."
