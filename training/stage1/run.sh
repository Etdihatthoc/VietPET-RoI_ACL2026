#!/bin/bash
#SBATCH --job-name=dice1
#SBATCH --output=result.txt
#SBATCH --error=error.txt
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --nodelist=dgx01

cd /home/user01/aiotlab/sondinh/ACL 2026/Hirra_model/training/stage1

conda init
conda activate dinhson

bash train.sh

echo "Training completed."


