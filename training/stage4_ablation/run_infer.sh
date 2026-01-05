#!/bin/bash
#SBATCH --job-name=dice4
#SBATCH --output=result.txt
#SBATCH --error=error.txt
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --nodelist=dgx02
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G

# Activate conda environment
conda activate dinhson

# Set working directory
cd '/home/user01/aiotlab/sondinh/ACL 2026/Hirra_model/training/stage4_ablation'

# Run inference on entire validation set
# Arguments: [config] [checkpoint] [output_csv] [num_samples]
# Leave num_samples empty to process ALL validation samples
bash infer.sh \
    config_stage4.yaml \
    outputs_ablation_stage2/checkpoints/best.pt \
    outputs_ablation_stage2/predictions/inference_predictions.csv

echo "SBATCH job completed!"
