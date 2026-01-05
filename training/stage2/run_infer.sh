#!/bin/bash
#SBATCH --job-name=stage2_infer
#SBATCH --output=inference_result.txt
#SBATCH --error=inference_error.txt
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --nodelist=dgx01

cd /home/user01/aiotlab/sondinh/ACL 2026/Hirra_model/training/stage2

conda init
conda activate dinhson

# Run inference
# Arguments: [config] [checkpoint] [output_csv] [num_samples]
bash infer.sh \
    config_stage2.yaml \
    outputs/checkpoints/best.pt \
    outputs/predictions/inference_predictions.csv \
    125

echo "Inference completed."
