#!/bin/bash

# Stage 1 Training Script
# Simple single-GPU training

echo "========================================="
echo "Stage 1: CLIP-style Pretraining"
echo "========================================="

# Set GPU device
conda init
conda activate dinhson
#unset PYTORCH_CUDA_ALLOC_CONF
# export CUDA_VISIBLE_DEVICES=0

# export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"


# Run training
python train_CLIP.py

echo "========================================="
echo "Training finished!"
echo "========================================="
