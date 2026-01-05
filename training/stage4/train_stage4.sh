#!/bin/bash

# Run training
python train.py --config config_stage4.yaml

# Optional: If you want to override the device in config
# python train.py --config config_stage4.yaml --device cuda:0
