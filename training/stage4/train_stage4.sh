#!/bin/bash

# Run training
python '/home/user01/aiotlab/sondinh/ACL 2026/Hirra_model/training/stage4/train.py' --config '/home/user01/aiotlab/sondinh/ACL 2026/Hirra_model/training/stage4/config_stage4.yaml'

# Optional: If you want to override the device in config
# python train.py --config config_stage4.yaml --device cuda:0
