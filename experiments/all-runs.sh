#!/usr/bin/env bash

set -e
set -x

source .venv/bin/activate

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,2,3,6
# export TORCH_LOGS="dynamo,inductor,graph_breaks"

# HAT Full Training
# accelerate launch -m src.train.train -c configs/train/hat_lsdir.yaml

# XLSIM Full Training
accelerate launch -m src.train.train -c configs/train/xlsim_lsdir.yaml --checkpoint runs/XLSIM/run_20260804_102207/checkpoints/checkpoint_30

# # Burstormer Full Training
# accelerate launch -m src.train.train -c configs/train/burstormer_lsdir.yaml