#!/usr/bin/env bash

set -e
set -x

source .venv/bin/activate

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=2,6
export TORCH_LOGS="dynamo,inductor,graph_breaks"

# HAT Full Training
accelerate launch -m src.train.train -c configs/train/hat_lsdir.yaml --num_iterations 10 --valid_freq 10 --warmup_iterations 5

# XLSIM Full Training
accelerate launch -m src.train.train -c configs/train/xlsim_lsdir.yaml --num_iterations 10 --valid_freq 10 --warmup_iterations 5

# Burstormer Full Training
# python -m src.train.train -c configs/train/burstormer_lsdir.yaml --num_iterations 10 --valid_freq 10