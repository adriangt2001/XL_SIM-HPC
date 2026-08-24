#!/usr/bin/env bash

set -e
set -x

source .venv/bin/activate

# export CUDA_DEVICE_ORDER=PCI_BUS_ID
# export CUDA_VISIBLE_DEVICES=2,6
# export TORCH_LOGS="dynamo,inductor,graph_breaks"

# # Swin2SR Finetuning
# uv run -m src.train.train -c configs/train/swin2sr_biosr_ft.yaml

# # GSASR Finetuning
# uv run -m src.train.train -c configs/train/gsasr_biosr_ft.yaml

# # HAT Finetuning
# uv run -m src.train.train -c configs/train/hat_biosr_ft.yaml

# # XLSIM Finetuning
# python -m src.train.train -c configs/train/xlsim_biosr_ft.yaml