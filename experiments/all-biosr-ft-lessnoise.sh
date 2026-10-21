#!/usr/bin/env bash

set -e
set -x

source .venv/bin/activate

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=2,6

# # Swin2SR BioSR Finetuning
# accelerate launch -m src.train.train -c configs/train/swin2sr_biosr_ft_lessnoise.yaml

# # GSASR BioSR Finetuning
# accelerate launch -m src.train.train -c configs/train/gsasr_biosr_ft_lessnoise.yaml

# # HAT BioSR Finetuning
# accelerate launch -m src.train.train -c configs/train/hat_biosr_ft_lessnoise.yaml

# XLSIM BioSR Finetuning
accelerate launch -m src.train.train -c configs/train/xlsim_biosr_ft_lessnoise.yaml