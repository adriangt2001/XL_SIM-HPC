#!/usr/bin/env bash

set -e
set -x

source .venv/bin/activate

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=2

## More Noise
# Swin2SR BioSR Test
uv run -m src.train.test -c configs/test/test_biosr_swin2sr.yaml --checkpoint runs/Swin2SR/best_biosr_noisy/best_psnr

# Swin2SR BioSR Test
uv run -m src.train.test -c configs/test/test_biosr_swin2sr_nonoise.yaml --checkpoint runs/Swin2SR/best_biosr_noisy/best_psnr

## Less Noise
# Swin2SR BioSR Test
uv run -m src.train.test -c configs/test/test_biosr_swin2sr.yaml

# Swin2SR BioSR Test
uv run -m src.train.test -c configs/test/test_biosr_swin2sr_nonoise.yaml
