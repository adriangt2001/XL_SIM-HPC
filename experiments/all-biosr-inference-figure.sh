#!/usr/bin/env bash

set -e
set -x

source .venv/bin/activate

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=2

# Max
uv run -m src.train.inference --index 5 --output images/biosr_inference -c configs/test/test_biosr_max_lessnoise.yaml
uv run -m src.train.inference --index 7 --output images/biosr_inference -c configs/test/test_biosr_max_lessnoise.yaml
uv run -m src.train.inference --index 15 --output images/biosr_inference -c configs/test/test_biosr_max_lessnoise.yaml

# Max+RL
uv run -m src.train.inference --index 5 --output images/biosr_inference -c configs/test/test_biosr_rlmax_lessnoise.yaml
uv run -m src.train.inference --index 7 --output images/biosr_inference -c configs/test/test_biosr_rlmax_lessnoise.yaml
uv run -m src.train.inference --index 15 --output images/biosr_inference -c configs/test/test_biosr_rlmax_lessnoise.yaml

# Swin2SR
uv run -m src.train.inference --index 5 --output images/biosr_inference -c configs/test/test_biosr_swin2sr_lessnoise.yaml
uv run -m src.train.inference --index 7 --output images/biosr_inference -c configs/test/test_biosr_swin2sr_lessnoise.yaml
uv run -m src.train.inference --index 15 --output images/biosr_inference -c configs/test/test_biosr_swin2sr_lessnoise.yaml

# HAT
uv run -m src.train.inference --index 5 --output images/biosr_inference -c configs/test/test_biosr_hat_lessnoise.yaml
uv run -m src.train.inference --index 7 --output images/biosr_inference -c configs/test/test_biosr_hat_lessnoise.yaml
uv run -m src.train.inference --index 15 --output images/biosr_inference -c configs/test/test_biosr_hat_lessnoise.yaml

# GSASR
uv run -m src.train.inference --index 5 --output images/biosr_inference -c configs/test/test_biosr_gsasr_lessnoise.yaml
uv run -m src.train.inference --index 7 --output images/biosr_inference -c configs/test/test_biosr_gsasr_lessnoise.yaml
uv run -m src.train.inference --index 15 --output images/biosr_inference -c configs/test/test_biosr_gsasr_lessnoise.yaml

# XLSIM
uv run -m src.train.inference --index 5 --output images/biosr_inference -c configs/test/test_biosr_xlsim_lessnoise.yaml
uv run -m src.train.inference --index 7 --output images/biosr_inference -c configs/test/test_biosr_xlsim_lessnoise.yaml
uv run -m src.train.inference --index 15 --output images/biosr_inference -c configs/test/test_biosr_xlsim_lessnoise.yaml
