#!/usr/bin/env bash

set -e
set -x

source .venv/bin/activate

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=2

# Max+RL Test
uv run -m src.train.test_efficiency -c configs/test/test_biosr_rlmax_nonoise.yaml

# Max Test
uv run -m src.train.test_efficiency -c configs/test/test_biosr_max_nonoise.yaml

# Swin2SR BioSR Test
uv run -m src.train.test_efficiency -c configs/test/test_biosr_swin2sr_nonoise.yaml

# GSASR BioSR Test
uv run -m src.train.test_efficiency -c configs/test/test_biosr_gsasr_nonoise.yaml

# HAT BioSR Test
uv run -m src.train.test_efficiency -c configs/test/test_biosr_hat_nonoise.yaml

# XLSIM BioSR Test
python -m src.train.test_efficiency -c configs/test/test_biosr_xlsim_nonoise.yaml