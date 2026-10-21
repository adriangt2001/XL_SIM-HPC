#!/usr/bin/env bash

set -e
set -x

source .venv/bin/activate

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=2

# Max+RL Test
uv run -m src.train.test -c configs/test/test_biosr_rlmax_lessnoise.yaml

# Max Test
uv run -m src.train.test -c configs/test/test_biosr_max_lessnoise.yaml

# # Swin2SR BioSR Test
# uv run -m src.train.test -c configs/test/test_biosr_swin2sr_lessnoise.yaml

# # GSASR BioSR Test
# uv run -m src.train.test -c configs/test/test_biosr_gsasr_lessnoise.yaml

# # HAT BioSR Test
# uv run -m src.train.test -c configs/test/test_biosr_hat_lessnoise.yaml

# # XLSIM BioSR Test
# python -m src.train.test -c configs/test/test_biosr_xlsim_lessnoise.yaml