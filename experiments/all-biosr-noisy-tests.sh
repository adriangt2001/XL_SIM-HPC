#!/usr/bin/env bash

set -e
set -x

source .venv/bin/activate

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=2

# Swin2SR BioSR Test
uv run -m src.train.test -c configs/test/test_biosr_swin2sr.yaml

# GSASR BioSR Test
uv run -m src.train.test -c configs/test/test_biosr_gsasr.yaml

# HAT BioSR Test
uv run -m src.train.test -c configs/test/test_biosr_hat.yaml

# XLSIM BioSR Test
uv run -m src.train.test -c configs/test/test_biosr_xlsim.yaml