#!/usr/bin/env bash

set -e

source .venv/bin/activate

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=6

config_files=(
    "mean_biosr.yaml"
    "rl_biosr.yaml"
    "jrl_biosr.yaml"
    "swin2sr_biosr.yaml"
    "hat_biosr.yaml"
    "gsasr_biosr.yaml"
    "xlsim_biosr.yaml"
)

for config in "${config_files[@]}"; do
    echo === RUNNING $config ===
    uv run -m src.test.test -c configs/test/$config --upscale 2 --out logs/test --microscope_config configs/simulator/lessnoise_microscope.yaml
done