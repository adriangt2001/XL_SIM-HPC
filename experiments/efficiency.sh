#!/usr/bin/env bash

set -e

source .venv/bin/activate

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=6

config_files=(
    "mean.yaml"
    "rl.yaml"
    "jrl.yaml"
    "swin2sr.yaml"
    "hat.yaml"
    "gsasr.yaml"
    "xlsim.yaml"
)

for config in "${config_files[@]}"; do
    echo === RUNNING $config ===
    uv run -m src.test.test_efficiency -c configs/test_efficiency/$config --upscale 2 --out logs/test_efficiency --microscope_config configs/simulator/lessnoise_microscope.yaml
done