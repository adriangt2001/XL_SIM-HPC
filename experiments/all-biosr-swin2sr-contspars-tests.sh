#!/usr/bin/env bash

set -e
set -x

source .venv/bin/activate

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=2

## Continuity

# Swin2SR BioSR Test
uv run -m src.train.inference_sample -c configs/test/test_biosr_swin2sr_lessnoise.yaml --checkpoint runs/Swin2SR/run_20260825_115356/best_psnr --index 19 --output images/biosr_inference_contspars/continuity10.png

# Swin2SR BioSR Test
uv run -m src.train.inference_sample -c configs/test/test_biosr_swin2sr_lessnoise.yaml --checkpoint runs/Swin2SR/run_20260825_170545/best_psnr --index 19 --output images/biosr_inference_contspars/continuity09.png

# Swin2SR BioSR Test
uv run -m src.train.inference_sample -c configs/test/test_biosr_swin2sr_lessnoise.yaml --checkpoint runs/Swin2SR/run_20260825_223923/best_psnr --index 19 --output images/biosr_inference_contspars/continuity07.png

# Swin2SR BioSR Test
uv run -m src.train.inference_sample -c configs/test/test_biosr_swin2sr_lessnoise.yaml --checkpoint runs/Swin2SR/run_20260826_041625/best_psnr --index 19 --output images/biosr_inference_contspars/continuity05.png

# Swin2SR BioSR Test
uv run -m src.train.inference_sample -c configs/test/test_biosr_swin2sr_lessnoise.yaml --checkpoint runs/Swin2SR/run_20260826_101511/best_psnr --index 19 --output images/biosr_inference_contspars/continuity02.png


## Sparsity
# Swin2SR BioSR Test
uv run -m src.train.inference_sample -c configs/test/test_biosr_swin2sr_lessnoise.yaml --checkpoint runs/Swin2SR/run_20260826_174034/best_psnr --index 19 --output images/biosr_inference_contspars/sparsity10.png

# Swin2SR BioSR Test
uv run -m src.train.inference_sample -c configs/test/test_biosr_swin2sr_lessnoise.yaml --checkpoint runs/Swin2SR/run_20260827_153650/best_psnr --index 19 --output images/biosr_inference_contspars/sparsity09.png

# Swin2SR BioSR Test
uv run -m src.train.inference_sample -c configs/test/test_biosr_swin2sr_lessnoise.yaml --checkpoint runs/Swin2SR/run_20260827_223039/best_psnr --index 19 --output images/biosr_inference_contspars/sparsity07.png

# Swin2SR BioSR Test
uv run -m src.train.inference_sample -c configs/test/test_biosr_swin2sr_lessnoise.yaml --checkpoint runs/Swin2SR/run_20260828_142806/best_psnr --index 19 --output images/biosr_inference_contspars/sparsity05.png

# Swin2SR BioSR Test
uv run -m src.train.inference_sample -c configs/test/test_biosr_swin2sr_lessnoise.yaml --checkpoint runs/Swin2SR/run_20260829_023734/best_psnr --index 19 --output images/biosr_inference_contspars/sparsity02.png

## Combo Wombo
# Swin2SR BioSR Test
uv run -m src.train.inference_sample -c configs/test/test_biosr_swin2sr_lessnoise.yaml --checkpoint runs/Swin2SR/run_20260903_031020/best_psnr --index 19 --output images/biosr_inference_contspars/contspars.png

# Swin2SR BioSR Test
uv run -m src.train.inference_sample -c configs/test/test_biosr_swin2sr_lessnoise.yaml --checkpoint runs/Swin2SR/best_biosr/best_psnr --index 19 --output images/biosr_inference_contspars/best_biosr.png
