#!/usr/bin/env bash

set -e
set -x

source .venv/bin/activate

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=2

## F-Actin
# Max
uv run -m src.train.line_profile_resolution frc --image images/resolution_test_basic/idx_00005/sr_prediction_a.tif --image2 images/resolution_test_basic/idx_00005/sr_prediction_b.tif --pixel-size 54.17 --unit nm --out images/frc/basic_factin.png
uv run -m src.train.line_profile_resolution frc --image images/resolution_test_basic/idx_00007/sr_prediction_a.tif --image2 images/resolution_test_basic/idx_00007/sr_prediction_b.tif --pixel-size 54.17 --unit nm --out images/frc/basic_ccps.png
uv run -m src.train.line_profile_resolution frc --image images/resolution_test_basic/idx_00015/sr_prediction_a.tif --image2 images/resolution_test_basic/idx_00015/sr_prediction_b.tif --pixel-size 54.17 --unit nm --out images/frc/basic_microtubules.png

# Max+RL
uv run -m src.train.line_profile_resolution frc --image images/resolution_test_rl/idx_00005/sr_prediction_a.tif --image2 images/resolution_test_rl/idx_00005/sr_prediction_b.tif --pixel-size 54.17 --unit nm --out images/frc/rl_factin.png
uv run -m src.train.line_profile_resolution frc --image images/resolution_test_rl/idx_00007/sr_prediction_a.tif --image2 images/resolution_test_rl/idx_00007/sr_prediction_b.tif --pixel-size 54.17 --unit nm --out images/frc/rl_ccps.png
uv run -m src.train.line_profile_resolution frc --image images/resolution_test_rl/idx_00015/sr_prediction_a.tif --image2 images/resolution_test_rl/idx_00015/sr_prediction_b.tif --pixel-size 54.17 --unit nm --out images/frc/rl_microtubules.png

# Swin2SR
uv run -m src.train.line_profile_resolution frc --image images/resolution_test_swin2sr/idx_00005/sr_prediction_a.tif --image2 images/resolution_test_swin2sr/idx_00005/sr_prediction_b.tif --pixel-size 54.17 --unit nm --out images/frc/swin2sr_factin.png
uv run -m src.train.line_profile_resolution frc --image images/resolution_test_swin2sr/idx_00007/sr_prediction_a.tif --image2 images/resolution_test_swin2sr/idx_00007/sr_prediction_b.tif --pixel-size 54.17 --unit nm --out images/frc/swin2sr_ccps.png
uv run -m src.train.line_profile_resolution frc --image images/resolution_test_swin2sr/idx_00015/sr_prediction_a.tif --image2 images/resolution_test_swin2sr/idx_00015/sr_prediction_b.tif --pixel-size 54.17 --unit nm --out images/frc/swin2sr_microtubules.png

# HAT
uv run -m src.train.line_profile_resolution frc --image images/resolution_test_hat/idx_00005/sr_prediction_a.tif --image2 images/resolution_test_hat/idx_00005/sr_prediction_b.tif --pixel-size 54.17 --unit nm --out images/frc/hat_factin.png
uv run -m src.train.line_profile_resolution frc --image images/resolution_test_hat/idx_00007/sr_prediction_a.tif --image2 images/resolution_test_hat/idx_00007/sr_prediction_b.tif --pixel-size 54.17 --unit nm --out images/frc/hat_ccps.png
uv run -m src.train.line_profile_resolution frc --image images/resolution_test_hat/idx_00015/sr_prediction_a.tif --image2 images/resolution_test_hat/idx_00015/sr_prediction_b.tif --pixel-size 54.17 --unit nm --out images/frc/hat_microtubules.png

# GSASR
uv run -m src.train.line_profile_resolution frc --image images/resolution_test_gsasr/idx_00005/sr_prediction_a.tif --image2 images/resolution_test_gsasr/idx_00005/sr_prediction_b.tif --pixel-size 54.17 --unit nm --out images/frc/gsasr_factin.png
uv run -m src.train.line_profile_resolution frc --image images/resolution_test_gsasr/idx_00007/sr_prediction_a.tif --image2 images/resolution_test_gsasr/idx_00007/sr_prediction_b.tif --pixel-size 54.17 --unit nm --out images/frc/gsasr_ccps.png
uv run -m src.train.line_profile_resolution frc --image images/resolution_test_gsasr/idx_00015/sr_prediction_a.tif --image2 images/resolution_test_gsasr/idx_00015/sr_prediction_b.tif --pixel-size 54.17 --unit nm --out images/frc/gsasr_microtubules.png

# XLSIM
uv run -m src.train.line_profile_resolution frc --image images/resolution_test_xlsim/idx_00005/sr_prediction_a.tif --image2 images/resolution_test_xlsim/idx_00005/sr_prediction_b.tif --pixel-size 54.17 --unit nm --out images/frc/xlsim_factin.png
uv run -m src.train.line_profile_resolution frc --image images/resolution_test_xlsim/idx_00007/sr_prediction_a.tif --image2 images/resolution_test_xlsim/idx_00007/sr_prediction_b.tif --pixel-size 54.17 --unit nm --out images/frc/xlsim_ccps.png
uv run -m src.train.line_profile_resolution frc --image images/resolution_test_xlsim/idx_00015/sr_prediction_a.tif --image2 images/resolution_test_xlsim/idx_00015/sr_prediction_b.tif --pixel-size 54.17 --unit nm --out images/frc/xlsim_microtubules.png
