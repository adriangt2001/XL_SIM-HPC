#!/usr/bin/env bash

set -e
set -x

source .venv/bin/activate

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=2

# GT
uv run -m src.train.line_profile_resolution line --image images/resolution_test_basic/idx_00005/hr_target.tif --p1 865,237 --p2 870,243 --mode single-peak --pixel-size 54.17 --unit nm --out images/cross-section/hr_factin.png
uv run -m src.train.line_profile_resolution line --image images/resolution_test_basic/idx_00007/hr_target.tif --p1 592,405 --p2 596,430 --mode single-peak --pixel-size 54.17 --unit nm --out images/cross-section/hr_ccps.png
uv run -m src.train.line_profile_resolution line --image images/resolution_test_basic/idx_00015/hr_target.tif --p1 284,525 --p2 264,538 --mode single-peak --pixel-size 54.17 --unit nm --out images/cross-section/hr_microtubules.png

# Max
uv run -m src.train.line_profile_resolution line --image images/resolution_test_basic/idx_00005/sr_prediction_a.tif --p1 865,237 --p2 870,243 --mode single-peak --pixel-size 54.17 --unit nm --out images/cross-section/basic_factin.png
uv run -m src.train.line_profile_resolution line --image images/resolution_test_basic/idx_00007/sr_prediction_a.tif --p1 592,405 --p2 596,430 --mode single-peak --pixel-size 54.17 --unit nm --out images/cross-section/basic_ccps.png
uv run -m src.train.line_profile_resolution line --image images/resolution_test_basic/idx_00015/sr_prediction_a.tif --p1 284,525 --p2 264,538 --mode single-peak --pixel-size 54.17 --unit nm --out images/cross-section/basic_microtubules.png

# Max+RL
uv run -m src.train.line_profile_resolution line --image images/resolution_test_rl/idx_00005/sr_prediction_a.tif --p1 865,237 --p2 870,243 --mode single-peak --pixel-size 54.17 --unit nm --out images/cross-section/rl_factin.png
uv run -m src.train.line_profile_resolution line --image images/resolution_test_rl/idx_00007/sr_prediction_a.tif --p1 592,405 --p2 596,430 --mode single-peak --pixel-size 54.17 --unit nm --out images/cross-section/rl_ccps.png
uv run -m src.train.line_profile_resolution line --image images/resolution_test_rl/idx_00015/sr_prediction_a.tif --p1 284,525 --p2 264,538 --mode single-peak --pixel-size 54.17 --unit nm --out images/cross-section/rl_microtubules.png

# Swin2SR
uv run -m src.train.line_profile_resolution line --image images/resolution_test_swin2sr/idx_00005/sr_prediction_a.tif --p1 865,237 --p2 870,243 --mode single-peak --pixel-size 54.17 --unit nm --out images/cross-section/swin2sr_factin.png
uv run -m src.train.line_profile_resolution line --image images/resolution_test_swin2sr/idx_00007/sr_prediction_a.tif --p1 592,405 --p2 596,430 --mode single-peak --pixel-size 54.17 --unit nm --out images/cross-section/swin2sr_ccps.png
uv run -m src.train.line_profile_resolution line --image images/resolution_test_swin2sr/idx_00015/sr_prediction_a.tif --p1 284,525 --p2 264,538 --mode single-peak --pixel-size 54.17 --unit nm --out images/cross-section/swin2sr_microtubules.png

# HAT
uv run -m src.train.line_profile_resolution line --image images/resolution_test_hat/idx_00005/sr_prediction_a.tif --p1 865,237 --p2 870,243 --mode single-peak --pixel-size 54.17 --unit nm --out images/cross-section/hat_factin.png
uv run -m src.train.line_profile_resolution line --image images/resolution_test_hat/idx_00007/sr_prediction_a.tif --p1 592,405 --p2 596,430 --mode single-peak --pixel-size 54.17 --unit nm --out images/cross-section/hat_ccps.png
uv run -m src.train.line_profile_resolution line --image images/resolution_test_hat/idx_00015/sr_prediction_a.tif --p1 284,525 --p2 264,538 --mode single-peak --pixel-size 54.17 --unit nm --out images/cross-section/hat_microtubules.png

# GSASR
uv run -m src.train.line_profile_resolution line --image images/resolution_test_gsasr/idx_00005/sr_prediction_a.tif --p1 865,237 --p2 870,243 --mode single-peak --pixel-size 54.17 --unit nm --out images/cross-section/gsasr_factin.png
uv run -m src.train.line_profile_resolution line --image images/resolution_test_gsasr/idx_00007/sr_prediction_a.tif --p1 592,405 --p2 596,430 --mode single-peak --pixel-size 54.17 --unit nm --out images/cross-section/gsasr_ccps.png
uv run -m src.train.line_profile_resolution line --image images/resolution_test_gsasr/idx_00015/sr_prediction_a.tif --p1 284,525 --p2 264,538 --mode single-peak --pixel-size 54.17 --unit nm --out images/cross-section/gsasr_microtubules.png

# XLSIM
uv run -m src.train.line_profile_resolution line --image images/resolution_test_xlsim/idx_00005/sr_prediction_a.tif --p1 865,237 --p2 870,243 --mode single-peak --pixel-size 54.17 --unit nm --out images/cross-section/xlsim_factin.png
uv run -m src.train.line_profile_resolution line --image images/resolution_test_xlsim/idx_00007/sr_prediction_a.tif --p1 592,405 --p2 596,430 --mode single-peak --pixel-size 54.17 --unit nm --out images/cross-section/xlsim_ccps.png
uv run -m src.train.line_profile_resolution line --image images/resolution_test_xlsim/idx_00015/sr_prediction_a.tif --p1 284,525 --p2 264,538 --mode single-peak --pixel-size 54.17 --unit nm --out images/cross-section/xlsim_microtubules.png
