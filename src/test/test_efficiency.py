from pathlib import Path

import torch
import torch.utils.benchmark as torch_benchmark
from configargparse import Namespace
from torch.utils.flop_counter import FlopCounterMode

import wandb
from src.data.preprocessing import crop_tensor
from src.methods import get_model
from src.microscope.microscope import Microscope

# The model's input (after the simulator + crop + preprocessing) is expected
# to be a 25-channel tensor.
HR_CHANNELS = 25


def _as_hw(size):
    """Accept either an int (square crop) or a (h, w) pair, as args.first_crop
    is used elsewhere in this codebase."""
    if isinstance(size, (tuple, list)):
        return int(size[0]), int(size[1])
    return int(size), int(size)


def main(args: Namespace):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Main model
    model, preprocess_fn, postprocess_fn = get_model(
        args.main_model_name,
        args.main_model_config,
        args.checkpoint,
        args.lora,
        lora=args.lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=args.lora_target_modules,
        lora_bias=args.lora_bias,
    )
    model = model.to(device=device)
    if args.lora:
        model = model.merge_and_unload()
    model.eval()

    # Simulator only: we keep it so the model receives correctly formatted
    # input (simulated microscope data), but we do not use the real dataset
    # at all, and the simulator's pass is never included in the metrics.
    simulator = Microscope.from_file(
        args.microscope_config
    ).to(device=device)

    run_name = f"test_{args.main_model_name}_efficiency_{'_'.join(Path(args.checkpoint).parts[2:])}".rstrip(
        "_"
    )

    wandb.init(
        project="XL-SIM",
        name=run_name,
    )
    total_parameters = sum(p.numel() for p in model.parameters())

    # --- Build one synthetic batch --------------------------------------
    # Replaces the dataset entirely. Shaped like the "hr" tensor the
    # simulator normally receives from the dataloader: (B, 25, H, W).
    # Padding is set to zero (no padding), since synthetic data has none.
    h, w = _as_hw(args.first_crop)
    with torch.inference_mode():
        targets = torch.randn(
            args.batch_size, HR_CHANNELS, h, w, device=device, dtype=torch.float32
        )

        # --- Simulator pass: NOT included in the metrics -----------------
        pixel_values, calibs = simulator(targets)
        pixel_values, targets, _ = crop_tensor(
            pixel_values,
            args.second_crop,
            pair_image=targets,
            pair_scale_factor=args.upscale,
            mode="center",
        )
        psf = simulator.microscope.psf_em
        preprocessed_batch = preprocess_fn(
            pixel_values=pixel_values, psf=psf, calibs=calibs, upscale=args.upscale
        )

    def run_model():
        with torch.inference_mode():
            return model(**preprocessed_batch)

    # --- FLOPs -------------------------------------------------------------
    # PyTorch's built-in FlopCounterMode hooks into every aten op and tallies
    # FLOPs analytically from a single forward pass. Doubles as a warmup call.
    with FlopCounterMode(display=False) as flop_counter:
        run_model()
    flops = flop_counter.get_total_flops()

    # --- Inference time ------------------------------------------------
    # torch.utils.benchmark.Timer handles CUDA synchronization and adaptive
    # repeat counts for us, so there's no need for a manual timing loop or
    # manual torch.cuda.Event bookkeeping.
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

    timer = torch_benchmark.Timer(
        stmt="run_model()",
        globals={"run_model": run_model},
        num_threads=torch.get_num_threads(),
        label="inference",
        description=args.main_model_name,
    )
    measurement = timer.timeit(50)
    avg_inference_time = measurement.mean  # seconds

    # --- Peak VRAM -----------------------------------------------------
    peak_vram = (
        torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    )

    metrics = {
        "efficiency/total_parameters": total_parameters,
        "efficiency/vram": peak_vram,
        "efficiency/inference_time": avg_inference_time,
        "efficiency/flops": flops if flops is not None else float("nan"),
    }

    wandb.log(metrics)

    print()
    print("==== Test Results ====")
    for k, v in metrics.items():
        print(f"{k}: {v:.6f}")

    wandb.finish()


if "__main__" == __name__:
    from configargparse import ArgumentParser
    parser = ArgumentParser()

    # Add arguments

    args = parser.parse_args()
    main(args)