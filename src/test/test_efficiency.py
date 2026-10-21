from pathlib import Path

import pandas as pd
import torch
import torch.utils.benchmark as torch_benchmark
from configargparse import Namespace
from torch.utils.flop_counter import FlopCounterMode

import wandb
from src.data.preprocessing import crop_tensor
from src.methods import get_model
from src.microscope.microscope import Microscope
from src.utils import main_logger

units = {"B": 1, "KB": 2**10, "MB": 2**20, "GB": 2**30,
         "": 1, "K": 1e3, "M": 1e6, "G": 1e9, "T": 1e12}

VRAM_UNIT = "GB"
PARAMS_UNIT = "M"
FLOPS_UNIT = "G"


def parse_size(s: str):
    h, w = s.split(",")
    return int(h), int(w)


def main(args: Namespace):
    logger = main_logger()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("Loading model...")
    model, preprocess_fn, postprocess_fn = get_model(
        args.model_name,
        args.model_config,
        args.weights,
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
    logger.info("Model loaded!")

    logger.info("Loading microscope simulator...")
    microscope = Microscope.from_file(args.microscope_config).to(device=device)
    logger.info("Microscope simulator loaded!")

    logger.info("Creating synthetic data...")
    h, w = args.hr_size
    with torch.inference_mode():
        targets = torch.randn(
            args.batch_size, 1, h, w, device=device, dtype=torch.float32
        )

        # --- Simulator pass: NOT included in the metrics -----------------
        pixel_values, calibs = microscope(targets)
        pixel_values, targets, _ = crop_tensor(
            pixel_values,
            h // args.upscale,
            pair_image=targets,
            pair_scale_factor=args.upscale,
            mode="center",
        )
        psf = microscope.psf_em
        preprocessed_batch = preprocess_fn(
            pixel_values=pixel_values, psf=psf, calibs=calibs, upscale=args.upscale
        )
    logger.info("Synthetic data created!")

    logger.info("Setting other stuff...")
    def run_model():
        with torch.inference_mode():
            return model(**preprocessed_batch)

    run_name = f"test_{args.model_name}_efficiency_{'_'.join(Path(args.weights).parts[2:])}".rstrip(
        "_"
    )

    wandb.init(
        project="XL-SIM",
        name=run_name
    )

    args.out.mkdir(exist_ok=True)
    output_csv: Path = args.out / "test_efficiency.csv"

    logger.info("Test ready to start!")

    logger.info("Getting number of parameters...")
    total_parameters = sum(p.numel() for p in model.parameters())
    logger.info("Number of parameters obtained!")

    logger.info("Computing FLOP count...")
    with FlopCounterMode(display=False) as flop_counter:
        run_model()
    flops = flop_counter.get_total_flops()
    logger.info("FLOPs computed!")

    logger.info("Computing inference time...")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

    timer = torch_benchmark.Timer(
        stmt="run_model()",
        globals={"run_model": run_model},
        num_threads=torch.get_num_threads(),
        label="inference",
        description=args.model_name,
    )
    measurement = timer.timeit(50)
    avg_inference_time = measurement.mean
    logger.info("Inference time computed!")

    logger.info("Computing peak VRAM usage...")
    peak_vram = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    logger.info("Peak VRAM usage computed!")

    metrics = {
        "total_parameters": total_parameters / units[PARAMS_UNIT],
        "vram": peak_vram / units[VRAM_UNIT],
        "inference_time": avg_inference_time,
        "flops": flops / units[FLOPS_UNIT] if flops is not None else float("nan"),
    }

    metrics_wandb = {}
    for k, v in metrics.items():
        metrics_wandb["efficiency/" + k] = v

    wandb.log(metrics_wandb)
    wandb.finish()

    if output_csv.exists():
        df = pd.read_csv(output_csv)
    else:
        df = pd.DataFrame(columns=["Method", "Configuration", "Upscale", f"# Params ({PARAMS_UNIT})", f"VRAM ({VRAM_UNIT})", "Time (s)", f"FLOPS ({FLOPS_UNIT})"])

    df.loc[len(df)] = [args.model_name, args.model_config, args.upscale, metrics["total_parameters"], metrics["vram"], metrics["inference_time"], metrics["flops"]]
    df = df.drop_duplicates()
    df.to_csv(output_csv, index=False)

    logger.info(f"Saved results to {output_csv}!")


if "__main__" == __name__:
    from configargparse import ArgumentParser

    parser = ArgumentParser()

    # Config file
    parser.add_argument(
        "-c", "--config", is_config_file=True, help="Path to config file"
    )
    parser.add_argument("--upscale", type=int, required=True, help="Upsampling factor")
    parser.add_argument("--out", type=Path, required=True, help="Folder for output")

    # Model configuration
    parser.add_argument(
        "--model_name", type=str, required=True, help="Name of the model"
    )
    parser.add_argument(
        "--model_config", type=Path, required=True, help="Model configuration"
    )
    parser.add_argument(
        "--weights", type=Path, default=None, help="Model weights folder"
    )
    parser.add_argument(
        "--lora", action="store_true", default=False, help="Whether to use lora or not"
    )
    parser.add_argument(
        "--lora_r", type=int, default=16, help="Rank of the lora matrices"
    )
    parser.add_argument("--lora_alpha", type=int, default=32, help="Alpha of the lora")
    parser.add_argument(
        "--lora_dropout", type=float, default=0.1, help="Dropout rate of the lora"
    )
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        nargs="+",
        default=["all-linear"],
        help="Layers to target with lora",
    )
    parser.add_argument(
        "--lora_bias", type=str, default="none", help="Bias to target with lora"
    )

    # Data
    parser.add_argument(
        "--hr_size", type=parse_size, required=True, help="Size of HR image"
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")

    # Microscope configuration
    parser.add_argument(
        "--microscope_config", type=Path, required=True, help="Microscope configuration"
    )

    args = parser.parse_args()
    main(args)
