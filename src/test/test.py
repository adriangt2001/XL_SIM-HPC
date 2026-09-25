from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from configargparse import Namespace
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from tqdm import tqdm

import wandb
from src.data.datasets import get_data
from src.data.preprocessing import crop_tensor
from src.methods import get_model
from src.microscope.microscope import Microscope
from src.utils import main_logger, save_image


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

    logger.info("Loading data...")
    _, _, test_loader = get_data(
        args.dataset,
        args.test_size,
        args.first_crop,
        args.split,
        args.batch_size,
        args.num_workers,
    )
    logger.info("Data loaded!")

    logger.info("Loading microscope simulator...")
    microscope = Microscope.from_file(args.microscope_config).to(device=device)
    logger.info("Microscope simulator loaded!")

    logger.info("Setting other stuff...")
    psnr_fn = PeakSignalNoiseRatio(data_range=1.0).to(device=device)
    ssim_fn = StructuralSimilarityIndexMeasure(data_range=1.0).to(device=device)

    run_name = f"test_{args.model_name}_{'_'.join(Path(args.weights).parts[2:])}".rstrip(
        "_"
    )

    wandb.init(
        project="XL-SIM",
        name=run_name,
    )

    args.out.mkdir(exist_ok=True)
    output_csv: Path = args.out / "test.csv"
    output_model: Path = args.out / args.model_name / args.dataset.stem
    output_model.mkdir(exist_ok=True, parents=True)

    logger.info("Test ready to start!")

    with torch.inference_mode():
        for idx, batch in enumerate(tqdm(test_loader, desc="Test progress...")):
            targets = batch["hr"].to(device=device)
            target_padding: torch.Tensor = batch["padding"]

            pixel_values, calibs = microscope(targets)
            pixel_values, targets, _ = crop_tensor(
                pixel_values,
                args.second_crop,
                pair_image=targets,
                pair_scale_factor=args.upscale,
                offset=target_padding.max(dim=0).values // (2 * args.upscale),
                mode="center",
            )
            psf = microscope.psf_em
            preprocessed_batch = preprocess_fn(
                pixel_values=pixel_values, psf=psf, calibs=calibs, upscale=args.upscale
            )

            outputs = model(**preprocessed_batch)
            outputs = postprocess_fn(outputs)

            psnr_fn.update(outputs, targets)
            ssim_fn.update(outputs, targets)

            if args.visual:
                for subidx, (target, input_im, output) in enumerate(zip(targets, pixel_values, outputs)):
                    real_idx = idx * args.batch_size + subidx

                    target_image_path = output_model / f"target{real_idx:05d}.png"
                    target = (target.cpu().numpy() * 255).astype(np.uint8).squeeze()
                    save_image(target_image_path, target)

                    # input_image_path = output_model / f"input{real_idx:05d}.tiff"
                    # input_im = input_im.cpu().numpy()
                    # save_image(input_image_path, input_im)

                    output_image_path = output_model / f"output{real_idx:05d}.png"
                    output = (output.cpu().numpy() * 255).astype(np.uint8).squeeze()
                    save_image(output_image_path, output)

        metrics = {
            "psnr": psnr_fn.compute().item(),
            "ssim": ssim_fn.compute().item(),
        }

        wandb.log(metrics)

        print()
        print("==== Test Results ====")
        for k, v in metrics.items():
            print(f"{k}: {v:.6f}")

        wandb.finish()

        if output_csv.exists():
            df = pd.read_csv(output_csv)
        else:
            df = pd.DataFrame(columns=["Dataset",  "Microscope", "Method", "Configuration", "Upscale", "PSNR", "SSIM"])

        df.loc[len(df)] = [args.dataset.stem, args.microscope_config.stem, args.model_name, args.model_config, args.upscale, metrics["psnr"], metrics["ssim"]]
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
    parser.add_argument("--visual", action="store_true", help="Save visual results")

    # Model configuration
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model")
    parser.add_argument("--model_config", type=Path, required=True, help="Model configuration")
    parser.add_argument("--weights", type=Path, default=None, help="Model weights folder")
    parser.add_argument("--lora", action="store_true", default=False, help="Whether to use lora or not")
    parser.add_argument("--lora_r", type=int, default=16, help="Rank of the lora matrices")
    parser.add_argument("--lora_alpha", type=int, default=32, help="Alpha of the lora")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="Dropout rate of the lora")
    parser.add_argument("--lora_target_modules", type=str, nargs="+", default=["all-linear"], help="Layers to target with lora")
    parser.add_argument("--lora_bias", type=str, default="none", help="Bias to target with lora")

    # Dataset, preprocessing and postprocessing configuration
    parser.add_argument("--dataset", type=Path, required=True, help="Dataset folder")
    parser.add_argument("--split", type=str, default="valid", help="Split name")
    parser.add_argument("--test_size", type=float, default=0.99, help="Validation/test size (1.0 not available, keep at 0.99 to use full dataset)")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="Dataloader num workers")
    parser.add_argument("--first_crop", type=int, default=256, help="Size of LR first cropping")
    parser.add_argument("--second_crop", type=int, default=64, help="Size of LR second cropping")

    # Microscope configuration
    parser.add_argument("--microscope_config", type=Path, required=True, help="Microscope configuration")
    
    args = parser.parse_args()
    main(args)
