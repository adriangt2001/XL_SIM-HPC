from pathlib import Path

import cv2
import numpy as np
import torch
from configargparse import ArgumentParser, Namespace
from tqdm import tqdm

from .methods import get_model
from .microscope.microscope import Microscope
from .utils import load_image_torch, main_logger


def main(args: Namespace):
    logger = main_logger()

    assert args.image.exists() and args.model_config.exists()
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

    images_paths: list[Path] = []
    if args.image.is_dir():
        images_paths = list(args.image.iterdir())
    else:
        images_paths.append(args.image)

    args.out.mkdir(exist_ok=True)
    output_model_folder: Path = args.out / args.model_name
    output_model_folder.mkdir(exist_ok=True)

    with torch.inference_mode():
        for file in tqdm(images_paths, desc="Inference...", unit="Images"):
            image = load_image_torch(file).to(device=device)
            image = image[None, None, ...]

            pixel_values, calibs = microscope(image)
            psf = microscope.psf_em
            preprocessed_batch = preprocess_fn(pixel_values=pixel_values, psf=psf, calibs=calibs, upscale=args.upscale)

            outputs = model(**preprocessed_batch)
            outputs: torch.Tensor = postprocess_fn(outputs)

            output_file = (output_model_folder / file.stem).with_suffix(".png")

            output_image = (outputs.cpu().numpy().squeeze() * 255).astype(np.uint8)
            cv2.imwrite(str(output_file), output_image)


if "__main__" == __name__:
    parser = ArgumentParser()

    parser.add_argument("-c", "--config", is_config_file=True, help="Path to config file")

    parser.add_argument("--image", type=Path, required=True, help="Image or folder to process")
    parser.add_argument("--out", type=Path, required=True, help="Output folder")
    parser.add_argument("--upscale", type=int, required=True, help="Upsampling factor")

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

    # Microscope configuration
    parser.add_argument("--microscope_config", type=Path, required=True, help="Microscope configuration")

    args = parser.parse_args()
    main(args)