import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from configargparse import Namespace
from PIL import Image

from .data.datasets import get_data
from .data.preprocessing import crop_tensor
from .methods import get_model
from .microscope.sim_pipeline import SimulatorPipeline
from .parser import parse_arguments_test


def parse_extra_args(argv):
    """Pull custom flags out of argv before handing the rest to parse_arguments_test()."""
    extra_parser = argparse.ArgumentParser(add_help=False)

    extra_parser.add_argument(
        "--index", type=int, default=0, help="Index of dataset item for inference"
    )
    extra_parser.add_argument(
        "--output", type=str, default="output", help="Directory path to save images"
    )

    extra_args, remaining_argv = extra_parser.parse_known_args(argv)
    return extra_args, remaining_argv


def save_grayscale_image(tensor: torch.Tensor, save_path: Path):
    """Saves a 2D image tensor directly to disk as a single-channel (grayscale) image."""
    img_np = tensor.squeeze().detach().cpu().numpy()
    img_np = np.clip(img_np, 0.0, 1.0)
    img_uint8 = (img_np * 255.0).astype(np.uint8)
    Image.fromarray(img_uint8, mode="L").save(save_path)


def main(args: Namespace, extra_args: Namespace):
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

    # Dataset and Simulator
    _, _, test_loader = get_data(
        args.dataset,
        args.test_size,
        args.first_crop,
        args.split,
        args.batch_size,
        args.num_workers,
    )

    test_dataset = test_loader.dataset
    if extra_args.index < 0 or extra_args.index >= len(test_dataset):
        raise IndexError(
            f"Index {extra_args.index} out of bounds for dataset of length {len(test_dataset)}."
        )

    simulator = SimulatorPipeline.from_file(
        args.microscope_config, args.noise_config
    ).to(device=device)

    # Fetch specific sample by index
    sample = test_dataset[extra_args.index]
    targets = sample["hr"].unsqueeze(0).to(device=device)
    target_padding: torch.Tensor = sample["padding"].unsqueeze(0).to(device=device)

    # Resolve Dataset and Class metadata strings
    dataset_name = Path(args.dataset).stem
    class_val = sample.get("class", "noclass")
    if (
        isinstance(class_val, int)
        and hasattr(test_dataset, "features")
        and "class" in test_dataset.features
    ):
        class_name = test_dataset.features["class"].int2str(class_val)
    else:
        class_name = str(class_val)

    with torch.inference_mode():
        pixel_values, calibs = simulator(targets)
        pixel_values, targets, _ = crop_tensor(
            pixel_values,
            args.second_crop,
            pair_image=targets,
            pair_scale_factor=args.upscale,
            offset=target_padding.max(dim=0).values // (2 * args.upscale),
            mode="center",
        )
        psf = simulator.microscope.psf_em
        preprocessed_batch = preprocess_fn(
            pixel_values=pixel_values, psf=psf, calibs=calibs, upscale=args.upscale
        )

        outputs = model(**preprocessed_batch)
        outputs = postprocess_fn(outputs)

        # Create output directory
        output_dir = Path(extra_args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save model inference output image (1-channel grayscale)
        model_filename = (
            f"{dataset_name}_{class_name}_{args.main_model_name}_{extra_args.index}.png".lower()
        )
        model_img_path = output_dir / model_filename
        save_grayscale_image(outputs, model_img_path)
        print(f"Saved model inference result to: {model_img_path}")

        # Save HR ground-truth target image (1-channel grayscale)
        hr_filename = f"{dataset_name}_{class_name}_HR_{extra_args.index}.png".lower()
        hr_img_path = output_dir / hr_filename
        save_grayscale_image(targets, hr_img_path)
        print(f"Saved HR ground-truth image to: {hr_img_path}")


if "__main__" == __name__:
    extra_args, remaining_argv = parse_extra_args(sys.argv[1:])
    sys.argv = [sys.argv[0]] + remaining_argv
    args = parse_arguments_test()
    main(args, extra_args)