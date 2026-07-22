from pathlib import Path

import torch
import torch.nn.functional as F
from configargparse import Namespace
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchvision.utils import make_grid
from tqdm import tqdm

import wandb
from src.models import get_model
from src.simulation.sim_pipeline import SimulatorPipeline
from src.utils.preprocessing import crop_tensor
from src.utils.visualization import plot_sr_comparison

from .datasets import get_data
from .parser import parse_arguments_test


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
    model.eval()

    # Comparison
    comparison_models = []
    for model_name, model_config, model_checkpoint in zip(
        args.comparison_model_names,
        args.comparison_model_configs,
        args.comparison_checkpoints,
    ):
        comparison_models.append(get_model(model_name, model_config, model_checkpoint))

    # Dataset and Simulator
    _, _, test_loader = get_data(
        args.dataset,
        args.test_size,
        args.first_crop,
        args.split,
        args.batch_size,
        args.num_workers,
    )

    simulator = SimulatorPipeline.from_file(
        args.microscope_config, args.noise_config
    ).to(device=device)
    psnr_fn = PeakSignalNoiseRatio(data_range=1.0).to(device=device)
    ssim_fn = StructuralSimilarityIndexMeasure(data_range=1.0).to(device=device)

    run_name = f"test_{args.main_model_name}_{'_'.join(Path(args.checkpoint).parts[2:])}".rstrip(
        "_"
    )

    wandb.init(
        project="XL-SIM",
        name=run_name,
    )

    logged_batch = False

    with torch.inference_mode():
        for batch in tqdm(test_loader, desc="Test progress"):
            targets = batch["hr"].to(device=device)
            target_padding: torch.Tensor = batch["padding"]

            pixel_values, calibs = simulator(targets)
            pixel_values, targets, _ = crop_tensor(
                pixel_values,
                args.second_crop,
                pair_image=targets,
                pair_scale_factor=args.upscale,
                offset=target_padding.max(dim=0).values // (2 * args.upscale),
                mode=False,
            )
            psf = simulator.microscope.psf_em
            preprocessed_batch = preprocess_fn(
                pixel_values=pixel_values, psf=psf, calibs=calibs, upscale=args.upscale
            )

            outputs = model(**preprocessed_batch)
            outputs = postprocess_fn(outputs)

            psnr_fn.update(outputs, targets)
            ssim_fn.update(outputs, targets)

            if not logged_batch:
                comparison_samples = []
                for comp_model, comp_pre_fn, comp_post_fn in comparison_models:
                    comp_preprocessed = comp_pre_fn(
                        pixel_values=pixel_values, psf=psf, calibs=calibs
                    )
                    comp_out = comp_model(**comp_preprocessed)
                    comparison_samples.append(
                        (comp_post_fn(comp_out, target=targets), comp_model._get_name())
                    )

                images = []
                for i, (pred, inp, target) in enumerate(
                    zip(outputs, pixel_values, targets)
                ):
                    if i > 4:
                        break
                    nrow = 2 + len(comparison_samples)
                    tiles = [
                        target,
                        pred,
                        *[img[i] for img, _ in comparison_samples],
                        F.interpolate(
                            inp[None, 12:13], scale_factor=args.upscale, mode="nearest"
                        )[0],
                        torch.abs(pred - target),
                        *[torch.abs(img[i] - target) for img, _ in comparison_samples],
                    ]

                    image = make_grid(tiles, nrow=nrow)
                    images.append(
                        wandb.Image(
                            image.clip(0, 1),
                            caption=f"Sample {i}:\n Top left: Target | Top right: Prediction\n Bottom left: Input C12",
                        )
                    )

                    plot_sr_comparison(
                        target,
                        inp[12:13],
                        [(img[i], name) for img, name in comparison_samples]
                        + [(pred, args.main_model_name)],
                        save_path=f"logs/{run_name}.png",
                    )
                wandb.log({"test/images": images})

                logged_batch = True

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


if "__main__" == __name__:
    args = parse_arguments_test()
    main(args)
