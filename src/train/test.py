from pathlib import Path

import torch
import torch.nn.functional as F
from configargparse import Namespace
from torch.nn.functional import l1_loss
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchvision.utils import make_grid
from tqdm import tqdm

import wandb
from src.models import get_model
from src.simulation.sim_pipeline import SimulatorPipeline

from .datasets import prepare_data
from .parser import parse_arguments_test


def main(args: Namespace):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Main model
    model, preprocess_fn, postprocess_fn = get_model(
        args.main_model_name, args.main_model_config, args.checkpoint
    )
    model = model.to(device=device)
    model.eval()

    # Comparison
    comparison_models = []
    for model_name, model_config, model_checkpoint in zip(args.comparison_model_names, args.comparison_model_configs, args.comparison_checkpoints):
        comparison_models.append(get_model(model_name, model_config, model_checkpoint))
    
    # Dataset and Simulator
    _, test_loader = prepare_data(args)

    simulator = SimulatorPipeline.from_file(
        args.microscope_config, args.noise_config
    ).to(device=device)
    psnr_fn = PeakSignalNoiseRatio(data_range=1.0).to(device=device)
    ssim_fn = StructuralSimilarityIndexMeasure(data_range=1.0).to(device=device)

    wandb.init(
        project="XL-SIM",
        name=f"test_{args.main_model_name}_{'_'.join(Path(args.checkpoint).parts[2:])}".rstrip(
            "_"
        ),
    )

    total_loss = 0.0
    count = 0

    logged_batch = False

    with torch.inference_mode():
        for batch in tqdm(test_loader, desc="Test progress"):
            targets = batch["hr"].to(device=device)
            pixel_values, calibs = simulator(targets)
            psf = simulator.microscope.psf_em
            preprocessed_batch = preprocess_fn(
                pixel_values=pixel_values, psf=psf, calibs=calibs, upscale=args.upscale
            )
            outputs = model(**preprocessed_batch)
            outputs = postprocess_fn(outputs)

            loss = l1_loss(outputs, targets)
            total_loss += loss.item()
            count += 1

            psnr_fn.update(outputs, targets)
            ssim_fn.update(outputs, targets)

            if not logged_batch:
                comparison_samples = []
                for comp_model, comp_pre_fn, comp_post_fn in comparison_models:
                    comp_preprocessed = comp_pre_fn(pixel_values=pixel_values, psf=psf, calibs=calibs)
                    comp_out = comp_model(**comp_preprocessed)
                    comparison_samples.append(comp_post_fn(comp_out, target=targets))
                comparison_samples = torch.stack(comparison_samples, dim=1)
                    
                images = []
                for i, (pred, comp_preds, inp, target) in enumerate(
                    zip(outputs, comparison_samples, pixel_values, targets)
                ):
                    if i > 4:
                        break
                    nrow = 2 + comp_preds.shape[0]
                    image = make_grid(
                        [
                            target,
                            pred,
                            *comp_preds,
                            F.interpolate(
                                inp[None, 12:13],
                                scale_factor=args.upscale,
                                mode="nearest",
                            )[0],
                            torch.abs(pred - target),
                            *[torch.abs(comp_pred - target) for comp_pred in comp_preds]
                        ],
                        nrow=nrow,
                    )
                    images.append(
                        wandb.Image(
                            image.clip(0, 1),
                            caption=f"Sample {i}:\n Top left: Target | Top right: Prediction\n Bottom left: Input C12",
                        )
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
