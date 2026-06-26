from pathlib import Path

import torch
import torch.nn.functional as F
from configargparse import Namespace
from torch.nn.functional import l1_loss
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchvision.utils import make_grid
from tqdm import tqdm

import wandb
from src.simulation.microscope import Microscope
from src.simulation.sim_pipeline import ImageNoiseModel, SimulatorPipeline

from .datasets import prepare_data
from .models import get_model
from .parser import parse_arguments


def main(args: Namespace):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, preprocess_fn, postprocess_fn = get_model(args)
    model = model.to(device=device)
    model.eval()
    _, test_loader = prepare_data(args)

    microscope = Microscope(
        resolution=(
            args.first_crop // args.upscale,
            args.first_crop // args.upscale,
        ),
        device=device,
    )
    noise = ImageNoiseModel(device=device)
    simulator = SimulatorPipeline(microscope, noise, device=device)

    psnr_fn = PeakSignalNoiseRatio(data_range=1.0).to(device=device)
    ssim_fn = StructuralSimilarityIndexMeasure(data_range=1.0).to(device=device)

    wandb.init(
        project="XL-SIM",
        name=f"test_{args.model_name}_{'_'.join(Path(args.checkpoint).parts[2:])}".rstrip(
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
                pixel_values=pixel_values, psf=psf, calibs=calibs
            )
            outputs = model(**preprocessed_batch)
            outputs = postprocess_fn(outputs)

            loss = l1_loss(outputs, targets)
            total_loss += loss.item()
            count += 1

            psnr_fn.update(outputs, targets)
            ssim_fn.update(outputs, targets)

            if not logged_batch:
                images = []
                for i, (pred, inp, target) in enumerate(
                    zip(outputs, pixel_values, targets)
                ):
                    if i > 4:
                        break
                    image = make_grid(
                        [
                            target,
                            pred,
                            F.interpolate(
                                inp[None, 12:13],
                                scale_factor=args.upscale,
                                mode="nearest",
                            )[0],
                        ],
                        nrow=2,
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
    args = parse_arguments()
    main(args)
