import torch
from configargparse import Namespace
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from tqdm import tqdm

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

    psnr_fn.reset()
    ssim_fn.reset()

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Test progress"):
            targets = batch["hr"].to(device=device)

            pixel_values, _ = simulator(targets)
            psf = simulator.microscope.psf_em
            preprocessed_batch = preprocess_fn(pixel_values=pixel_values, psf=psf)
            outputs = model(**preprocessed_batch)
            outputs = postprocess_fn(outputs)

            psnr_fn.update(outputs, targets)
            ssim_fn.update(outputs, targets)

        metrics = {
            "psnr": psnr_fn.compute().item(),
            "ssim": ssim_fn.compute().item(),
        }

        print(metrics)


if "__main__" == __name__:
    args = parse_arguments()
    main(args)
