from typing import Self

import deepinv as dinv
import torch

from src.utils.datasets import get_data

from .microscope import Microscope


class ImageNoiseModel:
    def __init__(
        self, inpainting_noise: float = 0.8, gaussian_noise: float = 0.1, device:str = "cuda", device_id: int = 0
    ):
        self.device = torch.device(device, device_id)
        self.inpainting_noise = inpainting_noise
        self.gaussian_noise = gaussian_noise

    def __call__(self, img: torch.Tensor):
        physics = dinv.physics.Inpainting(
            img.shape[-3:], self.inpainting_noise, device=self.device
        )
        physics.noise_model = dinv.physics.GaussianNoise(self.gaussian_noise)
        output_img = physics(img.to(device=self.device))
        # physics2 = dinv.physics.PoissonNoise(0.1)
        # output_img = physics2(output_img)
        return output_img


class SimulatorPipeline:
    def __init__(self, noise: ImageNoiseModel = None, microscope: Microscope = None, device: str = "cuda", device_id: int = 0):
        self.device = torch.device(device, device_id)
        if microscope is None:
            microscope = Microscope(device=self.device)

        if noise is None:
            noise = ImageNoiseModel(device=self.device)

        self.noise = noise
        self.microscope = microscope

    def __call__(self, image: torch.Tensor):
        # Expected shape: B x D x H x W
        assert image.ndim == 4

        image = image.to(device=self.device)
        
        # Dinv expects explicit channel dimension (B x C x H x W)
        noisy_img = torch.zeros_like(image, dtype=torch.float32)
        for depth_idx in range(image.shape[1]):
            noisy_img[:, depth_idx:depth_idx + 1, ...] = self.noise(image[:, depth_idx:depth_idx + 1, ...])

        # Microscope expects channel dimension (B x D x H x W)
        output, calibs = self.microscope(noisy_img)
        output = output.detach().cpu()
        calibs = calibs.detach().cpu()
        return output, calibs

    def change_microscope(self, microscope: Microscope) -> Self:
        return SimulatorPipeline(self.noise, microscope)


def main(args):
    import inspect
    import os

    import tifffile
    from tqdm import tqdm

    assert args.source_path is not None

    # Process args
    micr_args = {
        k: v
        for k, v in vars(args).items()
        if k in inspect.getfullargspec(Microscope).args
    }
    noise_args = {
        k: v
        for k, v in vars(args).items()
        if k in inspect.getfullargspec(ImageNoiseModel).args
    }
    source_path = args.source_path
    output_path = args.output_path
    batch_size = args.batch_size

    # Instantiate Microscope and Noise
    microscope = Microscope(**micr_args)
    noise = ImageNoiseModel(**noise_args)

    # Instantiate Pipeline
    pipeline = SimulatorPipeline(noise, microscope)

    # Create dataset and dataloader
    dataset, dataloader = get_data(source_path, batch_size=batch_size)

    for idx, images in tqdm(enumerate(dataloader), total=len(dataloader)):
        output, _ = pipeline(images)

        if output_path is not None:
            output = output.detach().cpu().numpy()

            for im in output:
                file_name = f"image_{idx:06d}.tiff"
                tifffile.imwrite(os.path.join(output_path, file_name), im)

    if args.visualize:
        import matplotlib.pyplot as plt

        im = tifffile.imread(os.path.join(output_path, file_name))
        tifffile.imshow(im)
        plt.show()
