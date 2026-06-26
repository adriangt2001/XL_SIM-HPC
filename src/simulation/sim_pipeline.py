import inspect
from typing import Self

import torch

from src.utils.datasets import get_data

from .microscope import Microscope
from .noise_model import ImageNoiseModel


class SimulatorPipeline(torch.nn.Module):
    @classmethod
    def from_file(cls, microscope_filename: str, noise_filename: str):
        microscope = Microscope.from_file(microscope_filename)
        noise = ImageNoiseModel.from_file(noise_filename)
        return cls(microscope, noise)

    @classmethod
    def from_params(cls, **kwargs):
        microscope = Microscope(
            **{
                k: v
                for k, v in kwargs.items()
                if k in inspect.signature(Microscope.__init__).parameters
            }
        )
        noise = ImageNoiseModel(
            **{
                k: v
                for k, v in kwargs.items()
                if k in inspect.signature(ImageNoiseModel.__init__).parameters
            }
        )
        return cls(microscope, noise)

    def __init__(self, microscope: Microscope = None, noise: ImageNoiseModel = None):
        super().__init__()
        self.microscope = microscope if microscope is not None else Microscope()
        self.noise = noise if noise is not None else ImageNoiseModel()
        self.requires_grad_(requires_grad=False)

    @torch.no_grad()
    def forward(self, image: torch.Tensor):
        # Expected shape: B x D x H x W
        assert image.ndim == 4

        # Dinv expects explicit channel dimension (B x C x H x W)
        noisy_img = torch.zeros_like(image, dtype=torch.float32)
        for depth_idx in range(image.shape[1]):
            noisy_img[:, depth_idx : depth_idx + 1, ...] = self.noise(
                image[:, depth_idx : depth_idx + 1, ...]
            )

        # Microscope expects channel dimension (B x D x H x W)
        output, calibs = self.microscope(noisy_img)
        return output, calibs

    def change_microscope(self, microscope: Microscope) -> Self:
        return SimulatorPipeline(microscope, self.noise)


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
