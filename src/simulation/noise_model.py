import deepinv as dinv
import torch
import yaml


class ImageNoiseModel(torch.nn.Module):
    @classmethod
    def from_file(cls, filename: str):
        with open(filename, mode="r") as f:
            config = yaml.safe_load(f)
        return cls(**config)

    def __init__(
        self,
        inpainting_noise: float = 0.8,
        gaussian_noise: float = 0.1,
    ):
        super().__init__()
        self.inpainting_noise = inpainting_noise
        self.gaussian_noise = gaussian_noise
        self.requires_grad_(requires_grad=False)

    def forward(self, img: torch.Tensor):
        physics = dinv.physics.Inpainting(
            img.shape[-3:], self.inpainting_noise, device=img.device
        )
        physics.noise_model = dinv.physics.GaussianNoise(self.gaussian_noise)
        output_img = physics(img.to(device=img.device))
        # physics2 = dinv.physics.PoissonNoise(0.1)
        # output_img = physics2(output_img)
        return output_img
