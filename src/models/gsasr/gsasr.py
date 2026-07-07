import json

import torch

from .fea2gs import Fea2GS
from .gaussian_splatting import generate_2D_gaussian_splatting_step
from .swinir import SwinIRNOUP


class GSASR(torch.nn.Module):
    @classmethod
    def from_json_file(cls, filename: str):
        with open(filename, mode="r") as f:
            config = json.load(f)
        return cls(**config)

    def __init__(
        self,
        encoder_kwargs: dict,
        decoder_kwargs: dict,
        rasterization_kwargs: dict
    ):
        super().__init__()
        self.encoder = SwinIRNOUP(**encoder_kwargs)
        self.decoder = Fea2GS(**decoder_kwargs)
        self.rasterization_kwargs = rasterization_kwargs
        self.rasterization = generate_2D_gaussian_splatting_step

    def forward(self, pixel_values: torch.Tensor, upscale: float):
        B, C, H, W = pixel_values.shape

        target_h = round(H * upscale)
        target_w = round(W * upscale)
        upscale_tensor = torch.ones(B, device=pixel_values.device) * upscale

        features = self.encoder(pixel_values)
        gaussian_params_batch = self.decoder(features, upscale_tensor)

        scale_modify = 1.0 / upscale_tensor

        output_batch = []
        for gaussian_params in gaussian_params_batch:
            output = generate_2D_gaussian_splatting_step(
                sr_size=(target_h, target_w),
                gs_parameters=gaussian_params,
                scale=upscale_tensor,
                scale_modify=scale_modify,
                **self.rasterization_kwargs)
            output_batch.append(output)
        output = torch.stack(output_batch, dim=0)
        return output
