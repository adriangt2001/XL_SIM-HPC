import math

import torch

from ..base_model import BaseModel
from .fea2gs import Fea2GS
from .gaussian_splatting import generate_2D_gaussian_splatting_step
from .swinir import SwinIRNOUP


class GSASR(BaseModel):
    def __init__(
        self, encoder_kwargs: dict, decoder_kwargs: dict, rasterization_kwargs: dict
    ):
        super().__init__()
        self.encoder = SwinIRNOUP(**encoder_kwargs)
        self.decoder = Fea2GS(**decoder_kwargs)
        self.rasterization_kwargs = rasterization_kwargs
        self.rasterization = generate_2D_gaussian_splatting_step
        self.window_size_lcm = math.lcm(encoder_kwargs["window_size"], decoder_kwargs["window_size"])

    def forward(self, pixel_values: torch.Tensor, upscale: float):
        # Ensure pixel_values is properly padded
        B, C, orig_H, orig_W = pixel_values.shape
        pad_h = (self.window_size_lcm - orig_H % self.window_size_lcm) % self.window_size_lcm
        pad_w = (self.window_size_lcm - orig_W % self.window_size_lcm) % self.window_size_lcm

        x = torch.nn.functional.pad(pixel_values, (0, pad_w, 0, pad_h))
        B, C, H, W = x.shape

        target_h = round(H * upscale)
        target_w = round(W * upscale)
        upscale_tensor = torch.ones(B, device=x.device) * upscale

        features = self.encoder(x)
        gaussian_params_batch = self.decoder(features, upscale_tensor)

        scale_modify = 1.0 / upscale_tensor

        output_batch = []
        for gaussian_params in gaussian_params_batch:
            output = generate_2D_gaussian_splatting_step(
                sr_size=(target_h, target_w),
                gs_parameters=gaussian_params,
                scale=upscale_tensor,
                scale_modify=scale_modify,
                **self.rasterization_kwargs,
            )
            output_batch.append(output)
        output = torch.stack(output_batch, dim=0)

        target_orig_h = round(orig_H * upscale)
        target_orig_w = round(orig_W * upscale)
        output = output[..., :target_orig_h, :target_orig_w]
        
        return output
