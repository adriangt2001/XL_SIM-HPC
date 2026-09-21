from typing import Literal

import torch
import torch.nn.functional as F
from skimage.restoration import richardson_lucy

from .base_model import BaseModel


class RichardsonLucy(BaseModel):
    def __init__(
        self,
        upscale: int,
        op: Literal["max", "sum", "mean"] = "max",
        mode: str = "bicubic",
        align_corners: bool = False,
        antialias: bool = True,
    ):
        super().__init__()
        self.upscale = upscale
        self.op = op
        self.mode = mode
        self.align_corners = align_corners
        self.antialias = antialias

    def preaggregate(self, pixel_values: torch.Tensor, dim: int):
        if "max" == self.op:
            aggregated_img = torch.max(pixel_values, dim=dim).values
        elif "mean" == self.op:
            aggregated_img = torch.mean(pixel_values, dim=dim)
        elif "sum" == self.op:
            aggregated_img = torch.sum(pixel_values, dim=dim)
        return aggregated_img

    def forward(self, pixel_values: torch.Tensor, psf: torch.Tensor):
        device = pixel_values.device
        img = self.preaggregate(pixel_values, 1)[:, None, ...]
        img = F.interpolate(
            img,
            scale_factor=self.upscale,
            mode=self.mode,
            align_corners=self.align_corners,
            antialias=self.antialias,
        )
        B, C, H, W = img.shape
        psf = psf[None, None, ...].expand((B, C, -1, -1))
        output = []
        for idx in range(len(img)):
            pred = richardson_lucy(
                img[idx].detach().cpu().numpy(), psf[idx].detach().cpu().numpy()
            )
            pred = torch.from_numpy(pred)
            output.append(pred)
        output = torch.stack(output).to(device=device)
        return output
