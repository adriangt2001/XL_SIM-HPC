from typing import Literal

import torch
import torch.nn.functional as F

from .base_model import BaseModel


class BasicOP(BaseModel):
    def __init__(
        self,
        upscale: int,
        op: Literal["max", "mean", "sum"],
        mode: str = "bicubic",
        align_corners: bool = True,
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

    def forward(self, pixel_values: torch.Tensor):
        img = self.preaggregate(pixel_values, 1)[:, None, ...]
        output = F.interpolate(
            img,
            scale_factor=self.upscale,
            mode=self.mode,
            align_corners=self.align_corners,
            antialias=self.antialias,
        )
        return output
