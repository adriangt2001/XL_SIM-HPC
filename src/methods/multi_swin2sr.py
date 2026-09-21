import torch
from transformers import Swin2SRConfig, Swin2SRForImageSuperResolution

from .base_model import BaseModel


class MultiImageSwin2SR(BaseModel):
    def __init__(
        self,
        input_channels: int,
        hidden_dims: int,
        kernel_size: int,
        swin2sr_cfg: dict,
    ):
        super().__init__()
        
        # Processing as independent images (Look for better architectures, perhaps looking at some CNN methods and updating to make them more similar to ConvNext stile)
        self.frame_conv = torch.nn.Sequential(
            torch.nn.Conv2d(input_channels, hidden_dims, kernel_size),
            torch.nn.GELU(),
            torch.nn.Conv2d(hidden_dims, swin2sr_cfg["num_channels"]),
            torch.nn.GELU(),
        )

        # Late fusion
        self.swin2sr = Swin2SRForImageSuperResolution(
            Swin2SRConfig.from_dict(swin2sr_cfg)
        )

    def forward(self, pixel_values: torch.Tensor):
        frames = []
        for idx in pixel_values.shape[1]:
            x = self.frame_conv(pixel_values[:, idx : idx + 1])
            frames.append(x)

        frames = torch.stack(frames, dim=1)
        x = torch.mean(frames, dim=1)

        output = self.swin2sr(x)
        return output
