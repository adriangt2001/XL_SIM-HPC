import torch
from transformers import Swin2SRConfig, Swin2SRForImageSuperResolution


class MultiImageSwin2SR(torch.nn.Module):
    def __init__(self, swin2sr_config: Swin2SRConfig):
        super().__init__()
        # Processing as independent images

        # Late fusion
        self.swin2sr = Swin2SRForImageSuperResolution(swin2sr_config)

    def forward(self, pixel_values: torch.Tensor):
        output = self.swin2sr(pixel_values)
        return output
