import torch

from ..base_model import BaseModel
from .attention import HAT
from .encoder import EDSR


class XLSIM(BaseModel):
    def __init__(self, encoder_kwargs: dict, fusion_kwargs: dict):
        super().__init__()
        self.encoder = EDSR(**encoder_kwargs)
        self.fusion = HAT(**fusion_kwargs)

    def forward(self, pixel_values: torch.Tensor):
        pixel_values = pixel_values.contiguous()
        
        # x.shape: BxNxCxHxW
        pixel_values = self.encoder(pixel_values)  # Returns: BxNxCxHxW
        pixel_values = self.fusion(pixel_values)  # Returns: BxCxSHxSW
        # x = self.decoder(x)  # Returns: BxCxSHxSW
        return pixel_values
