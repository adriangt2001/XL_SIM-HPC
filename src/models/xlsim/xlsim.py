import torch

from ..base_model import BaseModel
from .attention import HAT
from .encoder import EDSR


class XLSIM(BaseModel):
    def __init__(self, encoder_kwargs: dict, fusion_kwargs: dict, decoder_kwargs: dict):
        super().__init__()
        self.encoder = EDSR(**encoder_kwargs)
        self.fusion = HAT(**fusion_kwargs)
        self.decoder = torch.nn.PixelShuffle(**decoder_kwargs)

    def forward(self, x: torch.Tensor):
        # x.shape: BxNxCxHxW
        x = self.encoder(x)  # Returns: BxNxCxHxW
        x = self.fusion(x)  # Returns: BxCxSHxSW
        # x = self.decoder(x)  # Returns: BxCxSHxSW
        return x
