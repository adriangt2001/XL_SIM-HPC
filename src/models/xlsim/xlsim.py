import torch

from ..base_model import BaseModel
from .attention import HAT
from .encoder import EDSR


class Decoder(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        pass


class XLSIM(BaseModel):
    def __init__(self):
        super().__init__()
        self.encoder = EDSR()
        self.fusion = HAT()
        self.decoder = Decoder()

    def forward(self, x: torch.Tensor):
        # B, 25, 1, H, W
        x = self.encoder(x)
        x = self.fusion(x)
        x = self.decoder(x)
        return x
