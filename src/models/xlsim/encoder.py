import torch
from torch import nn


class LayerNorm2d(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=eps)

    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        return x.permute(0, 3, 1, 2)


class ResBlock(nn.Module):
    def __init__(
        self,
        n_feats: int,
        kernel_size: int,
        bias: bool = True,
        ln: bool = False,
        act: torch.nn.Module | None = None,
        res_scale: float = 1.0,
    ):
        super().__init__()

        if act is None:
            act = nn.GELU()
        m = []
        for i in range(2):
            m.append(
                nn.Conv2d(
                    n_feats, n_feats, kernel_size, padding=(kernel_size // 2), bias=bias
                )
            )
            if ln:
                m.append(LayerNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res


class EDSR(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        n_feats: int = 64,
        kernel_size: int = 3,
        n_resblocks: int = 16,
        res_scale: float = 1.0,
    ):
        super().__init__()

        padding = kernel_size // 2
        act = nn.GELU()

        # define head module
        m_head = [nn.Conv2d(in_channels, n_feats, kernel_size, padding=padding)]

        # define body module
        m_body = [
            ResBlock(n_feats, kernel_size, act=act, res_scale=res_scale)
            for _ in range(n_resblocks)
        ]
        m_body.append(nn.Conv2d(n_feats, n_feats, kernel_size, padding=padding))

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)

    def forward(self, x: torch.Tensor):
        is_5d = x.ndim == 5
        if is_5d:
            B, N, C, H, W = x.shape
            x = x.reshape(B * N, C, H, W).to(memory_format=torch.channels_last)

        x = self.head(x)
        res = self.body(x)
        res += x

        if is_5d:
            _, C_out, H_out, W_out = res.shape
            res = res.view(B, N, C_out, H_out, W_out)

        return res
