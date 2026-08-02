# Basically HAT with a weird input and a not-upsampled output
# Still have to adapt the code for this speific use-case

import math

import torch
from einops import rearrange
from torch import nn
from torch.nn.init import trunc_normal_


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    From: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (
        x.ndim - 1
    )  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).

    From: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py
    """

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor):
        return drop_path(x, self.drop_prob, self.training)


class ChannelAttention(nn.Module):
    """Channel attention used in RCAN."""

    def __init__(self, num_feat: int, squeeze_factor: int = 16):
        super().__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor):
        y = self.attention(x)
        return x * y


class CAB(nn.Module):
    def __init__(
        self, num_feat: int, compress_ratio: int = 3, squeeze_factor: int = 30
    ):
        super().__init__()

        self.cab = nn.Sequential(
            nn.Conv2d(num_feat, num_feat // compress_ratio, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(num_feat // compress_ratio, num_feat, 3, 1, 1),
            ChannelAttention(num_feat, squeeze_factor),
        )

    def forward(self, x: torch.Tensor):
        return self.cab(x)


class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_layer: type = nn.GELU,
        drop: float = 0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x: torch.Tensor, window_size: int):
    """
    Args:
        x: (b, h, w, c)
        window_size (int): window size

    Returns:
        windows: (num_windows*b, window_size, window_size, c)
    """
    b, h, w, c = x.shape
    x = x.view(b, h // window_size, window_size, w // window_size, window_size, c)
    windows = (
        x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, c)
    )
    return windows


def window_reverse(windows: torch.Tensor, window_size: int, h: int, w: int):
    """
    Args:
        windows: (num_windows*b, window_size, window_size, c)
        window_size (int): Window size
        h (int): Height of image
        w (int): Width of image

    Returns:
        x: (b, h, w, c)
    """
    b = int(windows.shape[0] / (h * w / window_size / window_size))
    x = windows.view(
        b, h // window_size, w // window_size, window_size, window_size, -1
    )
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)
    return x


class WindowAttention(nn.Module):
    r"""Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    """

    def __init__(
        self,
        dim: int,
        window_size: tuple[int, int],
        num_heads: int,
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )  # 2*Wh-1 * 2*Ww-1, nH
        trunc_normal_(self.relative_position_bias_table, std=0.02)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(
        self, x: torch.Tensor, rpi: torch.Tensor, mask: torch.Tensor | None = None
    ):
        """
        Args:
            x: input features with shape of (num_windows*b, n, c)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        b_, n, c = x.shape
        qkv: torch.Tensor = (
            self.qkv(x)
            .reshape(b_, n, 3, self.num_heads, c // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[rpi.view(-1)].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1,
        )  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1
        ).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(b_ // nw, nw, self.num_heads, n, n) + mask.unsqueeze(
                1
            ).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)

        attn: torch.Tensor = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(b_, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class HAB(nn.Module):
    r"""Hybrid Attention Block."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int = 7,
        shift_size: int = 0,
        compress_ratio: int = 3,
        squeeze_factor: int = 30,
        conv_scale: float = 0.01,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: type = nn.GELU,
        norm_layer: type = nn.LayerNorm,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.conv_scale = conv_scale

        assert 0 <= self.shift_size < self.window_size, (
            "shift_size must in 0-window_size"
        )

        self.norm1 = norm_layer(dim)

        self.attn = WindowAttention(
            dim=dim,
            window_size=(window_size, window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.conv_block = CAB(
            num_feat=dim, compress_ratio=compress_ratio, squeeze_factor=squeeze_factor
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(
        self,
        x: torch.Tensor,
        x_size: tuple[int, int],
        rpi_sa: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ):
        h, w = x_size
        b, _, c = x.shape
        # assert seq_len == h * w, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(b, h, w, c)

        # Conv_X
        conv_x = self.conv_block(x.permute(0, 3, 1, 2))
        conv_x = conv_x.permute(0, 2, 3, 1).contiguous().view(b, h * w, c)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(
                x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2)
            )
            mask = attn_mask
        else:
            shifted_x = x
            mask = None

        # partition windows
        x_windows = window_partition(
            shifted_x, self.window_size
        )  # nw*b, window_size, window_size, c
        x_windows = x_windows.view(
            -1, self.window_size * self.window_size, c
        )  # nw*b, window_size*window_size, c

        # W-MSA/SW-MSA (to be compatible for testing on images whose shapes are the multiple of window size
        attn_windows = self.attn(x_windows, rpi=rpi_sa, mask=mask)

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, c)
        shifted_x = window_reverse(attn_windows, self.window_size, h, w)  # b h' w' c

        # reverse cyclic shift
        if self.shift_size > 0:
            attn_x = torch.roll(
                shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2)
            )
        else:
            attn_x = shifted_x
        attn_x = attn_x.view(b, h * w, c)

        # FFN
        x = shortcut + self.drop_path(attn_x) + conv_x * self.conv_scale
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class OCAB(nn.Module):
    # overlapping cross-attention block

    def __init__(
        self,
        dim: int,
        window_size: int,
        overlap_ratio: float,
        num_heads: int,
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        mlp_ratio: float = 2.0,
        norm_layer: type = nn.LayerNorm,
    ):

        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.overlap_win_size = int(window_size * overlap_ratio) + window_size

        self.norm1 = norm_layer(dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.unfold = nn.Unfold(
            kernel_size=(self.overlap_win_size, self.overlap_win_size),
            stride=window_size,
            padding=(self.overlap_win_size - window_size) // 2,
        )

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(
                (window_size + self.overlap_win_size - 1)
                * (window_size + self.overlap_win_size - 1),
                num_heads,
            )
        )  # 2*Wh-1 * 2*Ww-1, nH

        trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

        self.proj = nn.Linear(dim, dim)

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU
        )

    def forward(self, x: torch.Tensor, x_size: tuple[int, int], rpi: torch.Tensor):
        h, w = x_size
        b, _, c = x.shape

        shortcut = x
        x = self.norm1(x)
        x = x.view(b, h, w, c)

        qkv = self.qkv(x).reshape(b, h, w, 3, c).permute(3, 0, 4, 1, 2)  # 3, b, c, h, w
        q = qkv[0].permute(0, 2, 3, 1)  # b, h, w, c
        kv = torch.cat((qkv[1], qkv[2]), dim=1)  # b, 2*c, h, w

        # partition windows
        q_windows = window_partition(
            q, self.window_size
        )  # nw*b, window_size, window_size, c
        q_windows = q_windows.view(
            -1, self.window_size * self.window_size, c
        )  # nw*b, window_size*window_size, c

        kv_windows = self.unfold(kv)  # b, c*w*w, nw
        kv_windows = rearrange(  # ?????
            kv_windows,
            "b (nc ch owh oww) nw -> nc (b nw) (owh oww) ch",
            nc=2,
            ch=c,
            owh=self.overlap_win_size,
            oww=self.overlap_win_size,
        ).contiguous()  # 2, nw*b, ow*ow, c
        k_windows, v_windows = kv_windows[0], kv_windows[1]  # nw*b, ow*ow, c

        b_, nq, _ = q_windows.shape
        _, n, _ = k_windows.shape
        d = self.dim // self.num_heads
        q = q_windows.reshape(b_, nq, self.num_heads, d).permute(
            0, 2, 1, 3
        )  # nw*b, nH, nq, d
        k = k_windows.reshape(b_, n, self.num_heads, d).permute(
            0, 2, 1, 3
        )  # nw*b, nH, n, d
        v = v_windows.reshape(b_, n, self.num_heads, d).permute(
            0, 2, 1, 3
        )  # nw*b, nH, n, d

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[rpi.view(-1)].view(
            self.window_size * self.window_size,
            self.overlap_win_size * self.overlap_win_size,
            -1,
        )  # ws*ws, wse*wse, nH
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1
        ).contiguous()  # nH, ws*ws, wse*wse
        attn = attn + relative_position_bias.unsqueeze(0)

        attn = self.softmax(attn)
        attn_windows = (attn @ v).transpose(1, 2).reshape(b_, nq, self.dim)

        # merge windows
        attn_windows = attn_windows.view(
            -1, self.window_size, self.window_size, self.dim
        )
        x = window_reverse(attn_windows, self.window_size, h, w)  # b h w c
        x = x.view(b, h * w, self.dim)

        x = self.proj(x) + shortcut

        x = x + self.mlp(self.norm2(x))
        return x


class AttenBlocks(nn.Module):
    """A series of attention blocks for one RHAG."""

    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: int,
        compress_ratio: int,
        squeeze_factor: int,
        conv_scale: float,
        overlap_ratio: float,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float | list[float] = 0.0,
        norm_layer: type = nn.LayerNorm,
    ):

        super().__init__()
        self.dim = dim
        self.depth = depth

        # build blocks
        self.blocks = nn.ModuleList(
            [
                HAB(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else window_size // 2,
                    compress_ratio=compress_ratio,
                    squeeze_factor=squeeze_factor,
                    conv_scale=conv_scale,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i]
                    if isinstance(drop_path, list)
                    else drop_path,
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )

        # OCAB
        self.overlap_attn = OCAB(
            dim=dim,
            window_size=window_size,
            overlap_ratio=overlap_ratio,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            mlp_ratio=mlp_ratio,
            norm_layer=norm_layer,
        )

    def forward(
        self, x: torch.Tensor, x_size: tuple[int, int], params: dict[str, torch.Tensor]
    ):
        for blk in self.blocks:
            x = blk(x, x_size, params["rpi_sa"], params["attn_mask"])

        x = self.overlap_attn(x, x_size, params["rpi_oca"])

        return x


class RHAG(nn.Module):
    """Residual Hybrid Attention Group (RHAG)."""

    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: int,
        compress_ratio: int,
        squeeze_factor: int,
        conv_scale: float,
        overlap_ratio: float,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float | list[float] = 0.0,
        norm_layer: type = nn.LayerNorm,
        resi_connection: str = "1conv",
    ):
        super().__init__()

        self.dim = dim

        self.residual_group = AttenBlocks(
            dim=dim,
            depth=depth,
            num_heads=num_heads,
            window_size=window_size,
            compress_ratio=compress_ratio,
            squeeze_factor=squeeze_factor,
            conv_scale=conv_scale,
            overlap_ratio=overlap_ratio,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop,
            attn_drop=attn_drop,
            drop_path=drop_path,
            norm_layer=norm_layer,
        )

        if resi_connection == "1conv":
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        elif resi_connection == "identity":
            self.conv = nn.Identity()

    def forward(
        self, x: torch.Tensor, x_size: tuple[int, int], params: dict[str, torch.Tensor]
    ):
        B, C, H, W = x.shape

        x_seq = x.flatten(2).transpose(1, 2)
        out_seq: torch.Tensor = self.residual_group(x_seq, x_size, params)
        out_2d = out_seq.transpose(1, 2).view(B, C, H, W)

        out: torch.Tensor = self.conv(out_2d) + x
        return out


class Upsample(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log2(scale))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(
                f"scale {scale} is not supported. Supported scales: 2^n and 3."
            )
        super().__init__(*m)


class SampleAttention(nn.Module):
    """Multi-Head Self-Attention along sample sequence dimension."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor):
        b_sp, s, c = x.shape
        qkv: torch.Tensor = (
            self.qkv(x)
            .reshape(b_sp, s, 3, self.num_heads, c // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(b_sp, s, c)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class SampleBlock(nn.Module):
    """Single Transformer Block for Sample Attention ()"""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float | list[float] = 0.0,
        act_layer: type = nn.GELU,
        norm_layer: type = nn.LayerNorm,
    ):
        super().__init__()

        self.norm1 = norm_layer(dim)
        self.attn = SampleAttention()

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer
        )

    def forward(self, x: torch.Tensor):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class RHAG_Sample(nn.Module):
    """Custom RHAG but across different frames instead of spatially in a single frame."""

    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        num_samples: int = 25,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float | list[float] = 0.0,
        act_layer: type = nn.GELU,
        norm_layer: type = nn.LayerNorm,
        resi_connection: str = "1conv",
    ):
        super().__init__()

        self.dim = dim
        self.num_samples = num_samples

        self.sample_pos_embed = nn.Parameter(torch.zeros(1, num_samples, dim))
        trunc_normal_(self.sample_pos_embed, std=0.02)

        self.blocks = nn.ModuleList(
            [
                SampleBlock(
                    dim=dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i]
                    if isinstance(drop_path, list)
                    else drop_path,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )

        if resi_connection == "1conv":
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        elif resi_connection == "identity":
            self.conv = nn.Identity()

    def forward(self, x: torch.Tensor):
        B, S, C, H, W = x.shape

        x_seq = x.permute(0, 3, 4, 1, 2).reshape(B * H * W, S, C)

        x_seq = x_seq + self.sample_pos_embed

        for blk in self.blocks:
            x_seq = blk(x_seq)

        out_5d = x_seq.view(B, H, W, S, C).permute(0, 3, 4, 1, 2)
        out_flat = out_5d.view(B * S, C, H, W)
        conv_out: torch.Tensor = self.conv(out_flat).view(B, S, C, H, W)

        return conv_out + x


class SampleFusion(nn.Module):
    """Sample Fusion Module.

    Fuses 5D features [B, S, C, H, W] across S subpixel samples into a 4D
    feature representation [B, C, H, W] for downstream spatial attention.
    """

    def __init__(
        self,
        dim=int,
        num_samples: int = 25,
        act_layer: type = nn.GELU,
        use_sample_attention: bool = True,
    ):
        super().__init__()

        self.dim = dim
        self.num_samples = num_samples
        self.use_sample_attention = use_sample_attention

        if use_sample_attention:
            self.sample_weight_net = nn.Sequential(
                nn.Conv3d(dim, dim // 2, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
                act_layer(),
                nn.Conv3d(dim // 2, 1, kernel_size=(1, 1, 1)),
            )

        self.fusion_conv = nn.Sequential(
            nn.Conv2d(num_samples * dim, dim, kernel_size=3, stride=1, padding=1),
            act_layer(),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x: torch.Tensor):
        B, S, C, H, W = x.shape

        shortcut = x.mean(dim=1)

        if self.use_sample_attention:
            x_3d = x.permute(0, 2, 1, 3, 4)
            weights = self.sample_weight_net(x_3d)
            weights = torch.softmax(weights, dim=2)
            x_weighted = (x_3d * weights).permute(0, 2, 1, 3, 4)
        else:
            x_weighted = x

        x_flat = x_weighted.reshape(B, S * C, H, W)
        out = self.fusion_conv(x_flat)

        return out + shortcut


class HAT(nn.Module):
    r"""Adapted Hybrid Attention Transformer"""

    def __init__(
        self,
        num_samples: int = 25,
        in_feats: int = 64,
        out_chans: int = 1,
        embed_dim: int = 64,
        sample_depths: tuple[int] = (6, 6, 6, 6),
        spatial_depths: tuple[int] = (6, 6),
        num_heads: tuple[int] = (4, 4, 4, 4, 4, 4),
        window_size: int = 8,
        compress_ratio: int = 3,
        squeeze_factor: int = 30,
        conv_scale: float = 0.01,
        overlap_ratio: float = 0.5,
        mlp_ratio=4.0,
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        act_layer: type = nn.GELU,
        norm_layer: type | None = nn.LayerNorm,
        upscale: int = 2,
        resi_connection: str = "1conv",
        upsample_feats: int = 64,
    ):
        super().__init__()

        self.num_samples = num_samples
        self.embed_dim = embed_dim
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.overlap_ratio = overlap_ratio
        self.upscale = upscale

        # relative position index
        relative_position_index_SA = self.calculate_rpi_sa()
        relative_position_index_OCA = self.calculate_rpi_oca()
        self.register_buffer("relative_position_index_SA", relative_position_index_SA)
        self.register_buffer("relative_position_index_OCA", relative_position_index_OCA)

        # ------------------------- 1, shallow feature extraction ------------------------- #
        self.conv_first = nn.Conv2d(in_feats, embed_dim, 3, 1, 1)

        # Learnable Position embedding
        self.sample_pos_embed = nn.Parameter(
            torch.zeros(1, num_samples, embed_dim, 1, 1)
        )
        trunc_normal_(self.sample_pos_embed, std=0.02)

        # ------------------------- 2, deep feature extraction ------------------------- #
        total_depth = sum(sample_depths) + sum(spatial_depths)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_depth)]

        # Phase 1
        self.sample_layers = nn.ModuleList()
        depth_offset = 0
        for i_layer in range(len(sample_depths)):
            layer = RHAG_Sample(
                dim=embed_dim,
                depth=sample_depths[i_layer],
                num_heads=num_heads[i_layer],
                num_samples=num_samples,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[depth_offset : depth_offset + sample_depths[i_layer]],
                act_layer=act_layer,
                norm_layer=norm_layer,
                resi_connection=resi_connection,
            )
            self.sample_layers.append(layer)
            depth_offset += sample_depths[i_layer]

        # Fusion Layer
        self.sample_fusion = SampleFusion(
            dim=embed_dim,
            num_samples=num_samples,
            act_layer=act_layer,
            use_sample_attention=True
        )

        # Phase 2
        self.spatial_layers = nn.ModuleList()
        for i_layer in range(len(spatial_depths)):
            layer_idx = len(sample_depths) + i_layer
            layer = RHAG(
                dim=embed_dim,
                depth=spatial_depths[i_layer],
                num_heads=num_heads[layer_idx],
                window_size=window_size,
                compress_ratio=compress_ratio,
                squeeze_factor=squeeze_factor,
                conv_scale=conv_scale,
                overlap_ratio=overlap_ratio,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[depth_offset : depth_offset + spatial_depths[i_layer]],
                norm_layer=norm_layer,
                resi_connection=resi_connection,
            )
            self.spatial_layers.append(layer)
            depth_offset += spatial_depths[i_layer]

        self.norm = norm_layer(embed_dim)

        if resi_connection == "1conv":
            self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        elif resi_connection == "identity":
            self.conv_after_body = nn.Identity()

        # ------------------------- 3, high quality image reconstruction ------------------------- #
        # for classical SR
        self.conv_before_upsample = nn.Sequential(
            nn.Conv2d(embed_dim, upsample_feats, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        self.upsample = Upsample(upscale, upsample_feats)
        self.conv_last = nn.Conv2d(upsample_feats, out_chans, 3, 1, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def calculate_rpi_sa(self):
        # calculate relative position index for SA
        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(
            torch.meshgrid([coords_h, coords_w], indexing="ij")
        )  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = (
            coords_flatten[:, :, None] - coords_flatten[:, None, :]
        )  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(
            1, 2, 0
        ).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        return relative_position_index

    def calculate_rpi_oca(self):
        # calculate relative position index for OCA
        window_size_ori = self.window_size
        window_size_ext = self.window_size + int(self.overlap_ratio * self.window_size)

        coords_h = torch.arange(window_size_ori)
        coords_w = torch.arange(window_size_ori)
        coords_ori = torch.stack(
            torch.meshgrid([coords_h, coords_w], indexing="ij")
        )  # 2, ws, ws
        coords_ori_flatten = torch.flatten(coords_ori, 1)  # 2, ws*ws

        coords_h = torch.arange(window_size_ext)
        coords_w = torch.arange(window_size_ext)
        coords_ext = torch.stack(
            torch.meshgrid([coords_h, coords_w], indexing="ij")
        )  # 2, wse, wse
        coords_ext_flatten = torch.flatten(coords_ext, 1)  # 2, wse*wse

        relative_coords = (
            coords_ext_flatten[:, None, :] - coords_ori_flatten[:, :, None]
        )  # 2, ws*ws, wse*wse

        relative_coords = relative_coords.permute(
            1, 2, 0
        ).contiguous()  # ws*ws, wse*wse, 2
        relative_coords[:, :, 0] += (
            window_size_ori - window_size_ext + 1
        )  # shift to start from 0
        relative_coords[:, :, 1] += window_size_ori - window_size_ext + 1

        relative_coords[:, :, 0] *= window_size_ori + window_size_ext - 1
        relative_position_index = relative_coords.sum(-1)
        return relative_position_index

    def calculate_mask(self, x_size):
        # calculate attention mask for SW-MSA
        h, w = x_size
        img_mask = torch.zeros((1, h, w, 1))  # 1 h w 1
        h_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        w_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(
            img_mask, self.window_size
        )  # nw, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, (-100.0)).masked_fill(
            attn_mask == 0, 0.0
        )

        return attn_mask

    def forward(self, x: torch.Tensor):
        B, N, C, H, W = x.shape
        x_size = (H, W)

        attn_mask = self.calculate_mask(x_size).to(x.device)
        params = {
            "attn_mask": attn_mask,
            "rpi_sa": self.relative_position_index_SA,
            "rpi_oca": self.relative_position_index_OCA,
        }
        x = x.view(B * N, C, H, W)
        x = self.conv_first(x)
        _, C_embed, _, _ = x.shape
        x = x.view(B, N, C_embed, H, W)
        x = (
            x + self.sample_pos_embed
        )  # Addition instead of concat. Harder to train, more memory efficient

        for layer in self.sample_layers:
            x = layer(x, x_size, params)

        x_fused: torch.Tensor = self.sample_fusion(x)

        x_spatial = x_fused
        for layer in self.spatial_layers:
            x_spatial: torch.Tensor = layer(x_spatial, x_size, params)

        x_spatial = x_spatial.flatten(2).transpose(1, 2)
        x_spatial = self.norm(x_spatial)
        x_spatial = x_spatial.transpose(1, 2).view(B, C_embed, H, W)

        x_out = self.conv_after_body(x_spatial) + x_fused

        x_out = self.conv_before_upsample(x_out)
        x_out = self.upsample(x_out)
        out = self.conv_last(x_out)

        return out
