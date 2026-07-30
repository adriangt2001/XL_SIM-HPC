from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch.nn.functional import l1_loss


def _continuity_loss(input: torch.Tensor, target: torch.Tensor):
    kernel_x = torch.tensor(
        [[[[1.0, -2.0, 1.0]]]], dtype=input.dtype, device=input.device
    )
    kernel_y = torch.tensor(
        [[[[1.0], [-2.0], [1.0]]]], dtype=input.dtype, device=input.device
    )

    d2x = F.conv2d(input, kernel_x, padding=(0, 1))
    d2y = F.conv2d(input, kernel_y, padding=(1, 0))

    loss_continuity = torch.mean(torch.abs(d2x)) + torch.mean(torch.abs(d2y))
    return loss_continuity


def _sparsity_loss(input: torch.Tensor, target: torch.Tensor):
    return torch.mean(torch.abs(input))


def _weighted_l1_loss(input: torch.Tensor, target: torch.Tensor):
    weight_map = target + 0.001
    weight_map = weight_map / weight_map.max()
    loss = l1_loss(input, target, reduction="none")
    weighted_loss = loss * weight_map
    return torch.mean(weighted_loss)


def get_loss(
    l1_weight: int,
    wl1_weight: int,
    continuity_weight: int,
    sparsity_weight: int
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    losses = [l1_loss, _weighted_l1_loss, _continuity_loss, _sparsity_loss]
    weights = [l1_weight, wl1_weight, continuity_weight, sparsity_weight]

    def loss(
        input: torch.Tensor, target: torch.Tensor
    ):
        
        total_loss = sum(
            loss_fn(input, target) * w for loss_fn, w in zip(losses, weights)
        )
        return total_loss

    return loss
