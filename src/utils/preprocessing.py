import torch
from torchvision.transforms.functional import crop


def sr_random_crop(source: torch.Tensor, target: torch.Tensor, source_size: int, target_size: int):
    top_lr = torch.randint(0, source.shape[2] - source_size + 1, (1,)).item()
    left_lr = torch.randint(0, source.shape[3] - source_size + 1, (1,)).item()
    top_gt = top_lr * 2
    left_gt = left_lr * 2
    
    source = crop(source, int(top_lr), int(left_lr), source_size, source_size)
    target = crop(target, int(top_gt), int(left_gt), target_size, target_size)

    return source, target
