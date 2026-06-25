from random import randrange

import torch
from PIL.ImageFile import ImageFile


def crop_pil(image: ImageFile, crop_size: int):
    image = image.crop((0, 0, crop_size, crop_size))
    return image

def random_crop_pil(image: ImageFile, crop_size: int):
    top_gt = max(randrange(0, image.size[1] - crop_size + 1), crop_size)
    left_gt = max(randrange(0, image.size[0] - crop_size + 1), crop_size)
    image = image.crop((left_gt, top_gt, left_gt + crop_size, top_gt + crop_size))
    return image

def sr_random_crop_pil(
    source: ImageFile, target: ImageFile, source_size: int, target_size: int
):
    top_gt = max(randrange(0, target.size[1] - target_size + 1), target_size)
    left_gt = max(randrange(0, target.size[0] - target_size + 1), target_size)
    target = target.crop((left_gt, top_gt, left_gt + target_size, top_gt + target_size))

    top_lr = top_gt // (target_size // source_size)
    left_lr = left_gt // (target_size // source_size)
    source = source.crop((left_lr, top_lr, left_lr + source_size, top_lr + source_size))

    return source, target


def sr_random_crop_tensor(
    source: torch.Tensor, target: torch.Tensor, source_size: int, target_size: int
):
    top_gt = randrange(0, target.shape[-2] - target_size + 1)
    left_gt = randrange(0, target.shape[-1] - target_size + 1)
    target = target[..., top_gt : top_gt + target_size, left_gt : left_gt + target_size]

    top_lr = top_gt // (target_size // source_size)
    left_lr = left_gt // (target_size // source_size)
    source = source[..., top_lr : top_lr + source_size, left_lr : left_lr + source_size]

    return source, target
