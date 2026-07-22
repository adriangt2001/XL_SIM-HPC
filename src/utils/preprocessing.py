from random import randrange

import torch
import torch.nn.functional as F
from PIL.ImageFile import ImageFile
from typing import Literal

def crop_pil(
    image: ImageFile,
    crop_size: int,
    pair_image: ImageFile | None = None,
    pair_scale_factor: float | None = None,
    offset: tuple[int, int, int, int] = (0, 0, 0, 0),
    mode: Literal['random', 'center'] = 'center',
):
    # Define limits of the image
    W, H = image.size

    left = 0 + offset[0]
    top = 0 + offset[1]
    right = W - offset[2]
    bottom = H - offset[3]

    # Pad image
    full_pad_v = max(0, crop_size - (bottom - top))
    first_pad_v = full_pad_v // 2
    second_pad_v = full_pad_v - first_pad_v
    top -= first_pad_v
    bottom += second_pad_v

    full_pad_h = max(0, crop_size - (right - left))
    first_pad_h = full_pad_h // 2
    second_pad_h = full_pad_h - first_pad_h
    left -= first_pad_h
    right += second_pad_h

    padding = [
        first_pad_h,
        first_pad_v,
        second_pad_h,
        second_pad_v
    ]

    # Crop image
    match mode:
        case 'random':
            top = randrange(top, bottom - crop_size + 1)
            left = randrange(left, right - crop_size + 1)
        case 'center':
            top = (top + bottom - crop_size) // 2
            left = (left + right - crop_size) // 2
        case _:
            raise ValueError(f"Mode {mode} not supported. Choose between 'random' and 'center'")

    image = image.crop(
        (
            left,
            top,
            left + crop_size,
            top + crop_size,
        )
    )

    if pair_image is not None:
        top = int(top * pair_scale_factor)
        left = int(left * pair_scale_factor)
        pair_crop_size = int(crop_size * pair_scale_factor)

        pair_image = pair_image.crop(
            (left, top, left + pair_crop_size, top + pair_crop_size)
        )

    return image, pair_image, padding


def crop_tensor(
    image: torch.Tensor,
    crop_size: int,
    pair_image: torch.Tensor | None = None,
    pair_scale_factor: float | None = None,
    offset: torch.Tensor | None = None,
    mode: Literal['random', 'center'] = 'center',
):
    if offset is None:
        offset = torch.zeros((4), dtype=torch.int32)
    # Define limits of the image
    H, W = image.shape[-2:]

    left_off = 0 + offset[0]
    top_off = 0 + offset[1]
    right_off = W - offset[2]
    bottom_off = H - offset[3]

    image = image[..., top_off:bottom_off, left_off:right_off]
    H, W = image.shape[-2:]
    left = 0
    top = 0
    right = W
    bottom = H

    # Pad the image
    full_pad_v = max(0, crop_size - H)
    first_pad_v = full_pad_v // 2
    second_pad_v = full_pad_v - first_pad_v

    full_pad_h = max(0, crop_size - W)
    first_pad_h = full_pad_h // 2
    second_pad_h = full_pad_h - first_pad_h

    padding = [
        first_pad_h,
        first_pad_v,
        second_pad_h,
        second_pad_v
    ]

    image = F.pad(
        image,
        [first_pad_h, second_pad_h, first_pad_v, second_pad_v],
        mode="constant",
        value=0,
    )

    H, W = image.shape[-2:]
    bottom = H
    right = W

    # Crop image
    match mode:
        case 'random':
            top = randrange(top, bottom - crop_size + 1)
            left = randrange(left, right - crop_size + 1)
        case 'center':
            top = (top + bottom - crop_size) // 2
            left = (left + right - crop_size) // 2
        case _:
            raise ValueError(f"Mode {mode} not supported. Choose between 'random' and 'center'")

    image = image[..., top : top + crop_size, left : left + crop_size]

    if pair_image is not None:
        top_off = int(top_off * pair_scale_factor)
        left_off = int(left_off * pair_scale_factor)
        bottom_off = int(bottom_off * pair_scale_factor)
        right_off = int(right_off * pair_scale_factor)

        top = int(top * pair_scale_factor)
        left = int(left * pair_scale_factor)
        pair_crop_size = int(crop_size * pair_scale_factor)

        pair_image = pair_image[
            ...,
            top_off : bottom_off,
            left_off : right_off,
        ]
        pair_image = F.pad(
            pair_image,
            [
                int(first_pad_h * pair_scale_factor),
                int(second_pad_h * pair_scale_factor),
                int(first_pad_v * pair_scale_factor),
                int(second_pad_v * pair_scale_factor),
            ],
            mode="constant",
            value=0,
        )

        pair_image = pair_image[
            ..., top : top + pair_crop_size, left : left + pair_crop_size
        ]

    return image, pair_image, padding
