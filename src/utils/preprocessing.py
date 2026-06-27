from random import randrange

import torch
from PIL.ImageFile import ImageFile

def crop_pil(
    image: ImageFile,
    crop_size: int,
    pair_image: ImageFile | None = None,
    pair_scale_factor: float | None = None,
    offset: int = 0,
    random: bool = False,
):
    # Pad + crop combooooo
    # Define limits of the image
    W, H = image.size

    left = 0 + offset
    top = 0 + offset
    right = W - offset
    bottom = H - offset
    padding = []

    if crop_size > bottom:
        full_pad = crop_size - H
        first_pad = full_pad // 2
        second_pad = full_pad - first_pad
        top -= first_pad
        bottom += second_pad
        padding.append(full_pad)
    else:
        padding.append(0)
    if crop_size > right:
        full_pad = crop_size - W
        first_pad = full_pad // 2
        second_pad = full_pad - first_pad
        left -= first_pad
        right += second_pad
        padding.append(full_pad)
    else:
        padding.append(0)

    # Define crop for image
    if random:
        top = randrange(top, bottom - crop_size + 1)
        left = randrange(left, right - crop_size + 1)

    image = image.crop(
        (
            left,
            top,
            left + crop_size,
            top + crop_size,
        )
    )

    if pair_image is not None:
        top *= pair_scale_factor
        left *= pair_scale_factor
        crop_size *= pair_scale_factor

        pair_image = pair_image.crop((left, top, left + crop_size, top + crop_size))

    return image, pair_image, padding


def crop_tensor(
    image: torch.Tensor,
    crop_size: int,
    pair_image: torch.Tensor | None = None,
    pair_scale_factor: int | None = None,
    offset: int = 0,
    random: bool = False,
):
    # Pad + crop combooooo
    # Define limits of the image
    H, W = image.shape[-2:]

    left = 0 + offset
    top = 0 + offset
    right = W - offset
    bottom = H - offset
    padding = []

    if crop_size > bottom:
        full_pad = crop_size - H
        first_pad = full_pad // 2
        second_pad = full_pad - first_pad
        top -= first_pad
        bottom += second_pad
        padding.append(full_pad)
    else:
        padding.append(0)
    if crop_size > right:
        full_pad = crop_size - W
        first_pad = full_pad // 2
        second_pad = full_pad - first_pad
        left -= first_pad
        right += second_pad
        padding.append(full_pad)
    else:
        padding.append(0)

    # Define crop for image
    if random:
        top = randrange(top, bottom - crop_size + 1)
        left = randrange(left, right - crop_size + 1)

    image = image[..., top : top + crop_size, left : left + crop_size]

    if pair_image is not None:
        top *= pair_scale_factor
        left *= pair_scale_factor
        crop_size *= pair_scale_factor

        pair_image = pair_image[..., top : top + crop_size, left : left + crop_size]

    return image, pair_image, padding
