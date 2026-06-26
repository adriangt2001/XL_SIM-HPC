from random import randrange

import torch
from PIL.ImageFile import ImageFile


def crop_pil(
    image: ImageFile,
    crop_size: int,
    pair_image: ImageFile | None = None,
    pair_scale_factor: float | None = None,
    random: bool = False,
):
    # Pad + crop combooooo
    # Define limits of the image
    W, H = image.size

    left = 0
    top = 0
    right = W
    bottom = H

    if crop_size > H:
        full_pad = crop_size - H
        first_pad = full_pad // 2
        second_pad = full_pad - first_pad
        top -= first_pad
        bottom += second_pad
    if crop_size > W:
        full_pad = crop_size - W
        first_pad = full_pad // 2
        second_pad = full_pad - first_pad
        left -= first_pad
        right += second_pad

    # Define crop for image
    if random:
        top = randrange(0, bottom - crop_size + 1)
        left = randrange(0, right - crop_size + 1)

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

    return image, pair_image


def crop_tensor(
    image: torch.Tensor,
    crop_size: int,
    pair_image: torch.Tensor | None = None,
    pair_scale_factor: int | None = None,
    random: bool = False,
):
    # Pad + crop combooooo
    # Define limits of the image
    W, H = image.size

    left = 0
    top = 0
    right = W
    bottom = H

    if crop_size > H:
        full_pad = crop_size - H
        first_pad = full_pad // 2
        second_pad = full_pad - first_pad
        top -= first_pad
        bottom += second_pad
    if crop_size > W:
        full_pad = crop_size - W
        first_pad = full_pad // 2
        second_pad = full_pad - first_pad
        left -= first_pad
        right += second_pad

    # Define crop for image
    if random:
        top = randrange(0, bottom - crop_size + 1)
        left = randrange(0, right - crop_size + 1)

    image = image[left:left + crop_size, top:top + crop_size]

    if pair_image is not None:
        top *= pair_scale_factor
        left *= pair_scale_factor
        crop_size *= pair_scale_factor

        pair_image = pair_image[left:left + crop_size, top:top + crop_size]

    return image, pair_image
