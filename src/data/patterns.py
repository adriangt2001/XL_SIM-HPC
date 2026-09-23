import numpy as np
from skimage.transform import resize


def sinusoidal_siemens_star(size_px: int, num_spokes: int):
    y, x = np.ogrid[-1 : 1 : complex(0, size_px), -1 : 1 : complex(0, size_px)]
    theta = np.arctan2(y, x)
    star = 0.5 * (1.0 + np.sin(num_spokes * theta))
    return star


def binary_siemens_star(size_px: int, num_spokes: int, oversample: int = 8):
    hr_size = size_px * oversample
    y, x = np.ogrid[-1 : 1 : complex(0, hr_size), -1 : 1 : complex(0, hr_size)]
    theta = np.arctan2(y, x)
    hr_star = (np.sin(num_spokes * theta) >= 0).astype(np.float64)
    lr_star = resize(hr_star, (size_px, size_px), anti_aliasing=True, order=3)
    return lr_star
