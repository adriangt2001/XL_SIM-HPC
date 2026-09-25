from pathlib import Path

import numpy as np
import torch


def main_logger():
    import logging
    import sys
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    log_handler = logging.StreamHandler(sys.stdout)
    log_handler.setLevel(logging.INFO)
    log_formatter = logging.Formatter('[%(levelname)s] %(message)s')
    log_handler.setFormatter(log_formatter)
    logger.addHandler(log_handler)

    return logger

def load_image_np(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix in (".tif", ".tiff"):
        try:
            import tifffile

            img = tifffile.imread(path)
            img = np.asarray(img).astype(np.float32)
            
        except ImportError:
            raise ImportError("Reading .tif requires `pip install tifffile`.")
    else:
        from PIL import Image

        img = np.array(Image.open(path).convert("F"))
        img = np.asarray(img).astype(np.float32)

        if img.ndim == 3:
            img = img.mean(axis=-1)
        if img.ndim != 2:
            raise ValueError(f"Expected a 2D (or 2D+channel) image, got shape {img.shape}")

    return img

def load_image_torch(path: Path) -> torch.Tensor:
    suffix = path.suffix.lower()
    if suffix in (".tif", ".tiff"):
        try:
            import tifffile

            img = tifffile.imread(path)
            img = np.asarray(img).astype(np.float32)
        except ImportError:
            raise ImportError("Reading .tif requires `pip install tifffile`.")
    else:
        from PIL import Image

        img = np.array(Image.open(path).convert("F"))
        img = np.asarray(img).astype(np.float32) / 255

        if img.ndim == 3:
            img = img.mean(axis=-1)
        if img.ndim != 2:
            raise ValueError(f"Expected a 2D (or 2D+channel) image, got shape {img.shape}")

    img = torch.from_numpy(img)

    return img

def save_image(filename: Path, image: np.ndarray):
    suffix = filename.suffix.lower()
    if suffix in (".tif", ".tiff"):
        try:
            import tifffile

            tifffile.imwrite(filename, image)
        except ImportError:
            raise ImportError("Writing .tif requires `pip install tifffile`.")
    else:
        import cv2

        cv2.imwrite(filename, image)