"""
visualize_microscope_output.py

Loads a single HR image from a dataset (LSDIR or BioSR, via datasets.py's
get_data()), passes it through the microscope simulator, and saves:
  - the HR input image (what went into the microscope) as a PNG
  - each raw output frame as its own individual PNG (NOT a stacked
    N-channel file)
  - a zoomed ROI crop of the input AND of one representative output frame
    (default: frame 0), so you can inspect fine detail directly
  - reference PNGs of the full input/frame with a box drawn where the ROI
    was taken from
  - a combined overview PNG (input + grid of all output frames)

--------------------------------------------------------------------------
NOTES
--------------------------------------------------------------------------
- PNGs are 8-bit: pixel values are contrast-stretched (0.5-99.5 percentile)
  and rounded to uint8 for display. This is lossy - if you need exact
  intensity values back for line-profile/FRC measurements, keep the earlier
  .tif-based script's outputs for that; use these PNGs for figure assembly.
- The HR input and the raw output frames are NOT on the same pixel grid:
  per microscope.py, the HR image is `binn_simu`x finer than the camera-
  resolution output frames. The ROI is specified in OUTPUT-frame pixel
  coordinates and automatically scaled by `binn_simu` for the input image,
  so both zoomed crops show the same physical region of the sample.
- `--first-crop` doesn't need to exactly match the microscope's internal
  simulation grid - Microscope.noisy_reading() bicubic-resizes internally
  regardless - but staying close to the native size (1024 for the supplied
  lessnoise_microscope.yaml) avoids unnecessary resampling.
- Assumes single_plane=True (D=1) - only the first depth slice is used.

Usage:
    python visualize_microscope_output.py \
        --dataset /path/to/LSDIR --microscope-config lessnoise_microscope.yaml \
        --noise-config noise.yaml --index 0 --out-dir microscope_vis \
        --roi-size 64 --roi-zoom 4
"""

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Subset

from src.microscope.sim_pipeline import SimulatorPipeline

from ..data.datasets import get_data

# ---------------------------------------------------------------------------
# Image conversion helpers
# ---------------------------------------------------------------------------

def to_uint8(img: np.ndarray) -> np.ndarray:
    """Contrast-stretch a float image (0.5-99.5 percentile) to 8-bit for PNG
    display. Percentile stretching (rather than plain min/max) avoids a few
    hot/dead pixels blowing out the whole image's contrast."""
    img = img.astype(np.float32)
    vmin, vmax = np.percentile(img, [0.5, 99.5])
    if vmax <= vmin:
        vmin, vmax = float(img.min()), float(img.max())
    if vmax <= vmin:
        vmax = vmin + 1e-6
    img_clipped = np.clip((img - vmin) / (vmax - vmin), 0, 1)
    return (img_clipped * 255).round().astype(np.uint8)


def save_png(img: np.ndarray, path: Path):
    Image.fromarray(to_uint8(img), mode="L").save(path)


def crop_roi(img: np.ndarray, x: int, y: int, size: int) -> tuple[np.ndarray, int, int]:
    """Crop a size x size ROI, clamping the top-left corner so it stays in
    bounds. Returns the crop plus the (possibly clamped) x, y actually used."""
    H, W = img.shape
    x = max(0, min(x, W - size))
    y = max(0, min(y, H - size))
    return img[y : y + size, x : x + size], x, y


def save_roi_png(img: np.ndarray, x: int, y: int, size: int, path: Path, zoom: int = 4):
    roi, x, y = crop_roi(img, x, y, size)
    pil_img = Image.fromarray(to_uint8(roi), mode="L")
    if zoom > 1:
        pil_img = pil_img.resize((size * zoom, size * zoom), resample=Image.NEAREST)
    pil_img.save(path)
    return x, y


def save_with_roi_box(img: np.ndarray, x: int, y: int, size: int, path: Path):
    """Save the full image (contrast-stretched) with a red rectangle marking
    where the ROI was taken from - handy when assembling a figure."""
    pil_img = Image.fromarray(to_uint8(img), mode="L").convert("RGB")
    draw = ImageDraw.Draw(pil_img)
    line_width = max(1, size // 32)
    draw.rectangle([x, y, x + size, y + size], outline=(255, 0, 0), width=line_width)
    pil_img.save(path)


# ---------------------------------------------------------------------------
# Dataset / simulator helpers
# ---------------------------------------------------------------------------

def get_single_sample(loader: DataLoader, index: int):
    """Fetch exactly one dataset sample by index, through a batch-size-1
    DataLoader so the original collate_fn / padding logic still applies."""
    subset = Subset(loader.dataset, [index])
    single_loader = DataLoader(
        subset, batch_size=1, shuffle=False, num_workers=0,
        collate_fn=getattr(loader, "collate_fn", None),
    )
    return next(iter(single_loader))


def make_overview_figure(input_img: np.ndarray, output_stack: np.ndarray, out_path: Path):
    """input_img: (H, W) HR image. output_stack: (N, H, W) raw output frames."""
    n_frames = output_stack.shape[0]
    n_cols = min(6, max(1, math.ceil(math.sqrt(n_frames))))
    n_rows = math.ceil(n_frames / n_cols)

    fig = plt.figure(figsize=(3 * (n_cols + 1), 3 * n_rows))
    gs = fig.add_gridspec(n_rows, n_cols + 1)

    ax_input = fig.add_subplot(gs[:, 0])
    ax_input.imshow(input_img, cmap="gray")
    ax_input.set_title("Microscope input\n(HR ground truth)")
    ax_input.axis("off")

    for n in range(n_frames):
        r, c = divmod(n, n_cols)
        ax = fig.add_subplot(gs[r, c + 1])
        ax.imshow(output_stack[n], cmap="gray")
        ax.set_title(f"frame {n}", fontsize=8)
        ax.axis("off")

    for n in range(n_frames, n_rows * n_cols):
        r, c = divmod(n, n_cols)
        ax = fig.add_subplot(gs[r, c + 1])
        ax.axis("off")

    fig.suptitle("Microscope simulation: input vs. raw output frames")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset", required=True, type=Path,
                         help="Path to an LSDIR or BioSR dataset root (see datasets.py get_data)")
    parser.add_argument("--microscope-config", required=True, type=Path)
    parser.add_argument("--noise-config", required=True, type=Path)
    parser.add_argument("--index", type=int, default=0, help="Dataset index to visualize")
    parser.add_argument("--first-crop", type=int, default=1024,
                         help="HR crop size fed to the dataset transform (see notes above)")
    parser.add_argument("--test-size", type=float, default=0.1)
    parser.add_argument("--split", type=str, default="test",
                         help="Only used for LSDIR (which split folder to load)")
    parser.add_argument("--out-dir", type=Path, default=Path("microscope_vis"))
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--roi-size", type=int, default=64,
                         help="ROI size in OUTPUT-frame pixels (square). Default 64.")
    parser.add_argument("--roi-x", type=int, default=None,
                         help="ROI top-left x, in output-frame pixels. Default: frame center.")
    parser.add_argument("--roi-y", type=int, default=None,
                         help="ROI top-left y, in output-frame pixels. Default: frame center.")
    parser.add_argument("--roi-zoom", type=int, default=4,
                         help="Nearest-neighbor upscale factor applied to saved ROI crops for visibility.")
    parser.add_argument("--roi-frame-index", type=int, default=0,
                         help="Which output frame to also save a zoomed ROI for (default: frame 0).")
    args = parser.parse_args()

    device = torch.device(args.device)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    _, _, test_loader = get_data(
        str(args.dataset), args.test_size, args.first_crop, args.split,
        batch_size=1, num_workers=0,
    )

    batch = get_single_sample(test_loader, args.index)
    targets = batch["hr"].to(device=device)  # (1, D, H, W)

    simulator = SimulatorPipeline.from_file(
        str(args.microscope_config), str(args.noise_config)
    ).to(device=device)

    with torch.inference_mode():
        output_stack, calibs = simulator(targets)  # output_stack: (1, N, H, W)

    input_np = targets[0, 0].detach().cpu().float().numpy()
    output_np = output_stack[0].detach().cpu().float().numpy()  # (N, H, W)

    # --- Full images as PNG ---
    save_png(input_np, out_dir / "microscope_input.png")
    for n in range(output_np.shape[0]):
        save_png(output_np[n], out_dir / f"microscope_output_frame_{n:03d}.png")
    print(f"Saved input + {output_np.shape[0]} output frame PNGs to {out_dir}")

    # --- Overview figure ---
    fig_path = out_dir / "microscope_overview.png"
    make_overview_figure(input_np, output_np, fig_path)
    print(f"Saved overview figure to {fig_path}")

    # --- ROI zoom (scaled between the HR input grid and the coarser output grid) ---
    scale = simulator.microscope.binn_simu  # HR is `scale`x finer than the output frames
    frame_idx = min(args.roi_frame_index, output_np.shape[0] - 1)
    frame_img = output_np[frame_idx]

    roi_size_lr = args.roi_size
    roi_x_lr = args.roi_x if args.roi_x is not None else frame_img.shape[1] // 2 - roi_size_lr // 2
    roi_y_lr = args.roi_y if args.roi_y is not None else frame_img.shape[0] // 2 - roi_size_lr // 2

    roi_size_hr = roi_size_lr * scale
    roi_x_hr = roi_x_lr * scale
    roi_y_hr = roi_y_lr * scale

    x_lr, y_lr = save_roi_png(
        frame_img, roi_x_lr, roi_y_lr, roi_size_lr,
        out_dir / f"output_frame_{frame_idx:03d}_roi.png", zoom=args.roi_zoom,
    )
    save_with_roi_box(
        frame_img, x_lr, y_lr, roi_size_lr,
        out_dir / f"output_frame_{frame_idx:03d}_with_roi_box.png",
    )

    x_hr, y_hr = save_roi_png(
        input_np, roi_x_hr, roi_y_hr, roi_size_hr,
        out_dir / "input_roi.png", zoom=max(1, args.roi_zoom // scale),
    )
    save_with_roi_box(
        input_np, x_hr, y_hr, roi_size_hr,
        out_dir / "input_with_roi_box.png",
    )

    print(
        f"Saved {roi_size_lr}x{roi_size_lr}px ROI (output-frame coords x={x_lr}, y={y_lr}) "
        f"from frame {frame_idx}, and the matching {roi_size_hr}x{roi_size_hr}px region "
        f"(x={x_hr}, y={y_hr}) from the HR input (scaled by binn_simu={scale})."
    )


if __name__ == "__main__":
    main()
