import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def _tensor_to_numpy(img):
    """Convert a tensor (C,H,W) or (H,W) to a numpy image."""
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu()

    if img.ndim == 3:
        img = img.permute(1, 2, 0).numpy()
        if img.shape[-1] == 1:
            img = img[..., 0]
    elif img.ndim == 2:
        img = img.numpy()
    else:
        raise ValueError("Expected tensor with shape (C,H,W) or (H,W).")

    return np.clip(img, 0, 1)


def _imshow(ax, img):
    img = _tensor_to_numpy(img)

    if img.ndim == 2:
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
    else:
        ax.imshow(img)

    ax.axis("off")


def plot_sr_comparison(
    gt,
    lr,
    method_images,
    figsize_per_image=2.6,
    separator_width=0.08,
    title_fontsize=12,
    header_fontsize=14,
    separator_lw=2,
    save_path=None,
    dpi=300,
):
    """
    Plot a comparison figure similar to those used in SR papers.

    Parameters
    ----------
    gt : Tensor
    lr : Tensor
    method_images : list[(Tensor, str)]
        List of (image, method_name)
    """

    n_methods = len(method_images)

    width_ratios = [1, 1, separator_width] + [1] * n_methods

    fig = plt.figure(
        figsize=((2 + n_methods) * figsize_per_image,
                 figsize_per_image + 0.8)
    )

    gs = GridSpec(
        1,
        3 + n_methods,
        figure=fig,
        width_ratios=width_ratios,
        wspace=0.04,
    )

    # ---------------------------------------------------------
    # References
    # ---------------------------------------------------------
    ax_gt = fig.add_subplot(gs[0])
    _imshow(ax_gt, gt)
    ax_gt.set_title("GT", fontsize=title_fontsize)

    ax_input = fig.add_subplot(gs[1])
    _imshow(ax_input, lr)
    ax_input.set_title("LR", fontsize=title_fontsize)

    # ---------------------------------------------------------
    # Separator
    # ---------------------------------------------------------
    ax_sep = fig.add_subplot(gs[2])
    ax_sep.set_xlim(0, 1)
    ax_sep.set_ylim(0, 1)
    ax_sep.axvline(0.5, color="black", lw=separator_lw)
    ax_sep.axis("off")

    # ---------------------------------------------------------
    # Compared methods
    # ---------------------------------------------------------
    ax_methods = []

    for i, (img, name) in enumerate(method_images):
        ax = fig.add_subplot(gs[i + 3])
        _imshow(ax, img)
        ax.set_title(name, fontsize=title_fontsize)
        ax_methods.append(ax)

    plt.subplots_adjust(
        left=0.02,
        right=0.98,
        bottom=0.02,
        top=0.88,
    )

    # ---------------------------------------------------------
    # Group titles (computed from image centers)
    # ---------------------------------------------------------
    fig.canvas.draw()

    ref_center = (
        (ax_gt.get_position().x0 + ax_gt.get_position().x1) / 2
        + (ax_input.get_position().x0 + ax_input.get_position().x1) / 2
    ) / 2

    fig.text(
        ref_center,
        0.92,
        "References",
        ha="center",
        va="bottom",
        fontsize=header_fontsize,
        fontweight="bold",
    )

    if ax_methods:
        method_center = (
            (ax_methods[0].get_position().x0 + ax_methods[0].get_position().x1) / 2
            + (ax_methods[-1].get_position().x0 + ax_methods[-1].get_position().x1) / 2
        ) / 2

        fig.text(
            method_center,
            0.92,
            "Compared methods",
            ha="center",
            va="bottom",
            fontsize=header_fontsize,
            fontweight="bold",
        )

    if save_path is not None:
        plt.savefig(
            save_path,
            dpi=dpi,
            bbox_inches="tight",
            facecolor="white",
        )

    return fig