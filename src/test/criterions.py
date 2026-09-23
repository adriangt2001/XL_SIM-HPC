import argparse
from argparse import Namespace
from pathlib import Path

import frc
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.optimize import curve_fit
from skimage.measure import profile_line

unit = "nm"


def load_image(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix in (".tif", ".tiff"):
        try:
            import tifffile

            img = tifffile.imread(path)
        except ImportError:
            raise ImportError("Reading .tif requires `pip install tifffile`.")
    else:
        from PIL import Image

        img = np.array(Image.open(path).convert("F"))

    img = np.asarray(img).astype(np.float32)

    if img.ndim == 3:
        img = img.mean(axis=-1)
    elif img.ndim != 2:
        raise ValueError(f"Expected a 2D (or 2D+channel) image, got shape {img.shape}")

    return img


def parse_point(s: str):
    x, y = s.split(",")
    return float(x), float(y)


def apply_fwhm(args: Namespace):
    img = load_image(args.image)
    x1, y1 = args.p1
    x2, y2 = args.p2

    profile = profile_line(img, args.p1, args.p2, linewidth=1, reduce_func=np.mean)
    segment_length = np.hypot(x2 - x1, y2 - y1)
    distance = np.linspace(0, segment_length, len(profile))
    distance_physic = distance * args.pixel_size

    def gaussian(x, amplitude, mean, sigma, background):
        return background + amplitude * np.exp(-((x - mean) ** 2) / (2 * sigma**2))

    # Initial guesses
    background0 = np.percentile(profile, 10)
    amplitude0 = np.max(profile) - background0
    mean0 = distance[np.argmax(profile)]
    sigma0 = segment_length / 6

    p0 = [amplitude0, mean0, sigma0, background0]

    # Fit
    bounds = [(0, distance.min(), 0, -np.inf), (np.inf, distance.max(), np.inf, np.inf)]

    popt, pcov = curve_fit(gaussian, distance, profile, p0=p0, bounds=bounds)
    amplitude, mean, sigma, background = popt
    mean_physic = mean * args.pixel_size
    sigma_physic = sigma * args.pixel_size

    # FWHM
    fwhm = 2 * np.sqrt(2 * np.log(2)) * sigma
    sigma_uncertainty = np.sqrt(pcov[2, 2])
    fwhm_uncertainty = 2 * np.sqrt(2 * np.log(2)) * sigma_uncertainty
    fwhm_physic = fwhm * args.pixel_size
    fwhm_uncertainty_physic = fwhm_uncertainty * args.pixel_size

    # R²
    fitted_profile = gaussian(distance, *popt)
    residuals = profile - fitted_profile

    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((profile - np.mean(profile)) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot != 0 else -1

    args.out.mkdir(exist_ok=True)
    output_csv: Path = args.out / "fwhm.csv"

    if output_csv.exists():
        df = pd.read_csv(
            output_csv,
        )
    else:
        df = pd.DataFrame(
            columns=[
                "Image",
                "Pixel Size",
                f"FWHM ({unit})",
                f"FWHM Uncertainty ({unit})",
                "R2",
            ]
        )

    df.loc[len(df)] = [
        args.image.stem,
        args.pixel_size,
        fwhm_physic,
        fwhm_uncertainty_physic,
        r_squared,
    ]
    df = df.drop_duplicates()
    df.to_csv(output_csv, index=False)

    if args.visual:
        output_image_folder: Path = args.out / args.image.stem
        output_image_folder.mkdir(exist_ok=True)
        output_image = output_image_folder / "image.png"
        output_plot = output_image_folder / "plot.png"

        # Image
        height, width = img.shape[:2]
        dpi=100
        fig, ax = plt.subplots(figsize = (width/dpi, height/dpi), dpi=dpi)
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

        ax.imshow(img, cmap="magma")
        ax.plot([x1, x2], [y1, y2], "r-", linewidth=1.5)
        ax.axis("off")

        xc = (x1 + x2) / 2
        yc = (y1 + y2) / 2
        segment_size = max(abs(x2 - x1), abs(y2 - y1))
        padding = 0.5
        half_size = segment_size * (1 + padding) / 2

        rect = Rectangle(
            (xc - half_size, yc - half_size),
            2 * half_size,
            2 * half_size,
            fill=False,
            edgecolor="green",
            linewidth=1.5
        )

        ax.add_patch(rect)

        ax_zoom = inset_axes(
            ax, width="40%", height="40%", loc="lower right", borderpad=1
        )
        ax_zoom.imshow(img, cmap="magma")
        ax_zoom.plot([x1, x2], [y1, y2], "r-", linewidth=1.5)
        ax_zoom.set_xlim(xc - half_size, xc + half_size)
        ax_zoom.set_ylim(yc + half_size, yc - half_size)
        ax_zoom.set_xticks([])
        ax_zoom.set_yticks([])
        for spine in ax_zoom.spines.values():
            spine.set_edgecolor("green")
            spine.set_linewidth(2)

        plt.savefig(output_image, dpi=dpi)
        plt.close()

        # Plot
        x_fit = np.linspace(distance.min(), distance.max(), 500)
        y_fit = gaussian(x_fit, *popt)
        half_max = background + amplitude / 2
        left = mean_physic - fwhm / 2
        right = mean_physic + fwhm / 2
        x_fit_physic = x_fit * args.pixel_size

        plt.figure(figsize=(7, 4))
        plt.plot(distance_physic, profile, "o", markersize=4, label="Measured")
        plt.plot(x_fit_physic, y_fit, "-", label="Gaussian fit")

        plt.axhline(half_max, linestyle="--", label="Half maximum")
        plt.axvline(left, linestyle=":")
        plt.axvline(right, linestyle=":")

        plt.xlabel(f"Distance along segment ({unit})")
        plt.ylabel("Intensity")
        plt.legend()

        plt.tight_layout()
        plt.savefig(output_plot)
        plt.close()


def apply_frc(args: Namespace):
    img1 = load_image(args.image1)
    img1 = frc.util.square_image(img1, add_padding=False)
    img1 = frc.util.apply_tukey(img1)

    img2 = None
    if args.image2 is not None:
        img2 = load_image(args.image2)
        img2 = frc.util.square_image(img2, add_padding=False)
        img2 = frc.util.apply_tukey(img2)

    if img2 is not None:
        frc_curve = frc.two_frc(img1, img2)
    else:
        frc_curve = frc.one_frc(img1, method=1)

    img_size = img1.shape[0]
    xs_pix = np.arange(len(frc_curve)) / img_size
    xs_nm_freq = xs_pix / args.pixel_size
    try:
        frc_res, res_y, thres = frc.frc_res(xs_nm_freq, frc_curve, img_size)
    except frc.NoIntersectionException as e:
        frc_res, res_y, thres = None, None, None
        print(
            f"No intersection has been found. Check the plot for further troubleshooting: {e}"
        )

    args.out.mkdir(exist_ok=True)
    output_csv: Path = args.out / "frc.csv"

    if output_csv.exists():
        df = pd.read_csv(
            output_csv,
        )
    else:
        df = pd.DataFrame(
            columns=["Image1", "Image2", "Pixel Size", f"Resolution ({unit})"]
        )

    df.loc[len(df)] = [
        args.image1.stem,
        args.image2.stem if args.image2 is not None else args.image2,
        args.pixel_size,
        frc_res,
    ]
    df = df.drop_duplicates()
    df.to_csv(output_csv, index=False)

    if args.visual:
        if args.image2 is not None:
            output_figure: Path = args.out / (
                args.image1.stem + "_" + args.image2.stem + "_frc_curve.png"
            )
        else:
            output_figure: Path = args.out / (args.image1.stem + "_frc_curve.png")
        if thres is not None:
            plt.plot(xs_nm_freq, thres(xs_nm_freq))
        plt.plot(xs_nm_freq, frc_curve)
        plt.tight_layout()
        plt.savefig(output_figure, dpi=300)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparser = parser.add_subparsers(dest="command", required=True)

    # Fourier Ring Correlation (FRC)
    p_frc = subparser.add_parser("frc", help="Fourier Ring Correlation (FRC)")
    p_frc.add_argument("--image1", type=Path, required=True, help="Base image to use")
    p_frc.add_argument(
        "--image2", type=Path, default=None, help="Image to compare against if any"
    )
    p_frc.add_argument(
        "--threshold",
        type=float,
        default=1 / 7,
        help="Threshold for FRC method. Default: 1/7",
    )
    p_frc.add_argument("--pixel_size", type=float, required=True, help="Pixel size")
    p_frc.add_argument(
        "--diff_limit", type=float, default=None, help="Abbe's diffraction limit"
    )
    p_frc.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output folder for log (frc.csv) and figures",
    )
    p_frc.add_argument("--visual", action="store_true", help="Generate and save figure")
    p_frc.set_defaults(func=apply_frc)

    # Full Width at Half Maximum (FWHM)
    p_fwhm = subparser.add_parser("fwhm", help="Full Width at Half Maximum (FWHM)")
    p_fwhm.add_argument("--image", type=Path, required=True, help="Image to use")
    p_fwhm.add_argument(
        "--p1", type=parse_point, required=True, help="Beginning of segment"
    )
    p_fwhm.add_argument("--p2", type=parse_point, required=True, help="End of segment")
    p_fwhm.add_argument("--pixel_size", type=float, required=True, help="Pixel size")
    p_fwhm.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output folder for log (fwhm.csv) and figures",
    )
    p_fwhm.add_argument(
        "--visual", action="store_true", help="Generate and save figures/images"
    )
    p_fwhm.set_defaults(func=apply_fwhm)

    args = parser.parse_args()
    args.func(args)
