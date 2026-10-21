import argparse
from argparse import Namespace
from pathlib import Path

import frc
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from nanopyx.core.analysis.decorr import DecorrAnalysis
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from skimage.measure import profile_line

UNIT = "nm"
RAYLEIGH_THRESHOLD = 0.735


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


def gaussian(x, amplitude, mean, sigma, background):
    return background + amplitude * np.exp(-((x - mean) ** 2) / (2 * sigma**2))


def fit_gaussian_fwhm(distance: np.ndarray, profile: np.ndarray, pixel_size: float):
    """Fit a single Gaussian to a (distance, profile) pair and derive its FWHM.

    Shared by `apply_fwhm` (fit over the whole requested segment) and
    `apply_houston` (fit over just one emitter's half of the profile).
    """
    # Initial guesses
    background0 = np.percentile(profile, 10)
    amplitude0 = np.max(profile) - background0
    mean0 = distance[np.argmax(profile)]
    sigma0 = (distance.max() - distance.min()) / 6

    p0 = [amplitude0, mean0, sigma0, background0]

    # Fit. Sigma is capped at a few times the fitted window's width: a single
    # peak that "needs" a much wider Gaussian than the window it was measured
    # over isn't a meaningful fit, and without this an under-constrained
    # window (e.g. a half-profile with no visible falling edge) can otherwise
    # run away to an arbitrarily large sigma.
    sigma_max = 5 * (distance.max() - distance.min())
    bounds = [
        (0, distance.min(), 0, -np.inf),
        (np.inf, distance.max(), sigma_max, np.inf),
    ]

    popt, pcov = curve_fit(gaussian, distance, profile, p0=p0, bounds=bounds)
    amplitude, mean, sigma, background = popt
    mean_physic = mean * pixel_size

    # FWHM
    fwhm = 2 * np.sqrt(2 * np.log(2)) * sigma
    sigma_uncertainty = np.sqrt(pcov[2, 2])
    fwhm_uncertainty = 2 * np.sqrt(2 * np.log(2)) * sigma_uncertainty
    fwhm_physic = fwhm * pixel_size
    fwhm_uncertainty_physic = fwhm_uncertainty * pixel_size

    # R²
    fitted_profile = gaussian(distance, *popt)
    residuals = profile - fitted_profile

    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((profile - np.mean(profile)) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot != 0 else -1

    return {
        "popt": popt,
        "amplitude": amplitude,
        "mean": mean,
        "sigma": sigma,
        "background": background,
        "mean_physic": mean_physic,
        "fwhm": fwhm,
        "fwhm_physic": fwhm_physic,
        "fwhm_uncertainty": fwhm_uncertainty,
        "fwhm_uncertainty_physic": fwhm_uncertainty_physic,
        "r_squared": r_squared,
    }


def apply_houston(
    image: Path,
    p1: tuple[int, int],
    p2: tuple[int, int],
    pixel_size: float,
    output_folder: Path,
    visual: bool,
):
    img = load_image(image)
    point1, point2 = np.asarray(p1, float), np.asarray(p2, float)
    sep = np.linalg.norm(point2 - point1)
    if sep == 0:
        raise ValueError("the two points must be different")
    u = (point2 - point1) / sep

    # Extend the segment past peaks
    margin = max(2.0, 0.5 * sep)
    start, end = point1 - margin * u, point2 + margin * u

    # Profile (profile_line expects row,col; points are x,y)
    prof = profile_line(
        img,
        start[::-1],
        end[::-1],
        linewidth=3,
        order=3,
        mode="nearest",
        reduce_func=np.mean,
    )
    total = sep + 2 * margin
    pos = np.linspace(0, total, len(prof))  # distance from `start`
    t1, t2 = margin, margin + sep  # emitter positions on that axis
    mid = (t1 + t2) / 2

    # Refine the two emitter positions, same as Rayleigh/Sparrow
    peaks, _ = find_peaks(prof)
    search_radius_px = max(1.5, 0.35 * sep)

    def best_peak(lo, hi):
        cand = peaks[(pos[peaks] >= lo) & (pos[peaks] <= hi)]
        return cand[np.argmax(prof[cand])] if len(cand) else None

    i1 = best_peak(t1 - search_radius_px, min(mid, t1 + search_radius_px))
    i2 = best_peak(max(mid, t2 - search_radius_px), t2 + search_radius_px)
    two_peaks_found = i1 is not None and i2 is not None and i1 != i2
    peak_distance_px = float(pos[i2] - pos[i1]) if two_peaks_found else sep

    # Houston criterion (empirical/self-referencing variant): fit each emitter
    # individually, splitting the profile at the midpoint between them, to get
    # its own FWHM (reusing the same fit as `apply_fwhm`). The two are then
    # resolved if their separation is at least the (average) individual FWHM.
    im = int(np.argmin(np.abs(pos - mid)))

    def fit_half(lo_idx, hi_idx):
        if hi_idx - lo_idx < 4:
            return None
        try:
            return fit_gaussian_fwhm(
                pos[lo_idx : hi_idx + 1], prof[lo_idx : hi_idx + 1], pixel_size
            )
        except RuntimeError:
            return None

    fit1 = fit_half(0, im)
    fit2 = fit_half(im, len(prof) - 1)

    nan = float("nan")
    fwhms_physic = [fit["fwhm_physic"] for fit in (fit1, fit2) if fit is not None]
    fwhm_ref_physic = float(np.mean(fwhms_physic)) if fwhms_physic else nan
    peak_distance_physic = peak_distance_px * pixel_size

    houston_result = {
        "resolved": bool(fwhms_physic) and peak_distance_physic >= fwhm_ref_physic,
        "peak_distance": peak_distance_physic,
        "peak_distance_px": peak_distance_px,
        "fwhm_ref": fwhm_ref_physic,
        "fwhm1": fit1["fwhm_physic"] if fit1 is not None else nan,
        "fwhm2": fit2["fwhm_physic"] if fit2 is not None else nan,
        "positions": pos,
        "profile": prof,
        "fit1": fit1,
        "fit2": fit2,
        "midpoint_idx": im,
    }

    results = {
        "Image": image.stem,
        "Pixel Size": pixel_size,
        "Resolved": houston_result["resolved"],
        f"Peak Distance ({UNIT})": houston_result["peak_distance"],
        "Peak Distance (px)": houston_result["peak_distance_px"],
        f"Reference FWHM ({UNIT})": houston_result["fwhm_ref"],
        f"FWHM1 ({UNIT})": houston_result["fwhm1"],
        f"FWHM2 ({UNIT})": houston_result["fwhm2"],
    }

    if visual:
        output_image_folder: Path = output_folder / image.stem
        output_image_folder.mkdir(parents=True, exist_ok=True)
        output_plot = output_image_folder / "houston_plot.png"

        pos_physic = pos * pixel_size

        plt.figure(figsize=(7, 4))
        plt.plot(pos_physic, prof, "o", markersize=4, label="Measured")

        for label, fit, (lo_idx, hi_idx) in (
            ("Peak 1 fit", fit1, (0, im)),
            ("Peak 2 fit", fit2, (im, len(prof) - 1)),
        ):
            if fit is None:
                continue
            x_fit = np.linspace(pos[lo_idx], pos[hi_idx], 200)
            y_fit = gaussian(x_fit, *fit["popt"])
            plt.plot(x_fit * pixel_size, y_fit, "-", label=label)

            half_max = fit["background"] + fit["amplitude"] / 2
            left = fit["mean_physic"] - fit["fwhm_physic"] / 2
            right = fit["mean_physic"] + fit["fwhm_physic"] / 2
            plt.axhline(half_max, linestyle="--", alpha=0.5)
            plt.axvline(left, linestyle=":", alpha=0.5)
            plt.axvline(right, linestyle=":", alpha=0.5)

        plt.xlabel(f"Distance along segment ({UNIT})")
        plt.ylabel("Intensity")
        plt.legend()

        plt.tight_layout()
        plt.savefig(output_plot)
        plt.close()

    return results


def apply_sparrow(
    image: Path,
    p1: tuple[int, int],
    p2: tuple[int, int],
    pixel_size: float,
    output_folder: Path,
    visual: bool,
):
    img = load_image(image)
    point1, point2 = np.asarray(p1, float), np.asarray(p2, float)
    sep = np.linalg.norm(point2 - point1)
    if sep == 0:
        raise ValueError("the two points must be different")
    u = (point2 - point1) / sep

    # Extend the segment past peaks
    margin = max(2.0, 0.5 * sep)
    start, end = point1 - margin * u, point2 + margin * u

    # Profile (profile_line expects row,col; points are x,y)
    prof = profile_line(
        img,
        start[::-1],
        end[::-1],
        linewidth=3,
        order=3,
        mode="nearest",
        reduce_func=np.mean,
    )
    total = sep + 2 * margin
    pos = np.linspace(0, total, len(prof))  # distance from `start`
    t1, t2 = margin, margin + sep  # emitter positions on that axis
    mid = (t1 + t2) / 2

    bg = float(np.median(img))
    prof = prof - bg

    # Try to refine the two emitter positions, same as Rayleigh
    peaks, _ = find_peaks(prof)
    search_radius_px = max(1.5, 0.35 * sep)

    def best_peak(lo, hi):
        cand = peaks[(pos[peaks] >= lo) & (pos[peaks] <= hi)]
        return cand[np.argmax(prof[cand])] if len(cand) else None

    i1 = best_peak(t1 - search_radius_px, min(mid, t1 + search_radius_px))
    i2 = best_peak(max(mid, t2 - search_radius_px), t2 + search_radius_px)
    two_peaks_found = i1 is not None and i2 is not None and i1 != i2

    # Sparrow criterion: the curvature of the profile at the midpoint between
    # the two emitters. A dip (curvature > 0, concave up) means the sources
    # are still resolved; a single merged hump (curvature < 0) means they
    # are not. Unlike Rayleigh, no arbitrary threshold is needed. If the two
    # peaks have already merged into a single blob (no dip left to detect),
    # fall back to the geometric midpoint between the requested points.
    im = (i1 + i2) // 2 if two_peaks_found else int(np.argmin(np.abs(pos - mid)))

    window = max(2, round(0.1 * sep))
    lo, hi = max(0, im - window), min(len(prof) - 1, im + window)
    coeffs = np.polyfit(pos[lo : hi + 1], prof[lo : hi + 1], 2)
    curvature = float(2 * coeffs[0])
    midpoint_intensity = float(np.polyval(coeffs, pos[im]))

    nan = float("nan")
    sparrow_result = {
        "resolved": curvature > 0,
        "curvature": curvature,
        "peak_distance": float(pos[i2] - pos[i1]) * pixel_size
        if two_peaks_found
        else nan,
        "peak_distance_px": float(pos[i2] - pos[i1]) if two_peaks_found else nan,
        "peak1_pos": float(pos[i1]) if two_peaks_found else nan,
        "peak2_pos": float(pos[i2]) if two_peaks_found else nan,
        "midpoint_pos": float(pos[im]),
        "peak1_intensity": float(prof[i1]) if two_peaks_found else nan,
        "peak2_intensity": float(prof[i2]) if two_peaks_found else nan,
        "midpoint_intensity": midpoint_intensity,
        "background": bg,
        "positions": pos,
        "profile": prof,
    }

    results = {
        "Image": image.stem,
        "Pixel Size": pixel_size,
        "Resolved": sparrow_result["resolved"],
        "Curvature": sparrow_result["curvature"],
        f"Peak Distance ({UNIT})": sparrow_result["peak_distance"],
        "Peak Distance (px)": sparrow_result["peak_distance_px"],
    }

    if visual:
        output_image_folder: Path = output_folder / image.stem
        output_image_folder.mkdir(parents=True, exist_ok=True)
        output_plot = output_image_folder / "sparrow_plot.png"

        # Plot
        pos_physic = sparrow_result["positions"] * pixel_size

        plt.figure(figsize=(7, 4))
        plt.plot(
            pos_physic, sparrow_result["profile"], "o-", markersize=4, label="Profile"
        )

        if not np.isnan(sparrow_result["peak1_pos"]):
            peaks_x = [
                sparrow_result["peak1_pos"] * pixel_size,
                sparrow_result["peak2_pos"] * pixel_size,
            ]
            peaks_y = [
                sparrow_result["peak1_intensity"],
                sparrow_result["peak2_intensity"],
            ]
            plt.plot(peaks_x, peaks_y, "^", markersize=8, label="Peaks")

        plt.plot(
            sparrow_result["midpoint_pos"] * pixel_size,
            sparrow_result["midpoint_intensity"],
            "s",
            markersize=8,
            label="Midpoint",
        )

        plt.xlabel(f"Distance along segment ({UNIT})")
        plt.ylabel("Intensity (background subtracted)")
        plt.legend()

        plt.tight_layout()
        plt.savefig(output_plot)
        plt.close()

    return results


def apply_rayleigh(
    image: Path,
    p1: tuple[int, int],
    p2: tuple[int, int],
    threshold: float,
    pixel_size: float,
    output_folder: Path,
    visual: bool,
):
    img = load_image(image)
    point1, point2 = np.asarray(p1, float), np.asarray(p2, float)
    sep = np.linalg.norm(point2 - point1)
    if sep == 0:
        raise ValueError("the two points must be different")
    u = (point2 - point1) / sep

    # Extend the segment past peaks
    margin = max(2.0, 0.5 * sep)
    start, end = point1 - margin * u, point2 + margin * u

    # Profile (profile_line expects row,col; points are x,y)
    prof = profile_line(
        img,
        start[::-1],
        end[::-1],
        linewidth=3,
        order=3,
        mode="nearest",
        reduce_func=np.mean,
    )
    total = sep + 2 * margin
    pos = np.linspace(0, total, len(prof))  # distance from `start`
    t1, t2 = margin, margin + sep  # emitter positions on that axis

    bg = float(np.median(img))
    prof = prof - bg

    # Peak detection
    peaks, _ = find_peaks(prof)
    search_radius_px = max(1.5, 0.35 * sep)
    mid = (t1 + t2) / 2

    def best_peak(lo, hi):
        cand = peaks[(pos[peaks] >= lo) & (pos[peaks] <= hi)]
        return cand[np.argmax(prof[cand])] if len(cand) else None

    i1 = best_peak(t1 - search_radius_px, min(mid, t1 + search_radius_px))
    i2 = best_peak(max(mid, t2 - search_radius_px), t2 + search_radius_px)

    nan = float("nan")
    if i1 is None or i2 is None or i1 == i2:
        # Fewer than two distinct peaks -> a single blob: no dip
        rayleigh_result = {
            "resolved": False,
            "dip_ratio": 1.0,
            "threshold": threshold,
            "peak_distance": nan,
            "peak_distance_px": nan,
            "peak1_pos": nan,
            "peak2_pos": nan,
            "valley_pos": nan,
            "peak1_intensity": nan,
            "peak2_intensity": nan,
            "valley_intensity": nan,
            "background": bg,
            "positions": pos,
            "profile": prof,
        }
    else:
        iv = i1 + int(np.argmin(prof[i1 : i2 + 1]))
        lower_peak = min(prof[i1], prof[i2])
        if lower_peak <= 0:
            raise ValueError(
                "Peak is not above the background; check points/background."
            )
        ratio = float(prof[iv] / lower_peak)

        dist_px = float(pos[i2] - pos[i1])
        rayleigh_result = {
            "resolved": ratio <= threshold,
            "dip_ratio": ratio,
            "threshold": threshold,
            "peak_distance": dist_px * pixel_size,
            "peak_distance_px": dist_px,
            "peak1_pos": float(pos[i1]),
            "peak2_pos": float(pos[i2]),
            "valley_pos": float(pos[iv]),
            "peak1_intensity": float(prof[i1]),
            "peak2_intensity": float(prof[i2]),
            "valley_intensity": float(prof[iv]),
            "background": bg,
            "positions": pos,
            "profile": prof,
        }

    results = {
        "Image": image.stem,
        "Pixel Size": pixel_size,
        "Resolved": rayleigh_result["resolved"],
        "Dip Ratio": rayleigh_result["dip_ratio"],
        "Rayleigh Threshold": rayleigh_result["threshold"],
        f"Peak Distance ({UNIT})": rayleigh_result["peak_distance"],
        "Peak Distance (px)": rayleigh_result["peak_distance_px"],
    }

    if visual:
        output_image_folder: Path = output_folder / image.stem
        output_image_folder.mkdir(parents=True, exist_ok=True)
        output_plot = output_image_folder / "rayleigh_plot.png"

        # Plot
        pos_physic = rayleigh_result["positions"] * pixel_size

        plt.figure(figsize=(7, 4))
        plt.plot(
            pos_physic, rayleigh_result["profile"], "o-", markersize=4, label="Profile"
        )

        if not np.isnan(rayleigh_result["peak1_pos"]):
            peaks_x = [
                rayleigh_result["peak1_pos"] * pixel_size,
                rayleigh_result["peak2_pos"] * pixel_size,
            ]
            peaks_y = [
                rayleigh_result["peak1_intensity"],
                rayleigh_result["peak2_intensity"],
            ]
            plt.plot(peaks_x, peaks_y, "^", markersize=8, label="Peaks")
            plt.plot(
                rayleigh_result["valley_pos"] * pixel_size,
                rayleigh_result["valley_intensity"],
                "v",
                markersize=8,
                label="Valley",
            )
            plt.axhline(
                min(peaks_y) * threshold,
                linestyle="--",
                label=f"Rayleigh threshold ({threshold:.3f})",
            )

        plt.xlabel(f"Distance along segment ({UNIT})")
        plt.ylabel("Intensity (background subtracted)")
        plt.legend()

        plt.tight_layout()
        plt.savefig(output_plot)
        plt.close()

    return results


def apply_fwhm(
    image: Path,
    p1: tuple[int, int],
    p2: tuple[int, int],
    pixel_size: float,
    output_folder: Path,
    visual: bool,
):
    img = load_image(image)
    x1, y1 = p1
    x2, y2 = p2

    # profile_line expects row,col; points are x,y
    profile = profile_line(img, (y1, x1), (y2, x2), linewidth=1, reduce_func=np.mean)
    segment_length = np.hypot(x2 - x1, y2 - y1)
    distance = np.linspace(0, segment_length, len(profile))
    distance_physic = distance * pixel_size

    fit = fit_gaussian_fwhm(distance, profile, pixel_size)

    results = {
        "Image": image.stem,
        "Pixel Size": pixel_size,
        f"FWHM ({UNIT})": fit["fwhm_physic"],
        f"FWHM Uncertainty ({UNIT})": fit["fwhm_uncertainty_physic"],
        "R2": fit["r_squared"],
    }

    if visual:
        output_image_folder: Path = output_folder / image.stem
        output_image_folder.mkdir(parents=True, exist_ok=True)
        output_plot = output_image_folder / "fwhm_plot.png"

        # Plot
        x_fit = np.linspace(distance.min(), distance.max(), 500)
        y_fit = gaussian(x_fit, *fit["popt"])
        half_max = fit["background"] + fit["amplitude"] / 2
        left = fit["mean_physic"] - fit["fwhm_physic"] / 2
        right = fit["mean_physic"] + fit["fwhm_physic"] / 2
        x_fit_physic = x_fit * pixel_size

        plt.figure(figsize=(7, 4))
        plt.plot(distance_physic, profile, "o", markersize=4, label="Measured")
        plt.plot(x_fit_physic, y_fit, "-", label="Gaussian fit")

        plt.axhline(half_max, linestyle="--", label="Half maximum")
        plt.axvline(left, linestyle=":")
        plt.axvline(right, linestyle=":")

        plt.xlabel(f"Distance along segment ({UNIT})")
        plt.ylabel("Intensity")
        plt.legend()

        plt.tight_layout()
        plt.savefig(output_plot)
        plt.close()

    return results


def apply_decorrelation(
    image: Path,
    pixel_size: float,
    output_folder: Path,
    visual: bool,
):
    img = load_image(image)

    # rmin/rmax are the fraction of the frequency spectrum considered (0 to 1,
    # i.e. the full range); this is the same default used in NanoPyx's own
    # example notebooks.
    decorr = DecorrAnalysis(pixel_size=pixel_size, units=UNIT, rmin=0.0, rmax=1.0)
    decorr.run_analysis(img)

    results = {
        "Image1": image.stem,
        "Pixel Size": pixel_size,
        f"Decorrelation Resolution ({UNIT})": decorr.resolution,
    }

    if visual:
        output_image_folder = output_folder / image.stem
        output_image_folder.mkdir(parents=True, exist_ok=True)
        output_figure = output_image_folder / "decorr_curve.png"
        plot_array = decorr.plot_results()
        plt.imsave(output_figure, plot_array)

    return results


def apply_frc(
    image1: Path,
    image2: Path | None,
    pixel_size: float,
    output_folder: Path,
    visual: bool,
):
    img1 = load_image(image1)
    img1 = frc.util.square_image(img1, add_padding=False)
    img1 = frc.util.apply_tukey(img1)

    img2 = None
    if image2 is not None:
        img2 = load_image(image2)
        img2 = frc.util.square_image(img2, add_padding=False)
        img2 = frc.util.apply_tukey(img2)

    if img2 is not None:
        frc_curve = frc.two_frc(img1, img2)
    else:
        frc_curve = frc.one_frc(img1, method=1)

    img_size = img1.shape[0]
    xs_pix = np.arange(len(frc_curve)) / img_size
    xs_nm_freq = xs_pix / pixel_size
    try:
        frc_res, res_y, thres = frc.frc_res(xs_nm_freq, frc_curve, img_size)
    except frc.NoIntersectionException as e:
        frc_res, res_y, thres = None, None, None
        print(
            f"No intersection has been found. Check the plot for further troubleshooting: {e}"
        )

    results = {
        "Image1": image1.stem,
        "Image2": image2.stem if image2 is not None else image2,
        "Pixel Size": pixel_size,
        f"Resolution ({UNIT})": frc_res,
    }

    if visual:
        output_image_folder = output_folder / image1.stem
        output_image_folder.mkdir(parents=True, exist_ok=True)
        if image2 is not None:
            output_figure = output_image_folder / (image2.stem + "_frc_curve.png")
        else:
            output_figure = output_image_folder / "frc_curve.png"
        plt.figure()
        plt.plot(xs_nm_freq, frc_curve, label="FRC curve")
        if thres is not None:
            plt.plot(xs_nm_freq, thres(xs_nm_freq), linestyle="--", label="Threshold")
        if frc_res is not None:
            plt.axvline(
                1 / frc_res,
                color="red",
                linestyle=":",
                label=f"Resolution ({frc_res:.1f} {UNIT})",
            )
        plt.title(f"Fourier Ring Correlation (FRC)")
        plt.xlabel(f"Spatial frequency (1/{UNIT})")
        plt.ylabel("FRC")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_figure, dpi=300)
        plt.close()

    return results


def save_zoomed_image(
    output_folder: Path,
    image_name: Path,
    img: np.ndarray,
    p1: tuple[int, int],
    p2: tuple[int, int],
):
    x1, y1 = p1
    x2, y2 = p2

    output_image_folder: Path = output_folder / image_name.stem
    output_image_folder.mkdir(parents=True, exist_ok=True)
    output_image = output_image_folder / "image.png"

    # Image
    height, width = img.shape[:2]
    dpi = 100
    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)
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
        linewidth=1.5,
    )

    ax.add_patch(rect)

    ax_zoom = inset_axes(ax, width="40%", height="40%", loc="lower right", borderpad=1)
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


def merge_dictionary(dict_list: list[dict]):
    merged_dict = {}
    for d in dict_list:
        if d is None:
            continue
        for key, value in d.items():
            if key not in merged_dict:
                merged_dict[key] = value
    return merged_dict


def save_results(output_folder: Path, results: list, filename: str):
    output_folder.mkdir(parents=True, exist_ok=True)
    output_csv: Path = output_folder / filename

    new_row = pd.DataFrame([merge_dictionary(results)])

    if output_csv.exists():
        df = pd.concat(
            [pd.read_csv(output_csv, float_precision="round_trip"), new_row],
            ignore_index=True,
        )
    else:
        df = new_row

    df = df.drop_duplicates()
    df.to_csv(output_csv, index=False)


def apply_automatic_criterions(args: Namespace):
    results = [
        apply_frc(args.image1, args.image2, args.pixel_size, args.out, args.visual),
        apply_decorrelation(args.image1, args.pixel_size, args.out, args.visual),
    ]
    save_results(args.out, results, "automatic.csv")


def apply_manual_criterions(args: Namespace):
    match args.method:
        case "fwhm":
            result = apply_fwhm(
                args.image, args.p1, args.p2, args.pixel_size, args.out, args.visual
            )
        case "rayleigh":
            result = apply_rayleigh(
                args.image,
                args.p1,
                args.p2,
                args.threshold,
                args.pixel_size,
                args.out,
                args.visual,
            )
        case "sparrow":
            result = apply_sparrow(
                args.image, args.p1, args.p2, args.pixel_size, args.out, args.visual
            )
        case "houston":
            result = apply_houston(
                args.image, args.p1, args.p2, args.pixel_size, args.out, args.visual
            )
        case _:
            raise ValueError(f"Unknown manual method: {args.method}")

    save_results(args.out, [result], "manual.csv")

    if args.visual:
        save_zoomed_image(
            args.out, args.image, load_image(args.image), args.p1, args.p2
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparser = parser.add_subparsers(dest="command", required=True)

    # Automatic criterions (FRC, ...)
    p_automatic = subparser.add_parser(
        "automatic", help="Automatic criterions (FRC, Decorrelation)"
    )
    p_automatic.add_argument(
        "--image1", type=Path, required=True, help="Base image to use"
    )
    p_automatic.add_argument(
        "--image2", type=Path, default=None, help="Image to compare against if any"
    )
    p_automatic.add_argument(
        "--threshold",
        type=float,
        default=1 / 7,
        help="Threshold for FRC method. Default: 1/7",
    )
    p_automatic.add_argument(
        "--pixel_size", type=float, required=True, help="Pixel size"
    )
    p_automatic.add_argument(
        "--diff_limit", type=float, default=None, help="Abbe's diffraction limit"
    )
    p_automatic.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output folder for log (automatic.csv) and figures",
    )
    p_automatic.add_argument(
        "--visual", action="store_true", help="Generate and save figure"
    )
    p_automatic.set_defaults(func=apply_automatic_criterions)

    # Manual criterions (FWHM, Rayleigh, ...)
    p_manual = subparser.add_parser(
        "manual", help="Manual criterions (FWHM, Rayleigh, Sparrow, Houston)"
    )
    p_manual.add_argument(
        "--method",
        choices=["fwhm", "rayleigh", "sparrow", "houston"],
        required=True,
        help="Which manual criterion to apply",
    )
    p_manual.add_argument("--image", type=Path, required=True, help="Image to use")
    p_manual.add_argument(
        "--p1", type=parse_point, required=True, help="Beginning of segment (x,y)"
    )
    p_manual.add_argument(
        "--p2", type=parse_point, required=True, help="End of segment (x,y)"
    )
    p_manual.add_argument(
        "--threshold",
        type=float,
        default=RAYLEIGH_THRESHOLD,
        help=f"Dip ratio threshold for Rayleigh criterion. Default: {RAYLEIGH_THRESHOLD}",
    )
    p_manual.add_argument("--pixel_size", type=float, required=True, help="Pixel size")
    p_manual.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output folder for log (manual.csv) and figures",
    )
    p_manual.add_argument(
        "--visual", action="store_true", help="Generate and save figures/images"
    )
    p_manual.set_defaults(func=apply_manual_criterions)

    args = parser.parse_args()
    args.func(args)
