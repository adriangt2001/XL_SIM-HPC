"""
line_profile_resolution.py

Two ways to measure resolution from a microscopy image, as commonly reported
in super-resolution papers:

1. LINE PROFILE (local): click two points across a point source (PSF) or
   across two adjacent structures, fit Gaussian(s) to the intensity profile,
   and report FWHM or peak separation. Interactive class: InteractiveLineProfiler.

2. FRC - Fourier Ring Correlation (global, whole-image): compares two
   independent images/reconstructions of the same field of view in Fourier
   space and reports the spatial frequency beyond which they no longer agree.
   See the module docstring of compute_frc() for the theory. Convenience
   function: measure_frc().

Requirements: numpy, torch, scipy, matplotlib, Pillow (and tifffile for .tif)

--------------------------------------------------------------------------
JUPYTER NOTEBOOK USAGE (interactive line profile)
--------------------------------------------------------------------------
    # In the first cell (only needed once per kernel):
    #   pip install ipympl
    %matplotlib widget

    from line_profile_resolution import load_image, InteractiveLineProfiler
    from pathlib import Path

    img = load_image(Path("cell.tif"))

    lp = InteractiveLineProfiler(img, mode="single-peak", pixel_size=100, unit="nm")
    lp.show()

    # -> A figure appears. Click two points on the LEFT panel to define the
    #    line (e.g. across a bead, or across two adjacent puncta).
    # -> Click "Measure". The fit + FWHM (or peak separation) appear on the
    #    RIGHT panel and are printed below the cell.
    # -> Click "Reset" to clear your points and pick a new line.
    # -> Results are always available afterwards as a dict: lp.results

    # For two adjacent structures instead of a single bead:
    lp2 = InteractiveLineProfiler(img, mode="two-peak", pixel_size=100, unit="nm")
    lp2.show()

If %matplotlib widget doesn't work (ipympl not installed / older Jupyter),
`%matplotlib notebook` is a (deprecated but often still working) fallback.
Plain `%matplotlib inline` will NOT work - it produces static, non-clickable
figures.

--------------------------------------------------------------------------
JUPYTER / SCRIPT USAGE (FRC, whole-image resolution)
--------------------------------------------------------------------------
    from line_profile_resolution import load_image, measure_frc
    from pathlib import Path

    img = load_image(Path("cell.tif"))

    # If you only have one image, it will be split into two independent
    # noisy realizations for you (see random_split_image for caveats):
    freq, frc_curve, resolution = measure_frc(img, pixel_size=100, unit="nm")

    # If you have two genuinely independent reconstructions of the same
    # field of view (e.g. odd/even-frame reconstructions from an SMLM
    # movie), pass both - this is the more rigorous option:
    img_odd  = load_image(Path("recon_odd_frames.tif"))
    img_even = load_image(Path("recon_even_frames.tif"))
    freq, frc_curve, resolution = measure_frc(img_odd, img_even, pixel_size=100, unit="nm")

--------------------------------------------------------------------------
COMMAND-LINE USAGE (non-interactive, for scripting/batch use)
--------------------------------------------------------------------------
    # Line profile with explicit coordinates:
    python line_profile_resolution.py line --image cell.tif --p1 120,80 --p2 180,80 \
        --mode single-peak --pixel-size 100 --unit nm

    # FRC from a single image (auto-split):
    python line_profile_resolution.py frc --image cell.tif --pixel-size 100 --unit nm

    # FRC from two independent images:
    python line_profile_resolution.py frc --image recon_odd.tif --image2 recon_even.tif \
        --pixel-size 100 --unit nm
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.widgets import Button
from scipy.optimize import curve_fit

# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

def load_image(path: Path) -> np.ndarray:
    """Load an image as a 2D float32 numpy array (grayscale)."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in (".tif", ".tiff"):
        try:
            import tifffile
            img = tifffile.imread(path)
        except ImportError:
            raise ImportError("Reading .tif requires `pip install tifffile`.")
    else:
        from PIL import Image
        img = np.array(Image.open(path).convert("F"))  # 32-bit float grayscale

    img = np.asarray(img).astype(np.float32)

    if img.ndim == 3:
        # Collapse channels (e.g. RGB) by averaging. If you have a specific
        # fluorescence channel to isolate, index it here instead, e.g. img[..., 1].
        img = img.mean(axis=-1)
    elif img.ndim != 2:
        raise ValueError(f"Expected a 2D (or 2D+channel) image, got shape {img.shape}")

    return img


# ---------------------------------------------------------------------------
# Line profile extraction (bilinear interpolation via torch.grid_sample)
# ---------------------------------------------------------------------------

def extract_line_profile(img: np.ndarray, p1, p2, n_samples=None, device="cpu"):
    """
    Sample intensities along the line from p1 to p2 (in (x, y) pixel
    coordinates) using bilinear interpolation via torch.grid_sample.

    Returns:
        distances: 1D array of distance along the line (in pixels), from 0
        values: 1D array of interpolated intensities
    """
    x1, y1 = p1
    x2, y2 = p2
    length_px = float(np.hypot(x2 - x1, y2 - y1))

    if n_samples is None:
        n_samples = max(int(np.ceil(length_px)) * 2, 10)

    t = np.linspace(0.0, 1.0, n_samples)
    xs = x1 + t * (x2 - x1)
    ys = y1 + t * (y2 - y1)

    H, W = img.shape
    img_t = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0).to(device)

    xs_norm = (xs / (W - 1)) * 2 - 1
    ys_norm = (ys / (H - 1)) * 2 - 1
    grid = torch.from_numpy(np.stack([xs_norm, ys_norm], axis=-1)).float()
    grid = grid.view(1, 1, n_samples, 2).to(device)

    sampled = torch.nn.functional.grid_sample(
        img_t, grid, mode="bilinear", align_corners=True, padding_mode="border"
    )
    values = sampled.view(-1).cpu().numpy()
    distances = t * length_px

    return distances, values


# ---------------------------------------------------------------------------
# Gaussian fitting
# ---------------------------------------------------------------------------

def gaussian(x, amp, mu, sigma, offset):
    return offset + amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def two_gaussians(x, amp1, mu1, sigma1, amp2, mu2, sigma2, offset):
    return (
        offset
        + amp1 * np.exp(-0.5 * ((x - mu1) / sigma1) ** 2)
        + amp2 * np.exp(-0.5 * ((x - mu2) / sigma2) ** 2)
    )


def fwhm_from_sigma(sigma):
    return 2.0 * np.sqrt(2.0 * np.log(2)) * sigma


def fit_single_peak(distances, values):
    offset0 = np.percentile(values, 10)
    amp0 = values.max() - offset0
    mu0 = distances[np.argmax(values)]
    span = distances[-1] - distances[0]
    sigma0 = span / 6

    p0 = [amp0, mu0, sigma0, offset0]
    bounds = (
        [0, distances[0], 1e-3, -np.inf],
        [np.inf, distances[-1], span, np.inf],
    )
    popt, pcov = curve_fit(gaussian, distances, values, p0=p0, bounds=bounds, maxfev=10000)
    perr = np.sqrt(np.diag(pcov))
    return popt, perr


def fit_two_peaks(distances, values):
    offset0 = np.percentile(values, 10)
    amp0 = values.max() - offset0
    span = distances[-1] - distances[0]

    mid = len(distances) // 2
    mu1_0 = distances[:mid][np.argmax(values[:mid])] if mid > 0 else distances[0]
    mu2_0 = distances[mid:][np.argmax(values[mid:])] if mid < len(distances) else distances[-1]
    sigma0 = span / 12

    p0 = [amp0, mu1_0, sigma0, amp0, mu2_0, sigma0, offset0]
    lo, hi = distances[0], distances[-1]
    bounds = (
        [0, lo, 1e-3, 0, lo, 1e-3, -np.inf],
        [np.inf, hi, span, np.inf, hi, span, np.inf],
    )
    popt, pcov = curve_fit(two_gaussians, distances, values, p0=p0, bounds=bounds, maxfev=20000)
    perr = np.sqrt(np.diag(pcov))
    return popt, perr


# ---------------------------------------------------------------------------
# Interactive line profiler (Jupyter / any interactive matplotlib backend)
# ---------------------------------------------------------------------------

class InteractiveLineProfiler:
    """
    Click-to-select line profile tool.

    Instantiate with an image, call .show(), click two points on the left
    panel, then click "Measure". See module docstring for a full example.
    """

    def __init__(self, img, mode="single-peak", pixel_size=None, unit="nm",
                 n_samples=None, device="cpu"):
        self.img = img
        self.mode = mode
        self.pixel_size = pixel_size
        self.unit = unit
        self.n_samples = n_samples
        self.device = device
        self.points = []
        self.results = None
        self._point_artists = []
        self._line_artist = None

        self.fig, (self.ax_img, self.ax_profile) = plt.subplots(1, 2, figsize=(11, 5))
        self.ax_img.imshow(img, cmap="gray")
        self.ax_img.set_title("Click 2 points, then press Measure")
        self.ax_profile.axis("off")
        self.ax_profile.set_title("Profile will appear here")

        self.fig.subplots_adjust(bottom=0.15)
        ax_measure = self.fig.add_axes([0.35, 0.02, 0.13, 0.06])
        ax_reset = self.fig.add_axes([0.52, 0.02, 0.13, 0.06])
        self.btn_measure = Button(ax_measure, "Measure")
        self.btn_reset = Button(ax_reset, "Reset")
        self.btn_measure.on_clicked(self._on_measure)
        self.btn_reset.on_clicked(self._on_reset)

        self.cid = self.fig.canvas.mpl_connect("button_press_event", self._on_click)

    def show(self):
        """Display the figure. In Jupyter with %matplotlib widget this
        returns the live interactive widget; in a script it opens a window."""
        return self.fig

    def _on_click(self, event):
        if event.inaxes != self.ax_img or event.xdata is None:
            return
        if len(self.points) >= 2:
            return
        self.points.append((event.xdata, event.ydata))
        artist, = self.ax_img.plot(event.xdata, event.ydata, "r+",
                                    markersize=12, markeredgewidth=2)
        self._point_artists.append(artist)
        if len(self.points) == 2:
            (x1, y1), (x2, y2) = self.points
            self._line_artist, = self.ax_img.plot([x1, x2], [y1, y2], "r-", linewidth=1.5)
        self.fig.canvas.draw_idle()

    def _on_reset(self, event):
        self.points = []
        for artist in self._point_artists:
            artist.remove()
        self._point_artists = []
        if self._line_artist is not None:
            self._line_artist.remove()
            self._line_artist = None
        self.results = None
        self.ax_img.set_title("Click 2 points, then press Measure")
        self.ax_profile.clear()
        self.ax_profile.axis("off")
        self.ax_profile.set_title("Profile will appear here")
        self.fig.canvas.draw_idle()

    def _on_measure(self, event):
        if len(self.points) != 2:
            self.ax_img.set_title("Need exactly 2 points - click on the image first")
            self.fig.canvas.draw_idle()
            return

        p1, p2 = self.points
        distances, values = extract_line_profile(
            self.img, p1, p2, n_samples=self.n_samples, device=self.device
        )
        scale = self.pixel_size if self.pixel_size is not None else 1.0
        unit = self.unit if self.pixel_size is not None else "px"
        d = distances * scale

        self.ax_profile.clear()
        self.ax_profile.axis("on")
        self.ax_profile.plot(d, values, "k.", markersize=4, label="data")

        try:
            if self.mode == "single-peak":
                popt, perr = fit_single_peak(d, values)
                amp, mu, sigma, offset = popt
                fwhm = fwhm_from_sigma(sigma)
                fwhm_err = fwhm_from_sigma(perr[2])
                xfit = np.linspace(d.min(), d.max(), 500)
                self.ax_profile.plot(xfit, gaussian(xfit, *popt), "r-", label="Gaussian fit")
                self.ax_profile.axvline(mu - fwhm / 2, color="b", linestyle="--")
                self.ax_profile.axvline(mu + fwhm / 2, color="b", linestyle="--")
                self.results = {"mode": "single-peak", "center": mu,
                                 "fwhm": fwhm, "fwhm_err": fwhm_err, "unit": unit}
                print(f"Peak center: {mu:.2f} {unit}")
                print(f"FWHM (resolution estimate): {fwhm:.2f} +/- {fwhm_err:.2f} {unit}")
            else:
                popt, perr = fit_two_peaks(d, values)
                amp1, mu1, sigma1, amp2, mu2, sigma2, offset = popt
                separation = abs(mu2 - mu1)
                sep_err = float(np.sqrt(perr[1] ** 2 + perr[4] ** 2))
                xfit = np.linspace(d.min(), d.max(), 500)
                self.ax_profile.plot(xfit, two_gaussians(xfit, *popt), "r-", label="Double Gaussian fit")
                self.ax_profile.axvline(mu1, color="b", linestyle="--")
                self.ax_profile.axvline(mu2, color="g", linestyle="--")
                self.results = {"mode": "two-peak", "center1": mu1, "center2": mu2,
                                 "separation": separation, "separation_err": sep_err, "unit": unit}
                print(f"Peak 1 center: {mu1:.2f} {unit}")
                print(f"Peak 2 center: {mu2:.2f} {unit}")
                print(f"Peak separation (resolution estimate): {separation:.2f} +/- {sep_err:.2f} {unit}")
        except RuntimeError as e:
            self.ax_profile.set_title("Fit failed - try a cleaner/shorter line")
            print(f"Gaussian fit did not converge: {e}")
            self.fig.canvas.draw_idle()
            return

        self.ax_profile.set_xlabel(f"Distance along line ({unit})")
        self.ax_profile.set_ylabel("Intensity")
        self.ax_profile.legend()
        self.ax_profile.set_title("Intensity profile & fit")
        self.fig.canvas.draw_idle()


# ---------------------------------------------------------------------------
# FRC - Fourier Ring Correlation (whole-image resolution)
# ---------------------------------------------------------------------------
#
# FRC compares two statistically INDEPENDENT images (or reconstructions) of
# the SAME underlying sample, in Fourier space, ring by ring (i.e. shell by
# shell of constant spatial frequency). At low spatial frequencies (coarse
# features) both images agree well because real signal dominates, so the
# correlation is close to 1. At high spatial frequencies (fine detail) the
# two images increasingly disagree because each has its own independent
# noise, so the correlation decays toward 0. The spatial frequency at which
# the FRC curve drops below a fixed threshold (1/7 is the standard criterion
# introduced for SMLM/super-resolution by Nieuwenhuizen et al. 2013,
# Nat. Methods) is taken as the resolution limit: beyond that frequency,
# apparent detail is not reproducible between independent measurements and
# is therefore considered noise rather than real structure. The resolution
# value reported is 1 / (that spatial frequency), in the same physical units
# as pixel_size.
#
# Unlike a line-profile FWHM, FRC uses information from the entire image and
# does not require you to find an isolated point source or a pair of
# adjacent features - which is why it's become the standard whole-image
# resolution metric in recent SMLM/STORM/PALM papers. Its main requirement
# is that you have two genuinely independent realizations of the same
# field of view (e.g. reconstructions from odd vs even frames of an SMLM
# movie, or two independent acquisitions). If you only have a single image,
# random_split_image() below gives an approximate substitute.
# ---------------------------------------------------------------------------

def _spatial_frequency_grid(shape, pixel_size=1.0):
    H, W = shape
    fy = np.fft.fftfreq(H, d=pixel_size)
    fx = np.fft.fftfreq(W, d=pixel_size)
    FX, FY = np.meshgrid(fx, fy)
    return np.sqrt(FX ** 2 + FY ** 2)


def compute_frc(img1, img2, pixel_size=1.0, n_bins=None):
    """
    Compute the FRC curve between two independent images of the same field
    of view.

    Args:
        img1, img2: 2D arrays, same shape, independent realizations of the
            same underlying sample.
        pixel_size: physical size of one pixel (e.g. nm). Spatial frequency
            is then reported in cycles per that unit. Leave at 1.0 for
            frequency in cycles/pixel.
        n_bins: number of radial frequency bins. Defaults to min(H, W) // 2.

    Returns:
        spatial_freq: 1D array of spatial-frequency bin centers
        frc: 1D array of FRC values (unitless, ideally in [-1, 1] but noisy
            near 0 at high frequency)
    """
    if img1.shape != img2.shape:
        raise ValueError("img1 and img2 must have the same shape")

    F1 = np.fft.fft2(img1)
    F2 = np.fft.fft2(img2)

    freq_radius = _spatial_frequency_grid(img1.shape, pixel_size=pixel_size)

    # A square FFT grid only has full angular coverage of a CIRCLE out to the
    # Nyquist frequency (0.5/pixel_size); beyond that, only the four corner
    # regions contribute, each ring having very few samples. Those bins are
    # extremely noisy (can spuriously spike up or down) and aren't physically
    # meaningful, so we exclude them entirely rather than lump them into the
    # last bin.
    nyquist = 0.5 / pixel_size
    max_freq = nyquist

    H, W = img1.shape
    if n_bins is None:
        n_bins = min(H, W) // 2

    bin_edges = np.linspace(0, max_freq, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    freq_flat = freq_radius.ravel()
    in_range = freq_flat <= nyquist

    bin_idx = np.digitize(freq_flat[in_range], bin_edges) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    F1_flat = F1.ravel()[in_range]
    F2_flat = F2.ravel()[in_range]
    cross = np.real(F1_flat * np.conj(F2_flat))
    p1 = np.abs(F1_flat) ** 2
    p2 = np.abs(F2_flat) ** 2

    numerator = np.bincount(bin_idx, weights=cross, minlength=n_bins)
    denom1 = np.bincount(bin_idx, weights=p1, minlength=n_bins)
    denom2 = np.bincount(bin_idx, weights=p2, minlength=n_bins)

    with np.errstate(invalid="ignore", divide="ignore"):
        frc = numerator / np.sqrt(denom1 * denom2)

    return bin_centers, frc


def frc_resolution(spatial_freq, frc, threshold=1.0 / 7.0, min_consistent_bins=8):
    """
    Find the resolution (1 / spatial frequency, in the same units as
    1/pixel_size) at which the FRC curve drops below `threshold` and stays
    there for at least `min_consistent_bins` consecutive bins.

    threshold=1/7 (~0.143) is the fixed criterion from Nieuwenhuizen et al.
    2013 and the most common choice in recent super-resolution papers.

    This is deliberately NOT "stays below forever": even within the
    Nyquist-limited range compute_frc() now uses, the last few bins
    (thinnest rings, fewest samples) can still show a one-off noise spike
    above threshold without that being a real second crossing. If you see a
    real, SUSTAINED rise back above threshold rather than a brief spike (i.e.
    it persists for longer than min_consistent_bins), that's telling you
    something real (periodic/gridding artifacts, too few independent samples
    at high frequency, etc.) - don't just raise this parameter to hide it,
    look at the plot.
    """
    frc_smooth = np.copy(frc)
    if len(frc_smooth) > 5:
        kernel = np.ones(5) / 5
        frc_smooth = np.convolve(frc_smooth, kernel, mode="same")

    n = len(frc_smooth)
    below = frc_smooth < threshold
    window = max(1, min(min_consistent_bins, n))

    crossing_idx = None
    for i in range(n):
        w = min(window, n - i)
        if np.all(below[i : i + w]):
            crossing_idx = i
            break

    if crossing_idx is None or crossing_idx == 0:
        return None

    freq_at_crossing = spatial_freq[crossing_idx]
    if freq_at_crossing <= 0:
        return None
    return 1.0 / freq_at_crossing


def random_split_image(img, seed=None):
    """
    Split a single image into two statistically independent realizations by
    binomial thinning of each pixel's intensity (treated as photon/detector
    counts): each count is randomly assigned to image1 or image2 with 50/50
    probability.

    This is a convenience approximation for when you only have one image and
    no independent repeat acquisition or odd/even-frame split. It assumes
    pixel values are roughly proportional to photon counts (true for raw,
    reasonably low-noise-floor camera data; less accurate for images that
    have already been heavily processed/denoised/deconvolved). For a
    rigorous FRC in SMLM, prefer splitting the raw localization list (or
    frames) into two halves and reconstructing each independently.
    """
    rng = np.random.default_rng(seed)
    img_nonneg = np.clip(img, 0, None)
    counts = np.round(img_nonneg).astype(np.int64)
    img1 = rng.binomial(counts, 0.5).astype(np.float32)
    img2 = (counts - img1).astype(np.float32)
    return img1, img2


def measure_frc(img1, img2=None, pixel_size=None, unit="nm",
                 threshold=1.0 / 7.0, min_consistent_bins=8, out=None, seed=None):
    """
    Compute, plot, and report FRC resolution.

    If img2 is None, img1 is split via random_split_image (see its docstring
    for caveats). Otherwise img1 and img2 should be two independent
    reconstructions of the same field of view.

    min_consistent_bins: see frc_resolution() - how many consecutive bins the
    curve must stay below `threshold` for before that's accepted as the
    resolution crossing (guards against a single noisy high-frequency bin
    triggering, or invalidating, a crossing).

    Returns:
        spatial_freq, frc, resolution (resolution is None if the curve never
        reliably crosses the threshold, e.g. because the images are too
        noisy or too small)
    """
    if img2 is None:
        img1_split, img2_split = random_split_image(img1, seed=seed)
    else:
        img1_split, img2_split = img1, img2

    px = pixel_size if pixel_size is not None else 1.0
    unit_label = unit if pixel_size is not None else "px"

    spatial_freq, frc = compute_frc(img1_split, img2_split, pixel_size=px)
    resolution = frc_resolution(spatial_freq, frc, threshold=threshold,
                                 min_consistent_bins=min_consistent_bins)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(spatial_freq, frc, "k-", label="FRC")
    diffraction_limit_nm = 220
    diffraction_freq = 1 / diffraction_limit_nm
    plt.axvline(diffraction_freq, color='gray', linestyle=':', label=f'Abbe limit ({diffraction_limit_nm} nm)')
    ax.axhline(threshold, color="r", linestyle="--", label=f"threshold ({threshold:.3f})")
    if resolution is not None:
        freq_at_res = 1.0 / resolution
        ax.axvline(freq_at_res, color="b", linestyle="--",
                   label=f"resolution = {resolution:.1f} {unit_label}")
    ax.set_xlabel(f"Spatial frequency (cycles/{unit_label})")
    ax.set_ylabel("FRC")
    ax.set_ylim(-0.1, 1.05)
    ax.legend()
    ax.set_title("Fourier Ring Correlation")

    plt.tight_layout()
    if out:
        plt.savefig(out, dpi=200)
        print(f"Saved FRC plot to {out}")
    else:
        plt.show()

    if resolution is not None:
        print(f"FRC resolution estimate: {resolution:.2f} {unit_label}")
    else:
        print("Could not determine a reliable FRC resolution crossing "
              "(curve never drops below threshold and stays there).")

    return spatial_freq, frc, resolution


# ---------------------------------------------------------------------------
# Command-line interface (non-interactive; for scripting/batch use)
# ---------------------------------------------------------------------------

def parse_point(s):
    x, y = s.split(",")
    return float(x), float(y)


def _draw_line_location(ax, img, p1, p2, title=None):
    """Draw the full image with the sampled line overlaid in red."""
    ax.imshow(img, cmap="magma")
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], "r-", linewidth=1.5)
    if title:
        ax.set_title(title)
    ax.axis("off")
 
 
def _make_aspect_matched_figure(img, title, base_size=6.0, title_margin_in=0.45):
    """
    Build a (fig, ax) pair whose Axes bounding box has EXACTLY the same
    aspect ratio as `img` - i.e. no letterboxing/blank margin inside the
    axes - with a small fixed strip reserved above for a title.
 
    This matters for the corner zoom inset: inset_axes(loc="lower right")
    positions itself relative to the *axes* bounding box, not the image.
    If the axes were a plain square (6, 6) figure but the image is, say,
    2:1, imshow's default equal-aspect behavior pads the axes with blank
    space and "lower right of the axes" ends up sitting in that blank
    margin instead of the image's actual corner. Matching the axes aspect
    to the image aspect keeps the inset anchored to the real image corner
    for every image, regardless of its shape.
    """
    H, W = img.shape
    if W >= H:
        fig_w, img_h_in = base_size, base_size * H / W
    else:
        fig_w, img_h_in = base_size * W / H, base_size
 
    fig_h = img_h_in # + title_margin_in
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_axes([0, 0, 1, img_h_in / fig_h])
    # fig.suptitle(title, y=1 - 0.4 * (title_margin_in / fig_h), fontsize=11)
    return fig, ax
 
 
def _square_crop_around_line(p1, p2, W, H, pad_frac=0.15, min_pad=5.0):
    """
    Compute a SQUARE (xmin, xmax, ymin, ymax) crop centered on the line
    p1-p2, sized to comfortably contain the line plus padding on all
    sides. Using a square crop (rather than a padded bounding box, which
    is wide-and-short for a near-horizontal line) matters because imshow
    keeps pixel aspect ratio 1:1 - a non-square data range shown in a
    square axes gets letterboxed, making the zoomed content look a
    different, inconsistent size from image to image even though the
    inset box itself is a fixed square.
    """
    x1, y1 = p1
    x2, y2 = p2
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    line_len = float(np.hypot(x2 - x1, y2 - y1))
    pad = max(line_len * pad_frac, min_pad)
    half = max(abs(x2 - x1), abs(y2 - y1)) / 2.0 + pad
 
    # Cannot exceed what the image actually has in either dimension.
    half = min(half, (W - 1) / 2.0, (H - 1) / 2.0)
 
    xmin, xmax = cx - half, cx + half
    if xmin < 0:
        xmax -= xmin
        xmin = 0.0
    if xmax > W - 1:
        xmin -= (xmax - (W - 1))
        xmax = W - 1
    xmin = max(xmin, 0.0)
 
    ymin, ymax = cy - half, cy + half
    if ymin < 0:
        ymax -= ymin
        ymin = 0.0
    if ymax > H - 1:
        ymin -= (ymax - (H - 1))
        ymax = H - 1
    ymin = max(ymin, 0.0)
 
    return xmin, xmax, ymin, ymax
 
 
def _add_corner_zoom_inset(ax, img, p1, p2, pad_frac=0.15, min_pad=5.0, side_frac=0.5):
    """
    Add an inset axes in the bottom-right corner of `ax`. The inset is
    always a fixed physical square - side_frac (default 0.5) of the
    shorter side of the *image's own axes box* in inches, so with a square
    image that's half its width/height (~1/4 of its area) - kept identical
    in proportion across every exported figure for visual cohesion,
    regardless of the source image's own aspect ratio. The cropped region
    it zooms into is likewise forced to be a square (see
    _square_crop_around_line) so the zoomed content fully fills that
    square with no letterboxing, keeping the on-screen zoom size and scale
    visually consistent across images.
 
    The connecting lines run bottom-left-to-bottom-left and
    top-right-to-top-right between the zoomed region on the main image and
    the inset box.
    """
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
 
    H, W = img.shape
    x1, y1 = p1
    x2, y2 = p2
    xmin, xmax, ymin, ymax = _square_crop_around_line(
        p1, p2, W, H, pad_frac=pad_frac, min_pad=min_pad
    )
 
    # Fixed physical (inch) square size, computed from this Axes' own box
    # (not the whole figure, which may include a title strip), so the
    # inset is always square and proportioned the same way relative to the
    # actual displayed image, however the image's own aspect ratio.
    bbox = ax.get_position()
    fig_w_in, fig_h_in = ax.figure.get_size_inches()
    ax_w_in = bbox.width * fig_w_in
    ax_h_in = bbox.height * fig_h_in
    side_in = side_frac * min(ax_w_in, ax_h_in)
 
    axins = inset_axes(ax, width=side_in, height=side_in, loc="lower right", borderpad=0)
    axins.imshow(img, cmap="magma")
    axins.plot([x1, x2], [y1, y2], "r-", linewidth=2)
    axins.set_xlim(xmin, xmax)
    axins.set_ylim(ymax, ymin)  # inverted to match image (row) orientation
    axins.set_xticks([])
    axins.set_yticks([])
    for spine in axins.spines.values():
        spine.set_edgecolor("yellow")
        spine.set_linewidth(1.5)
 
    # loc1=3 (lower-left) / loc2=1 (upper-right): connect the zoomed
    # region's bottom-left corner to the inset's bottom-left corner, and
    # its top-right corner to the inset's top-right corner.
    # mark_inset(ax, axins, loc1=3, loc2=1, fc="none", ec="yellow", linewidth=1)


def _run_line_cli(args):
    img = load_image(args.image)
    distances, values = extract_line_profile(
        img, args.p1, args.p2, n_samples=args.n_samples, device=args.device
    )
    scale = args.pixel_size if args.pixel_size is not None else 1.0
    unit = args.unit if args.pixel_size is not None else "px"
    d = distances * scale

    if args.mode == "single-peak":
        popt, perr = fit_single_peak(d, values)
        amp, mu, sigma, offset = popt
        fwhm = fwhm_from_sigma(sigma)
        fwhm_err = fwhm_from_sigma(perr[2])
        print(f"Peak center: {mu:.2f} {unit}")
        print(f"FWHM (resolution estimate): {fwhm:.2f} +/- {fwhm_err:.2f} {unit}")
    else:
        popt, perr = fit_two_peaks(d, values)
        amp1, mu1, sigma1, amp2, mu2, sigma2, offset = popt
        separation = abs(mu2 - mu1)
        sep_err = float(np.sqrt(perr[1] ** 2 + perr[4] ** 2))
        print(f"Peak 1 center: {mu1:.2f} {unit}")
        print(f"Peak 2 center: {mu2:.2f} {unit}")
        print(f"Peak separation (resolution estimate): {separation:.2f} +/- {sep_err:.2f} {unit}")

    def _build_profile_axis(ax):
        ax.plot(d, values, "k.", markersize=4, label="data")
        if args.mode == "single-peak":
            xfit = np.linspace(d.min(), d.max(), 500)
            ax.plot(xfit, gaussian(xfit, *popt), "r-", label="Gaussian fit")
            ax.axvline(mu - fwhm / 2, color="b", linestyle="--")
            ax.axvline(mu + fwhm / 2, color="b", linestyle="--")
        else:
            xfit = np.linspace(d.min(), d.max(), 500)
            ax.plot(xfit, two_gaussians(xfit, *popt), "r-", label="Double Gaussian fit")
            ax.axvline(mu1, color="b", linestyle="--")
            ax.axvline(mu2, color="g", linestyle="--")
        ax.set_xlabel(f"Distance along line ({unit})")
        ax.set_ylabel("Intensity")
        ax.legend()
        ax.set_title("Intensity profile & fit")

    if not args.out:
        # No output path given: fall back to a single interactive combined view.
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        _draw_line_location(axes[0], img, args.p1, args.p2)
        _build_profile_axis(axes[1])
        plt.tight_layout()
        plt.show()
        return

    out = Path(args.out)
    folder = out.parent if str(out.parent) else Path(".")
    folder.mkdir(parents=True, exist_ok=True)
    stem = out.stem
    ext = out.suffix if out.suffix else ".png"

    image_path = folder / f"{stem}_image{ext}"
    zoom_path = folder / f"{stem}_image_zoom{ext}"
    plot_path = folder / f"{stem}_profile{ext}"

    # 1) Plain image with the line location.
    fig_img, ax_img = _make_aspect_matched_figure(img, "Line location")
    _draw_line_location(ax_img, img, args.p1, args.p2)
    plt.savefig(image_path, dpi=200)
    plt.close(fig_img)
    print(f"Saved image to {image_path}")

    # 2) Same image, with a zoomed-in view of the line region inset in the
    #    bottom-right corner (~1/4 of the image area).
    fig_zoom, ax_zoom = _make_aspect_matched_figure(img, "Line location (zoomed)")
    _draw_line_location(ax_zoom, img, args.p1, args.p2)
    _add_corner_zoom_inset(ax_zoom, img, args.p1, args.p2)
    plt.savefig(zoom_path, dpi=200)
    plt.close(fig_zoom)
    print(f"Saved zoomed image to {zoom_path}")

    # 3) Intensity profile plot.
    fig_plot, ax_plot = plt.subplots(figsize=(6, 5))
    _build_profile_axis(ax_plot)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.close(fig_plot)
    print(f"Saved plot to {plot_path}")


def _run_frc_cli(args):
    img1 = load_image(args.image)
    img2 = load_image(args.image2) if args.image2 else None
    measure_frc(img1, img2, pixel_size=args.pixel_size, unit=args.unit,
                threshold=args.threshold, out=args.out, seed=args.seed)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_line = sub.add_parser("line", help="Line-profile FWHM / peak-separation measurement")
    p_line.add_argument("--image", required=True, type=Path)
    p_line.add_argument("--p1", required=True, type=parse_point, help="'x,y' in pixels")
    p_line.add_argument("--p2", required=True, type=parse_point, help="'x,y' in pixels")
    p_line.add_argument("--mode", choices=["single-peak", "two-peak"], default="single-peak")
    p_line.add_argument("--pixel-size", type=float, default=None)
    p_line.add_argument("--unit", type=str, default="nm")
    p_line.add_argument("--n-samples", type=int, default=None)
    p_line.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p_line.add_argument("--out", type=Path, default=None)
    p_line.set_defaults(func=_run_line_cli)

    p_frc = sub.add_parser("frc", help="Fourier Ring Correlation whole-image resolution")
    p_frc.add_argument("--image", required=True, type=Path)
    p_frc.add_argument("--image2", type=Path, default=None,
                        help="Optional second independent image. If omitted, "
                             "--image is randomly split (see random_split_image).")
    p_frc.add_argument("--pixel-size", type=float, default=None)
    p_frc.add_argument("--unit", type=str, default="nm")
    p_frc.add_argument("--threshold", type=float, default=1.0 / 7.0)
    p_frc.add_argument("--seed", type=int, default=None)
    p_frc.add_argument("--out", type=Path, default=None)
    p_frc.set_defaults(func=_run_frc_cli)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()