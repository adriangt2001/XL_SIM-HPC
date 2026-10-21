from argparse import ArgumentParser, Namespace
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, gaussian_filter
from skimage.morphology import disk
from skimage.transform import resize


def generate_siemens_star(args: Namespace):
    name = f"siemens_{args.mode}_{args.num_spokes:02d}"

    match args.mode:
        case "binary":
            hr_size = args.size * args.oversample
            y, x = np.ogrid[-1 : 1 : complex(0, hr_size), -1 : 1 : complex(0, hr_size)]
            theta = np.arctan2(y, x)
            hr_star = (np.sin(args.num_spokes * theta) >= 0).astype(np.float64)
            pattern = resize(hr_star, (args.size, args.size), anti_aliasing=True, order=3)
        case "sinusoidal":
            y, x = np.ogrid[-1 : 1 : complex(0, args.size), -1 : 1 : complex(0, args.size)]
            theta = np.arctan2(y, x)
            pattern = 0.5 * (1.0 + np.sin(args.num_spokes * theta))
        case _:
            raise ValueError(f"Unknown mode: {args.mode}")

    pattern = (pattern * 255).astype(np.uint8)
    return pattern, name


def generate_chirp_grid(args: Namespace):
    name = f"chirp_grid_{args.orientation}_px{args.pixel_size:.02f}".replace(".", "-")

    height, width = args.size, args.size
    yy, xx = np.mgrid[0:height, 0:width]
 
    match args.orientation:
        case "horizontal":
            pos = xx * args.pixel_size
            span = width * args.pixel_size
        case "vertical":
            pos = yy * args.pixel_size
            span = height * args.pixel_size
        case "radial":
            cy, cx = height / 2, width / 2
            pos = np.hypot(xx - cx, yy - cy) * args.pixel_size
            span = pos.max()
        case _:
            raise ValueError(f"Unknown orientation: {args.orientation}")
 
    f_min, f_max = 1 / args.period_max, 1 / args.period_min
    phase = 2 * np.pi * (f_min * pos + (f_max - f_min) / (2 * span) * pos**2)
    pattern: np.ndarray = 0.5 + 0.5 * np.cos(phase)
 
    if args.binary:
        name = name + "_binary"
        pattern = (pattern >= 0.5).astype(np.float32)

    pattern = (pattern * 255).astype(np.uint8)
    return pattern, name


def generate_filament_network(args: Namespace):
    name = f"filament_px{args.pixel_size:.02f}".replace(".", "-")

    rng = np.random.default_rng(args.seed)
    height, width = args.size, args.size
    step_px = args.step_length / args.pixel_size
    width_px = args.filament_width / args.pixel_size
    n_steps_min = max(1, round(args.min_length / args.step_length))
    n_steps_max = max(n_steps_min + 1, round(args.max_length / args.step_length))

    segments = []
 
    def grow(x, y, angle, depth=0):
        n_steps = rng.integers(n_steps_min, n_steps_max)

        dtheta = rng.uniform(
            -args.max_turn,
            args.max_turn,
            size=n_steps,
        )

        angles = angle + np.cumsum(dtheta)

        dx = step_px * np.cos(angles)
        dy = step_px * np.sin(angles)

        xs = x + np.cumsum(dx)
        ys = y + np.cumsum(dy)

        valid = (
            (xs >= 0)
            & (xs < width)
            & (ys >= 0)
            & (ys < height)
        )

        if not np.all(valid):
            first_invalid = np.flatnonzero(~valid)[0]

            xs = xs[:first_invalid]
            ys = ys[:first_invalid]
            angles = angles[:first_invalid]

        if len(xs) == 0:
            return

        segments.append(
            np.column_stack((xs, ys))
        )

        if depth < args.max_branch_depth:
            branch_events = (
                rng.random(len(xs))
                < args.branch_probability
            )

            branch_indices = np.flatnonzero(branch_events)

            for i in branch_indices:
                branch_angle = (
                    angles[i]
                    + rng.choice([-1.0, 1.0])
                    * rng.uniform(0.5, 1.2)
                )

                grow(
                    xs[i],
                    ys[i],
                    branch_angle,
                    depth=depth + 1,
                )
 
    for _ in range(args.num_filaments):
        start_x = rng.uniform(0, width)
        start_y = rng.uniform(0, height)
        start_angle = rng.uniform(0, 2 * np.pi)
        grow(start_x, start_y, start_angle)

    mask = np.zeros((height, width), dtype=bool)
    sample_spacing = 0.5

    for points in segments:
        if len(points) == 0:
            continue

        if len(points) == 1:
            px = np.rint(points[:, 0]).astype(np.intp)
            py = np.rint(points[:, 1]).astype(np.intp)

            valid = (
                (px >= 0)
                & (px < width)
                & (py >= 0)
                & (py < height)
            )

            mask[py[valid], px[valid]] = True
            continue

        # Differences between consecutive polyline vertices.
        dx = np.diff(points[:, 0])
        dy = np.diff(points[:, 1])

        lengths = np.sqrt(dx * dx + dy * dy)

        # Number of interpolation intervals for each edge.
        n = np.maximum(
            1,
            np.ceil(lengths / sample_spacing).astype(np.intp),
        )

        # Total number of interpolated samples.
        total_samples = int(n.sum())

        # Allocate all samples for this polyline at once.
        t = np.empty(total_samples, dtype=np.float32)

        # Starting index of each edge in the flattened array.
        offsets = np.empty(len(n) + 1, dtype=np.intp)
        offsets[0] = 0
        np.cumsum(n, out=offsets[1:])

        # Generate interpolation parameters.
        for i, ni in enumerate(n):
            start = offsets[i]
            stop = offsets[i + 1]

            # Exclude endpoint so neighboring edges don't duplicate it.
            t[start:stop] = np.arange(ni, dtype=np.float32) / ni

        # Edge index for each sample.
        edge_idx = np.repeat(
            np.arange(len(n)),
            n,
        )

        x0 = points[edge_idx, 0]
        y0 = points[edge_idx, 1]

        x1 = points[edge_idx + 1, 0]
        y1 = points[edge_idx + 1, 1]

        x = x0 + t * (x1 - x0)
        y = y0 + t * (y1 - y0)

        # Include the final vertex.
        x = np.concatenate((x, points[-1:, 0]))
        y = np.concatenate((y, points[-1:, 1]))

        # Convert to pixel coordinates.
        px = np.rint(x).astype(np.intp)
        py = np.rint(y).astype(np.intp)

        valid = (
            (px >= 0)
            & (px < width)
            & (py >= 0)
            & (py < height)
        )

        mask[py[valid], px[valid]] = True

    radius = max(1, round(width_px / 2))
    thick = binary_dilation(mask, disk(radius))
    pattern = gaussian_filter(thick.astype(np.float32), sigma=args.edge_sigma)
    peak = pattern.max()
    pattern = pattern / peak if peak > 0 else pattern

    pattern = (pattern * 255).astype(np.uint8)
    return pattern, name


def generate_point_grid(args: Namespace):
    name = f"point_grid_px{args.pixel_size:.02f}".replace(".", "-")

    rng = np.random.default_rng(args.seed)
    height, width = args.size, args.size

    spacing_px = args.spacing / args.pixel_size
    sigma_px = args.point_sigma / args.pixel_size
    jitter_px = args.jitter / args.pixel_size
    margin_px = args.margin / args.pixel_size

    xs = np.arange(margin_px, width - margin_px, spacing_px)
    ys = np.arange(margin_px, height - margin_px, spacing_px)

    x0, y0 = np.meshgrid(xs, ys)
    x0 = x0.ravel()
    y0 = y0.ravel()

    n = x0.size

    if jitter_px:
        x0 = x0 + rng.uniform(-jitter_px, jitter_px, n)
        y0 = y0 + rng.uniform(-jitter_px, jitter_px, n)

    if isinstance(args.intensity, tuple):
        amp = rng.uniform(*args.intensity, n)
    else:
        amp = np.full(n, args.intensity)

    # Bilinearly splat each point's amplitude onto the pixel grid, then blur
    # once with the target sigma. This is mathematically equivalent to
    # summing n Gaussians, but O(n + H*W) instead of O(n*H*W).
    impulses = np.zeros((height, width), dtype=np.float64)
    x0c = np.clip(x0, 0, width - 1.001)
    y0c = np.clip(y0, 0, height - 1.001)
    x_lo = np.floor(x0c).astype(int)
    y_lo = np.floor(y0c).astype(int)
    fx = x0c - x_lo
    fy = y0c - y_lo

    for dx, dy, w in (
        (0, 0, (1 - fx) * (1 - fy)),
        (1, 0, fx * (1 - fy)),
        (0, 1, (1 - fx) * fy),
        (1, 1, fx * fy),
    ):
        np.add.at(impulses, (y_lo + dy, x_lo + dx), amp * w)

    # gaussian_filter normalizes its kernel to unit integral, so a point of
    # mass `amp` ends up with peak amp/(2*pi*sigma**2) after blurring;
    # rescale so an isolated point still peaks at its own amplitude, same
    # convention as the direct-sum version.
    pattern = gaussian_filter(impulses, sigma=sigma_px, mode="constant")
    pattern *= 2 * np.pi * sigma_px**2

    pattern = np.clip(pattern, 0.0, 1.0).astype(np.float32)
    pattern = (pattern * 255).astype(np.uint8)
    return pattern, name


def generate_pattern(args: Namespace):
    pattern, name = args.func(args)

    args.out.mkdir(exist_ok=True)
    output_file: Path = (args.out / name).with_suffix(".png")
    Image.fromarray(pattern).save(output_file)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--out", type=Path, required=True, help="Output folder")

    subparser = parser.add_subparsers(dest="command", required=True)

    p_siemens = subparser.add_parser("siemens", help="Siemens pattern")
    p_siemens.add_argument("--mode", type=str, choices=["binary", "sinusoidal"], help="Type of siemens star to generate")
    p_siemens.add_argument("--num_spokes", type=int, required=True, help="Number of spokes")
    p_siemens.add_argument("--size", type=int, required=True, help="Image size")
    p_siemens.add_argument("--oversample", type=int, default=8, help="Scaling factor for degradation to ensure sharpness")
    p_siemens.set_defaults(func=generate_siemens_star)

    p_chirp = subparser.add_parser("chirp", help="Chirp grid pattern")
    p_chirp.add_argument("--size", type=int, required=True, help="Image size")
    p_chirp.add_argument("--pixel_size", type=float, required=True, help="Pixel size (nm)")
    p_chirp.add_argument("--period_min", type=float, required=True, help="Minimum period")
    p_chirp.add_argument("--period_max", type=float, required=True, help="Maximum period")
    p_chirp.add_argument("--orientation", type=str, choices=["horizontal", "vertical", "radial"], required=True, help="Orientation of the chirps") 
    p_chirp.add_argument("--binary", action="store_true", help="Whether to binarize the image or not")
    p_chirp.set_defaults(func=generate_chirp_grid)

    p_filament = subparser.add_parser("filament", help="Filament network pattern")
    p_filament.add_argument("--size", type=int, required=True, help="Image size")
    p_filament.add_argument("--pixel_size", type=float, required=True, help="Pixel size (nm)")
    p_filament.add_argument("--num_filaments", type=int, required=True, help="Number of filaments")
    p_filament.add_argument("--filament_width", type=float, required=True, help="Filament width (nm)")
    p_filament.add_argument("--min_length", type=float, default=1000.0, help="")
    p_filament.add_argument("--max_length", type=float, default=4000.0, help="")
    p_filament.add_argument("--step_length", type=float, default=50.0, help="")
    p_filament.add_argument("--max_turn", type=float, default=0.3, help="")
    p_filament.add_argument("--branch_probability", type=float, default=0.03, help="")
    p_filament.add_argument("--max_branch_depth", type=int, default=2, help="")
    p_filament.add_argument("--edge_sigma", type=float, default=0.5, help="")
    p_filament.add_argument("--seed", type=int, default=None, help="Seed for RNG")
    p_filament.set_defaults(func=generate_filament_network)

    p_pgrid = subparser.add_parser("point_grid", help="Point grid pattern")
    p_pgrid.add_argument("--size", type=int, required=True, help="")
    p_pgrid.add_argument("--pixel_size", type=float, required=True, help="")
    p_pgrid.add_argument("--spacing", type=float, required=True, help="")
    p_pgrid.add_argument("--point_sigma", type=float, default=20.0, help="")
    p_pgrid.add_argument("--jitter", type=float, default=0.0, help="")
    p_pgrid.add_argument("--intensity", type=float, default=1.0, help="")
    p_pgrid.add_argument("--margin", type=float, default=0.0, help="")
    p_pgrid.add_argument("--seed", type=int, default=None, help="")
    p_pgrid.set_defaults(func=generate_point_grid)

    args = parser.parse_args()

    generate_pattern(args)
