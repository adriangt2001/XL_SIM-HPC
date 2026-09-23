import argparse

import cv2
import imageio
import numpy as np
import torch

from src.microscope.microscope import Microscope


def generate_rolling_calibration(
    filename: str,
    microscope_config: str,
    output: str,
    fps: int = 5,
) -> None:
    # Load grayscale sample image
    sample_gray = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    if sample_gray is None:
        raise FileNotFoundError(f"{filename} not found.")

    # Load microscope
    microscope = Microscope.from_file(microscope_config)

    # Generate calibration pattern stack
    _, pattern_stack = microscope(torch.from_numpy(sample_gray))
    pattern_stack = pattern_stack.cpu().numpy()

    # Apply static colormap to the target sample
    sample_colored = cv2.applyColorMap(
        sample_gray,
        cv2.COLORMAP_VIRIDIS,
    )

    print(f"{pattern_stack.shape=}")
    print(f"{sample_colored.shape=}")

    frames = []

    # Generate rolling pattern animation
    for frame_idx in range(pattern_stack.shape[0]):
        pattern: np.ndarray = pattern_stack[frame_idx][
            : sample_colored.shape[0],
            : sample_colored.shape[1],
        ]

        pattern = (pattern * 255).astype(np.uint8)

        # Apply colormap to dynamic calibration pattern
        pattern_colored = cv2.applyColorMap(
            pattern,
            cv2.COLORMAP_HOT,
        )

        print(f"Frame {frame_idx}: {pattern_colored.shape=}")

        # Blend static sample with dynamic pattern
        blended = cv2.addWeighted(
            sample_colored,
            0.6,
            pattern_colored,
            0.4,
            0,
        )

        # Convert BGR -> RGB for GIF generation
        frames.append(
            cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)
        )

    # Export GIF
    imageio.mimsave(
        output,
        frames,
        fps=fps,
        loop=0,
    )

    print(f"Saved calibration GIF to: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a rolling microscope calibration GIF."
    )

    parser.add_argument(
        "--filename",
        required=True,
        help="Path to the input grayscale sample image.",
    )

    parser.add_argument(
        "--microscope-config",
        required=True,
        help="Path to the microscope YAML configuration.",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Path to the output GIF.",
    )

    parser.add_argument(
        "--fps",
        type=int,
        default=5,
        help="GIF frame rate (default: 5).",
    )

    args = parser.parse_args()

    generate_rolling_calibration(
        filename=args.filename,
        microscope_config=args.microscope_config,
        output=args.output,
        fps=args.fps,
    )


if __name__ == "__main__":
    main()
