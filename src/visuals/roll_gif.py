import cv2
import imageio
import numpy as np
import torch

from src.microscope.microscope import Microscope

filename = 'gifs/biosr_microtubules_hr_2.png'
microscope_config = 'configs/simulator/lessnoise_microscope.yaml'
output = 'gifs/rolling_callibration.gif'

# 1. Load or define grayscale sample image
sample_gray = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
assert sample_gray is not None, FileNotFoundError(f"{filename} not found.")

h, w = sample_gray.shape

microscope = Microscope.from_file(microscope_config)
_, pattern_stack = microscope(torch.from_numpy(sample_gray))
pattern_stack = pattern_stack.cpu().numpy()

# 3. Apply static colormap to the target sample
sample_colored = cv2.applyColorMap(sample_gray, cv2.COLORMAP_VIRIDIS)

frames = []
print(f"{pattern_stack.shape=}")
print(f"{sample_colored.shape=}")

# 4. Generate rolling pattern animation loop
for frame_idx in range(pattern_stack.shape[0]):
    pattern: np.ndarray = pattern_stack[frame_idx][:sample_colored.shape[0], :sample_colored.shape[1]]
    pattern = (pattern * 255).astype(np.uint8)

    # Apply secondary colormap to dynamic calibration pattern
    pattern_colored = cv2.applyColorMap(pattern, cv2.COLORMAP_HOT)

    print(f"{pattern_colored.shape=}")

    # Blend static sample with dynamic pattern overlay
    blended = cv2.addWeighted(sample_colored, 0.6, pattern_colored, 0.4, 0)

    # Store frame in RGB format for GIF generation
    frames.append(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))

# 5. Export result to GIF
imageio.mimsave("gifs/rolling_calibration.gif", frames, fps=5, loop=0)