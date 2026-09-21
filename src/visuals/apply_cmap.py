from pathlib import Path

import cv2

filepath = 'gifs/biosr_microtubules_rl_2.png'
output = str(Path('gifs/') / (Path(filepath).stem + '_magma')) + '.png'

# Load the image in grayscale
img_gray = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

# Apply the Magma colormap
img_magma = cv2.applyColorMap(img_gray, cv2.COLORMAP_MAGMA)

# Save the colored result
cv2.imwrite(output, img_magma)