import numpy as np
import cv2

# Output path
#HEATMAP_PATH = "heatmap.png"
HEATMAP_PATH_TEMPLATE = "heatmap_%d.png"

# Image size
H, W = 512, 512

# Number of blobs
NUM_BLOBS = 30
NUM_CAMERAS = 10

# Helper: draw a Gaussian blob
def add_blob(img, center, sigma, intensity):
    cx, cy = center
    x = np.arange(0, W, 1, float)
    y = np.arange(0, H, 1, float)[:, np.newaxis]
    gauss = np.exp(-((x - cx)**2 + (y - cy)**2) / (2.0 * sigma**2))
    img += intensity * gauss


for CAMERA_INDEX in range(NUM_CAMERAS):

 # Create empty heatmap
 heatmap = np.zeros((H, W), dtype=np.float32)

 # Add random blobs
 rng = np.random.default_rng()

 for _ in range(NUM_BLOBS):
    cx = rng.integers(0, W)
    cy = rng.integers(0, H)
    sigma = rng.uniform(10, 80)         # blob size
    intensity = rng.uniform(0.5, 1.5)   # blob strength
    add_blob(heatmap, (cx, cy), sigma, intensity)

 # Normalize to 0–255
 heatmap -= heatmap.min()
 if heatmap.max() > 0:
    heatmap /= heatmap.max()
 heatmap_img = (heatmap * 255).astype(np.uint8)

 # Option A: save as grayscale
 hm_path = HEATMAP_PATH_TEMPLATE % CAMERA_INDEX
 cv2.imwrite(hm_path, heatmap_img)

# (Optional) Option B: apply a colormap and save
# color_map = cv2.applyColorMap(heatmap_img, cv2.COLORMAP_JET)
# cv2.imwrite("heatmap_color.png", color_map)

print("Saved heatmap to:", hm_path)

