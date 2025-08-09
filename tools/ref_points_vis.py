#!/usr/bin/env python3
"""
plot_vp_points.py

Loads an image and a .npy file of normalized 2D points,
plots each point onto the image with a contrasting outline
plus an additional absolute coordinate marker at (610, 188),
and saves the result.
"""

import os
import cv2
import numpy as np

# ─── Configuration ─────────────────────────────────────────────────────────────
INPUT_IMAGE_FILE   = '/project/aimm/SSC/SemanticKITTI/dataset/sequences/08/image_2/002022.png'
INPUT_NPY_FILE     = '/project/aimm/SSC/Symphonies_VP_1526/vp_sample.npy'
OUTPUT_IMAGE_FILE  = './output.png'

# Plot parameters for normalized points
POINT_RADIUS       = 4               # radius of the inner (colored) circle
POINT_COLOR        = (0, 0, 255)     # BGR for red
OUTER_COLOR        = (0, 0, 0)       # BGR for black outline
OUTER_MARGIN       = 3               # outline thickness = margin added to radius

# Parameters for the extra absolute point
ABS_X, ABS_Y       = 590, 162        # absolute pixel coordinates
ABS_RADIUS         = 5              # make this point a bit larger
ABS_COLOR          = (0, 255, 255)   # BGR for yellow
ABS_OUTER_COLOR    = (0, 0, 0)       # black outline
ABS_OUTER_MARGIN   = 4               # margin for outline
# ────────────────────────────────────────────────────────────────────────────────

def main():
    # Load image
    img = cv2.imread(INPUT_IMAGE_FILE, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {INPUT_IMAGE_FILE!r}")
    h, w = img.shape[:2]

    # Load normalized points
    pts = np.load(INPUT_NPY_FILE)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"Expected pts.shape == (n,2), got {pts.shape}")

    # Denormalize coordinates (assume values in [0,1])
    xs = (pts[:, 0] * w).astype(int)
    ys = (pts[:, 1] * h).astype(int)

    # Plot each normalized point with black outline + red core
    for x, y in zip(xs, ys):
        x = max(0, min(w - 1, x))
        y = max(0, min(h - 1, y))
        # Outline
        cv2.circle(img, (x, y), POINT_RADIUS + OUTER_MARGIN, OUTER_COLOR, thickness=-1)
        # Core
        cv2.circle(img, (x, y), POINT_RADIUS, POINT_COLOR, thickness=-1)

    # Plot the extra absolute point at (610, 188)
    ax = max(0, min(w - 1, ABS_X))
    ay = max(0, min(h - 1, ABS_Y))
    # Outline
    cv2.circle(img, (ax, ay), ABS_RADIUS + ABS_OUTER_MARGIN, ABS_OUTER_COLOR, thickness=-1)
    # Core
    cv2.circle(img, (ax, ay), ABS_RADIUS, ABS_COLOR, thickness=-1)

    # Ensure output directory exists
    out_dir = os.path.dirname(OUTPUT_IMAGE_FILE) or '.'
    os.makedirs(out_dir, exist_ok=True)

    # Save result
    if not cv2.imwrite(OUTPUT_IMAGE_FILE, img):
        raise IOError(f"Failed to write output image to {OUTPUT_IMAGE_FILE!r}")
    print(f"Saved plotted image to {OUTPUT_IMAGE_FILE}")

if __name__ == "__main__":
    main()
