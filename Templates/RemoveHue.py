import cv2
import numpy as np

# Load image (BGR)s
img = cv2.imread(r"C:\Users\biggi\Desktop\STB600_Project\Templates\0.png", cv2.IMREAD_COLOR)

# ---------------------------
# 1. Invert the image
# ---------------------------
inverted = cv2.bitwise_not(img)

# ---------------------------
# 2. Convert to HSL
# OpenCV uses HLS (H, L, S)
# ---------------------------
hls = cv2.cvtColor(inverted, cv2.COLOR_BGR2HLS)

h, l, s = cv2.split(hls)

# ---------------------------
# 3. Define hue range to remove
# OpenCV Hue range: 0–179
# Given H = 38 (out of 360), scale it:
# 38 * 179 / 360 ≈ 19
# ---------------------------
target_hue = int(38 * 179 / 360)
tolerance = 5  # adjust if needed

lower = target_hue - tolerance
upper = target_hue + tolerance

# Create mask for the hue
hue_mask = cv2.inRange(h, lower, upper)

# ---------------------------
# 4. Remove hue (make transparent)
# ---------------------------
# Convert to BGRA so we can use alpha channel
bgra = cv2.cvtColor(inverted, cv2.COLOR_BGR2BGRA)

# Set alpha to 0 where hue matches
bgra[:, :, 3][hue_mask > 0] = 0

# ---------------------------
# 5. Save result
# ---------------------------
cv2.imwrite(r"C:\Users\biggi\Desktop\STB600_Project\Templates\0_MASK.png", bgra)
