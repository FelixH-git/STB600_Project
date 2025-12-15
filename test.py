import cv2
import numpy as np

def resize(img, scale):
    return cv2.resize(img, (int(img.shape[1]/scale),int(img.shape[0]/scale)))

# Load image
current_color = cv2.imread("Images/Yellow.jpg")

# Convert to HSV (best for color masking)
hsv = cv2.cvtColor(current_color, cv2.COLOR_BGR2HSV)

# Green color range
lower_green = np.array([35, 80, 40])   # adjust if needed
upper_green = np.array([85, 255, 255])

# Create green mask
mask = cv2.inRange(hsv, lower_green, upper_green)

# Invert mask (keep non-green areas)
inv_mask = cv2.bitwise_not(mask)

# Remove green by applying the inverted mask
no_green = cv2.bitwise_and(current_color, current_color, mask=inv_mask)

cv2.imshow("no green", resize(no_green, 4))
cv2.waitKey(0)
cv2.imwrite("Coins/no_green_test_coin.jpg", no_green)