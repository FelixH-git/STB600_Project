import cv2
import numpy as np
import time

orgimg = cv2.imread("Generated_Image.png")

def resize(img, scale):
    return cv2.resize(
        img,
        (int(img.shape[1] / scale), int(img.shape[0] / scale))
    )



img = cv2.cvtColor(orgimg, cv2.COLOR_BGR2GRAY)

_, thresh = cv2.threshold(img, 40, 255, cv2.THRESH_BINARY)




contours, hierarchy = cv2.findContours(
    thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
)

# FPS setup
prev_time = time.time()

for i, contour in enumerate(contours):
    area = cv2.contourArea(contour)
    if 500 < area < 1_000_000:

        # Draw contour
        cv2.drawContours(orgimg, [contour], -1, (0, 255, 0), 3)

        # Get parent index
        parent = hierarchy[0][i][3]

        # Compute centroid
        M = cv2.moments(contour)
        if M["m00"] == 0:
            continue

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        # Parent-child text
        if parent == -1:
            text = f"ID:{i} Parent:None"
        else:
            text = f"ID:{i} Parent:{parent}"

        cv2.putText(
            orgimg,
            text,
            (cx, cy),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,              # text size
            (0, 0, 255),
            2,              # thickness
            cv2.LINE_AA
        )

# FPS calculation
curr_time = time.time()
fps = 1 / (curr_time - prev_time)

cv2.putText(
    orgimg,
    f"FPS: {fps:.2f}",
    (20, 40),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,                  # text size
    (255, 0, 0),
    2,                  # thickness
    cv2.LINE_AA
)

cv2.imshow("orgimg", resize(orgimg, 3))
cv2.waitKey(0)
cv2.destroyAllWindows()
