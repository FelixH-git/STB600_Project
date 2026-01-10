import cv2
import numpy as np




# Read offline image
frame = cv2.imread("Images/Yellow.jpg")
if frame is None:
    raise IOError("Image not loaded")

# --- Brick detection (HSV) ---
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

mask = cv2.inRange(hsv, (0, 0, 0), (180, 255, 80))
mask = cv2.morphologyEx(
    mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)
)

contours, _ = cv2.findContours(
    mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)


bf = cv2.BFMatcher(cv2.NORM_HAMMING)

coin = None
max_area = 0
for c in contours:
    x, y, w, h = cv2.boundingRect(c)
    if h > w * 2:
        area = w * h
        if area > max_area:
            max_area = area
            coin = (x, y, w, h)

if coin is None:
    print("Brick not found")
    exit()

# --- Crop brick ---
x, y, w, h = coin
brick_img = frame[y:y+h, x:x+w]

# --- Split into 4 vertical parts ---
h_part = h // 4
top = brick_img[0:h_part, :]

# --- Yellow symbol masking ---
top_gray = cv2.cvtColor(top, cv2.COLOR_BGR2GRAY)
top_hsv = cv2.cvtColor(top, cv2.COLOR_BGR2HSV)

symbol_mask = cv2.inRange(top_hsv, (15, 80, 80), (40, 255, 255))



cnt_ref = np.load("symbol_contour.npy", allow_pickle=True)

h, w, _ = brick_img.shape
h_part = h // 4

best_score = float("inf")
best_index = -1

for i in range(4):
    region = brick_img[i*h_part:(i+1)*h_part, :]

    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (15, 80, 80), (40, 255, 255))

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        continue

    cnt = max(cnts, key=cv2.contourArea)

    score = cv2.matchShapes(cnt_ref, cnt, cv2.CONTOURS_MATCH_I1, 0)
    print(f"Region {i} score:", score)

    if score < best_score:
        best_score = score
        best_index = i

