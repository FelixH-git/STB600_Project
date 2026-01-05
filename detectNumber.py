import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

MODEL_PATH = "digit_cnn.keras"
IMG_PATH = "Images/Red.jpg"
IMG_SIZE = 64

model = load_model(MODEL_PATH)

img = cv2.imread(IMG_PATH)
if img is None:
    raise ValueError("Image not found")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

binary = cv2.adaptiveThreshold(
    gray,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV,
    51,
    5
)

kernel = np.ones((3,3), np.uint8)
binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

num_labels, labels = cv2.connectedComponents(binary)

blob_boxes = []
for label in range(1, num_labels):
    ys, xs = np.where(labels == label)
    if len(xs) == 0:
        continue
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    if (x1 - x0) * (y1 - y0) < 50:
        continue
    blob_boxes.append((x0, y0, x1, y1))

def group_blobs(boxes, x_thresh=30, y_thresh=70):
    boxes = sorted(boxes, key=lambda b: b[1])
    groups = []
    for box in boxes:
        x0, y0, x1, y1 = box
        placed = False
        for i, (gx0, gy0, gx1, gy1) in enumerate(groups):
            x_dist = abs((x0 + x1)//2 - (gx0 + gx1)//2)
            y_gap = y0 - gy1
            if x_dist < x_thresh and y_gap < y_thresh:
                groups[i] = (
                    min(gx0, x0),
                    min(gy0, y0),
                    max(gx1, x1),
                    max(gy1, y1),
                )
                placed = True
                break
        if not placed:
            groups.append(box)
    return groups

digit_boxes = group_blobs(blob_boxes)

def resize_pad(img, size=64):
    h, w = img.shape
    scale = size / max(h, w)
    img = cv2.resize(img, (int(w*scale), int(h*scale)))
    top = (size - img.shape[0]) // 2
    bottom = size - img.shape[0] - top
    left = (size - img.shape[1]) // 2
    right = size - img.shape[1] - left
    return cv2.copyMakeBorder(
        img, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=255
    )

results = []

for (x0, y0, x1, y1) in digit_boxes:
    crop = gray[y0:y1+1, x0:x1+1]
    crop_mask = binary[y0:y1+1, x0:x1+1]

    ys, xs = np.where(crop_mask > 0)
    if len(xs) == 0:
        continue

    crop = crop[ys.min():ys.max()+1, xs.min():xs.max()+1]
    crop = resize_pad(crop, IMG_SIZE)
    crop = crop.astype("float32") / 255.0
    crop = np.expand_dims(crop, axis=-1)
    crop = np.expand_dims(crop, axis=0)

    pred = model.predict(crop, verbose=0)
    digit = int(np.argmax(pred))
    conf = float(np.max(pred))

    if conf < 0.7:
        continue

    results.append((digit, conf, x0, y0, x1, y1))

vis = img.copy()
for digit, conf, x0, y0, x1, y1 in results:
    cv2.rectangle(vis, (x0, y0), (x1, y1), (0,255,0), 2)
    cv2.putText(
        vis,
        f"{digit} ({conf:.2f})",
        (x0, max(y0 - 6, 15)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0,255,0),
        2
    )

plt.figure(figsize=(10,5))
plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()
