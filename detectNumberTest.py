import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import math

MODEL_PATH = "digit_cnn.keras"
IMG_PATH = "Images/Blue.jpg"
IMG_SIZE = 64

model = load_model(MODEL_PATH)

img = cv2.imread(IMG_PATH)
if img is None:
    raise ValueError("Image not found")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

_, binary = cv2.threshold(
    gray, 0, 255,
    cv2.THRESH_BINARY + cv2.THRESH_OTSU
)

if np.mean(binary) > 127:
    binary = 255 - binary


kernel = np.ones((3, 3), np.uint8)
binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

plt.figure(figsize=(6, 6))
plt.imshow(binary, cmap="gray", vmin=0, vmax=255)
plt.title("Binary (Black blobs on White)")
plt.axis("off")
plt.show()

num_labels, labels = cv2.connectedComponents((binary == 0).astype(np.uint8))

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

def box_distance(a, b):
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b

    dx = max(bx0 - ax1, ax0 - bx1, 0)
    dy = max(by0 - ay1, ay0 - by1, 0)

    return math.hypot(dx, dy)


def group_blobs(boxes, dist_thresh=22):
    groups = []

    for box in boxes:
        merged = False

        for i, g in enumerate(groups):
            if box_distance(box, g) < dist_thresh:
                groups[i] = (
                    min(g[0], box[0]),
                    min(g[1], box[1]),
                    max(g[2], box[2]),
                    max(g[3], box[3]),
                )
                merged = True
                break

        if not merged:
            groups.append(box)

    return groups

digit_boxes = group_blobs(blob_boxes)

def resize_pad_bw(img, size=64):
    h, w = img.shape
    scale = size / max(h, w)
    img = cv2.resize(img, (int(w * scale), int(h * scale)))
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
    crop = binary[y0:y1+1, x0:x1+1]
    crop_bin = binary[y0:y1+1, x0:x1+1]

    ys, xs = np.where(crop_bin == 0)
    if len(xs) == 0:
        continue

    crop = crop[ys.min():ys.max()+1, xs.min():xs.max()+1]
    crop = resize_pad_bw(crop, IMG_SIZE)
    crop = crop.astype("float32") / 255.0
    crop = np.expand_dims(crop, axis=-1)
    crop = np.expand_dims(crop, axis=0)

    pred = model.predict(crop, verbose=0)
    digit = int(np.argmax(pred))
    conf = float(np.max(pred))

    plt.figure(figsize=(2,2))
    plt.imshow(crop.squeeze(), cmap="gray")
    plt.axis("off")
    plt.show()


    if conf < 0.7:
        continue

    results.append((digit, conf, x0, y0, x1, y1))

vis = img.copy()
for digit, conf, x0, y0, x1, y1 in results:
    cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 255, 0), 2)
    cv2.putText(
        vis,
        f"{digit} ({conf:.2f})",
        (x0, max(y0 - 5, 15)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

plt.figure(figsize=(10, 5))
plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()
