import cv2
import numpy as np
import os
import random
import shutil
import random
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from tensorflow.keras.utils import image_dataset_from_directory


BASE_DIR = "templates" 
OUT_DIR = "dataset/train"
IMG_SIZE = 64
SAMPLES_PER_DIGIT = 1000

os.makedirs(OUT_DIR, exist_ok=True)

for digit in range(10):
    img = cv2.imread(f"{BASE_DIR}/{digit}.png", cv2.IMREAD_GRAYSCALE)
    img = 255 - img
    if img is None:
        raise ValueError(f"Missing {digit}.png")

    os.makedirs(f"{OUT_DIR}/{digit}", exist_ok=True)

    pad = int(2.5 * max(img.shape))
    img = cv2.copyMakeBorder(
        img,
        pad, pad, pad, pad,
        cv2.BORDER_CONSTANT,
        value=255
    )


    h, w = img.shape
    center = (w // 2, h // 2)

    for i in range(SAMPLES_PER_DIGIT):
        angle = random.uniform(-45, 45)
        scale = random.uniform(0.8, 1.2)
        tx = random.randint(-5, 5)
        ty = random.randint(-5, 5)

        M = cv2.getRotationMatrix2D(center, angle, scale)
        M[:, 2] += [tx, ty]
        

        aug = cv2.warpAffine(
            img, M, (w, h),
            flags=cv2.INTER_NEAREST,
            borderValue=255
        )


        ys, xs = np.where(aug < 255)
        if len(xs) < 50:
            continue
        aug = aug[ys.min():ys.max()+1, xs.min():xs.max()+1]

        aug = cv2.resize(aug, (IMG_SIZE, IMG_SIZE))

        cv2.imwrite(
            f"{OUT_DIR}/{digit}/{digit}_{i}.png",
            aug
        )

TRAIN = "dataset/train"
VAL = "dataset/val"

os.makedirs(VAL, exist_ok=True)

for digit in range(10):
    src_dir = f"{TRAIN}/{digit}"
    dst_dir = f"{VAL}/{digit}"

    os.makedirs(dst_dir, exist_ok=True)

    files = [
        f for f in os.listdir(src_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    random.shuffle(files)
    val_files = files[:50]

    for f in val_files:
        src = os.path.join(src_dir, f)
        dst = os.path.join(dst_dir, f)
        shutil.move(src, dst)


model = models.Sequential([
    layers.Input(shape=(64,64,1)),
    layers.Conv2D(16, 3, activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, activation="relu"),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(64, activation="relu"),
    layers.Dense(10, activation="softmax")
])


BATCH_SIZE = 32

train_ds = image_dataset_from_directory(
    "dataset/train",
    image_size=(64, 64),
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    label_mode="int"
)

val_ds = image_dataset_from_directory(
    "dataset/val",
    image_size=(64, 64),
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    label_mode="int"
)

# Normalize to [0,1]
train_ds = train_ds.map(lambda x, y: (x / 255.0, y))
val_ds = val_ds.map(lambda x, y: (x / 255.0, y))

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history = model.fit(
    train_ds,
    epochs=9,
    validation_data=val_ds
)

plt.figure(figsize=(8, 5))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

loss, acc = model.evaluate(val_ds)
print("Validation accuracy:", acc)

model.save("digit_cnn.keras")
