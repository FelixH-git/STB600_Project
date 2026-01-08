import cv2
import numpy as np
import math
from rotate import rotate_and_crop
from rotate import upsidedown
from pypylon import pylon


# Define color ranges in HSV for masking
COLOR_RANGES = {
    "blue":   ([90, 50, 50], [130, 255, 255]),
    "red1":   ([0, 50, 50], [10, 255, 255]),
    "red2":   ([160, 50, 50], [180, 255, 255]),
    "yellow": ([20, 100, 100], [35, 255, 255])
}

DRAW_COLORS = {
    "black": (0, 0, 0),
    "blue": (255, 0, 0),
    "red": (0, 0, 255),
    "yellow": (0, 255, 255)
}

STAR_WARS_NUMBERS = {
    0: ["SMALL"],
    1: ["SMALL", "SMALL"],
    2: ["SMALL", "SMALL", "SMALL"],
    3: ["SMALL", "SMALL", "SMALL", "SMALL"],
    4: ["MEDIUM", "SMALL"],
    5: ["MEDIUM", "SMALL", "SMALL"],
    6: ["MEDIUM", "SMALL", "SMALL", "SMALL"],
    7: ["BIG", "SMALL"],
    8: ["BIG", "SMALL", "SMALL"],
    9: ["BIG", "SMALL", "SMALL", "SMALL"]
}

DIST_THRESHOLD = 150

GROUP_COLORS = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
]

def euclidean(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def classify_shape(area):
    if area < 50 or area > 3000:
        return None
    if area < 190:
        return "SMALL"
    if area < 500:
        return "MEDIUM"
    return "BIG"

def classify_group(size_labels):
    size_labels_sorted = sorted(size_labels)
    for digit, pattern in STAR_WARS_NUMBERS.items():
        if sorted(pattern) == size_labels_sorted:
            return digit
    return None

def coin_value(color, digits):
    if not digits:
        return None
    if color == "blue":
        return int("".join(map(str, digits)))
    if color == "yellow":
        return int("".join(map(str, digits))) * 10
    if color == "red":
        value = 1
        for d in digits:
            value *= d
        return value
    return None


camera = pylon.InstantCamera(
    pylon.TlFactory.GetInstance().CreateFirstDevice()
)
camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

converter = pylon.ImageFormatConverter()
converter.OutputPixelFormat = pylon.PixelType_BGR8packed
converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned


while camera.IsGrabbing():
    grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

    if grabResult.GrabSucceeded():
        image = converter.Convert(grabResult)
        img = image.GetArray()

        img = cv2.resize(img, None, fx=0.3, fy=0.3, interpolation=cv2.INTER_AREA)

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_green = np.array([35, 50, 50])
        upper_green = np.array([85, 255, 255])
        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        mask_inv = cv2.bitwise_not(mask_green)
        img_no_green = cv2.bitwise_and(img, img, mask=mask_inv)

        gray = cv2.cvtColor(img_no_green, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)

        all_contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        coins = []

        for contour in all_contours:
            area = cv2.contourArea(contour)
            if 3000 < area < 2_000_000:
                x, y, w, h = cv2.boundingRect(contour)
                coin_img = rotate_and_crop(img, contour)

                cv2.imshow("Coin Crop", coin_img)
                cv2.waitKey(1)

                coins.append({
                    "contour": contour,
                    "image": coin_img,
                    "bbox": (x, y, w, h),
                })

        for coin in coins:
            coin_img = coin["image"]
            hsv = cv2.cvtColor(coin_img, cv2.COLOR_BGR2HSV)

            results = []
            coin_blobs = []

            for color_name in ["blue", "red", "yellow"]:
                if color_name == "red":
                    mask1 = cv2.inRange(hsv, *map(np.array, COLOR_RANGES["red1"]))
                    mask2 = cv2.inRange(hsv, *map(np.array, COLOR_RANGES["red2"]))
                    mask = cv2.bitwise_or(mask1, mask2)
                else:
                    mask = cv2.inRange(
                        hsv,
                        np.array(COLOR_RANGES[color_name][0]),
                        np.array(COLOR_RANGES[color_name][1])
                    )

                contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                centroids = []

                for i, c in enumerate(contours):
                    area = cv2.contourArea(c)
                    if 100 < area < 3000:
                        M = cv2.moments(c)
                        if M["m00"] == 0:
                            continue
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        centroids.append((i, (cx, cy)))

                groups = []
                for idx, (_, c) in enumerate(centroids):
                    for g in groups:
                        _, ref = centroids[g[0]]
                        if euclidean(c, ref) < DIST_THRESHOLD:
                            g.append(idx)
                            break
                    else:
                        groups.append([idx])

                for g_idx, group in enumerate(groups):
                    sizes, centers = [], []

                    for idx in group:
                        ci, (cx, cy) = centroids[idx]
                        area = cv2.contourArea(contours[ci])
                        s = classify_shape(area)
                        if s:
                            sizes.append(s)
                            centers.append((cx, cy))
                            coin_blobs.append(((cx, cy), s))

                    digit = classify_group(sizes)
                    if centers:
                        avg = np.mean(centers, axis=0).astype(int)
                        results.append({
                            "color": color_name,
                            "digit": digit,
                            "center": tuple(avg)
                        })

            if not results:
                continue

            coin_color = results[0]["color"]
            flipped = upsidedown(coin_img, coin_color, coin_blobs) if coin_color in ("blue", "yellow") else False

            results.sort(key=lambda r: r["center"][1])
            if flipped:
                results.reverse()

            coin["digits"] = [r["digit"] for r in results if r["digit"] is not None]
            coin["value"] = coin_value(coin_color, coin["digits"])

            label = f"Digits: {''.join(map(str, coin['digits']))}  Value: {coin['value']}"
            x, y, _, _ = coin["bbox"]

            cv2.putText(
                img, label, (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                (255, 255, 255), 2
            )

        cv2.imshow("Result", img)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

camera.StopGrabbing()
cv2.destroyAllWindows()
