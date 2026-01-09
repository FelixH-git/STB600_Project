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


def main():
    img = cv2.imread("Blueup.png")

    # Remove green background
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 50, 50])
    upper_green = np.array([85, 255, 255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    mask_inv = cv2.bitwise_not(mask_green)
    img_no_green = cv2.bitwise_and(img, img, mask=mask_inv)



    img_no_green_gray = cv2.cvtColor(img_no_green, cv2.COLOR_BGR2GRAY)

    _, binary = cv2.threshold(img_no_green_gray,0, 255, cv2.THRESH_BINARY)


    all_contours, hierarchy_coins = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    coins = []

    for contour in all_contours:
        area = cv2.contourArea(contour)
    #   print(contour)
        if 3000 < area < 2000000:
            cv2.drawContours(img, [contour], -1, (255,255,255), 3)
            x, y, w, h = cv2.boundingRect(contour)
            coin_img = rotate_and_crop(img, contour)
            plt.imshow(coin_img, cmap="gray")
            plt.title("after crop")
            plt.axis("off")
            plt.show()
            coins.append({
                "id": len(coins),
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
                # Red has two HSV ranges
                lower1 = np.array(COLOR_RANGES["red1"][0])
                upper1 = np.array(COLOR_RANGES["red1"][1])
                lower2 = np.array(COLOR_RANGES["red2"][0])
                upper2 = np.array(COLOR_RANGES["red2"][1])

                mask1 = cv2.inRange(hsv, lower1, upper1)
                mask2 = cv2.inRange(hsv, lower2, upper2)
                mask = cv2.bitwise_or(mask1, mask2)
            else:
                lower = np.array(COLOR_RANGES[color_name][0])
                upper = np.array(COLOR_RANGES[color_name][1])
                mask = cv2.inRange(hsv, lower, upper)


            #cv2.imshow("mask", mask)
            # Find contours
            contours, hierarchy = cv2.findContours(
                mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )
            if not contours:
                continue


            # -----------------------------
            # FIRST PASS: collect centroids
            # -----------------------------
            centroids = []  # (contour_index, (cx, cy))

            for i, contour in enumerate(contours):
                area = cv2.contourArea(contour)

                if 100 < area < 3000 or 3000 < area < 200000:
                    M = cv2.moments(contour)
                    if M["m00"] == 0:
                        continue

                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])

                    centroids.append((i, (cx, cy)))

            # -----------------------------
            # SECOND PASS: group centroids
            # -----------------------------
            groups = []  # list of lists (indices into centroids list)

            for idx, (_, c) in enumerate(centroids):
                placed = False

                for group in groups:
                    _, ref_c = centroids[group[0]]
                    if euclidean(c, ref_c) < DIST_THRESHOLD:
                        group.append(idx)
                        placed = True
                        break

                if not placed:
                    groups.append([idx])

            # -----------------------------
            # DRAW RESULTS
            # -----------------------------
            
            for g_idx, group in enumerate(groups):
                color = GROUP_COLORS[g_idx % len(GROUP_COLORS)]

                group_sizes = []
                group_centers = []

                for centroid_idx in group:
                    contour_i, (cx, cy) = centroids[centroid_idx]
                    contour = contours[contour_i]

                    area = cv2.contourArea(contour)
                    size_label = classify_shape(area)

                    

                    if size_label:
                        group_sizes.append(size_label)
                        group_centers.append((cx, cy))

                for center, size in zip(group_centers, group_sizes):
                    coin_blobs.append((center, size))

            
            


                # -----------------------------
                # CLASSIFY DIGIT
                # -----------------------------
                digit = classify_group(group_sizes)

                # Compute group label position (average centroid)
                if group_centers:
                    avg_x = int(sum(p[0] for p in group_centers) / len(group_centers))
                    avg_y = int(sum(p[1] for p in group_centers) / len(group_centers))

                    label = str(digit) + f": G_{g_idx}" if digit is not None else "?"

                    xs = [p[0] for p in group_centers]
                    ys = [p[1] for p in group_centers]

                    min_x, max_x = min(xs), max(xs)
                    min_y, max_y = min(ys), max(ys)

                    # Place text slightly to the right of the group
                    text_x = max_x + 30
                    text_y = min_y + 30

                    # Clamp inside image
                    h, w = img.shape[:2]
                    text_x = min(text_x, w - 40)
                    text_y = max(text_y, 20)

                    

                    results.append({
                    "color": color_name,
                    "group_id": g_idx,
                    "digit": digit,
                    "center": (avg_x, avg_y)
                    })

            coin_color = results[0]["color"]

            needs_order = coin_color in ("blue", "yellow")
            if needs_order:
                flipped = upsidedown(coin_img, coin_color, coin_blobs)
            else:
                flipped = False

            # Order digits top to bottom
            results.sort(key=lambda r: r["center"][1])

            if flipped:
                results.reverse()

            # SAVE TO THIS COIN
            coin["digits"] = [r["digit"] for r in results]
            value = coin_value(coin_color, coin["digits"])
            coin["value"] = value


            digits_str = "".join(map(str, coin["digits"]))
            label = f"Digits: {digits_str} - Value: {coin['value']}"


            x, y, w, h = coin["bbox"]
            top_point = min(contour, key=lambda p: p[0][1])  # smallest y
            tx = top_point[0][0]
            ty = top_point[0][1]

            cv2.putText(
                img,
                label,
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2
            )






# plt.imshow(img, cmap="gray")
# plt.title("Results")
# plt.axis("off")
# plt.show()