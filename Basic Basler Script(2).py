'''
Basler camera live feed with colored contour detection
Contours: black, blue, red, yellow
Green background removed
'''
from pypylon import pylon
import cv2
import numpy as np
import time
import math



# Connecting to the first available camera
camera = pylon.InstantCamera(
    pylon.TlFactory.GetInstance().CreateFirstDevice()
)

# Grab continuously with minimal delay
camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

converter = pylon.ImageFormatConverter()
converter.OutputPixelFormat = pylon.PixelType_BGR8packed
converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

# Resize helper
def resize(img, scale):
    return cv2.resize(
        img,
        (int(img.shape[1] / scale), int(img.shape[0] / scale))
    )

# FPS initialization
prev_time = time.time()

# Define color ranges in HSV for masking
COLOR_RANGES = {
    "blue":   ([90, 50, 50], [130, 255, 255]),
    "red1":   ([0, 50, 50], [10, 255, 255]),    # red split range
    "red2":   ([160, 50, 50], [180, 255, 255]),
    "yellow": ([20, 100, 100], [35, 255, 255])
}

# Contour colors for drawing (BGR)
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

DIST_THRESHOLD = 50

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

def classify_area(area):
    #print("AREA: ", area)
    if 100 < area < 190:
        return "SMALL"
    elif 330 < area < 500:
        return "MEDIUM"
    elif 500 < area < 700:
        return "BIG"
    return None

def classify_group(size_labels):
    size_labels_sorted = sorted(size_labels)

    for digit, pattern in STAR_WARS_NUMBERS.items():
        if sorted(pattern) == size_labels_sorted:
            return digit

    return None

def is_inside_coin(cx, cy, coin_contours):
    for coin in coin_contours:
        if cv2.pointPolygonTest(coin, (cx, cy), False) >= 0:
            return True
    return False


coint_amount = 0
color_masks = []
total_digits = []
digit_test = []

while camera.IsGrabbing():
    grabResult = camera.RetrieveResult(
        5000, pylon.TimeoutHandling_ThrowException
    )

    if grabResult.GrabSucceeded():
        # Convert image
        image = converter.Convert(grabResult)
        img = image.GetArray()

        # Resize for performance
        img = cv2.resize(img, None, fx=0.3, fy=0.3, interpolation=cv2.INTER_AREA)

        # Remove green background
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_green = np.array([35, 50, 50])
        upper_green = np.array([85, 255, 255])
        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        mask_inv = cv2.bitwise_not(mask_green)
        img_no_green = cv2.bitwise_and(img, img, mask=mask_inv)



        img_no_green_gray = cv2.cvtColor(img_no_green, cv2.COLOR_BGR2GRAY)

        _, binary = cv2.threshold(img_no_green_gray,0, 255, cv2.THRESH_BINARY)


        coin_contours, hierarchy_coins = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for contour in coin_contours:
            area = cv2.contourArea(contour)
         #   print(contour)
            if 3000 < area < 2000000:
                coint_amount += 1
                cv2.drawContours(img, [contour], -1, (255,255,255), 3)
        
        
        # Now find contours for each color
        hsv = cv2.cvtColor(img_no_green, cv2.COLOR_BGR2HSV)
        
        for color_name in ["red", "yellow", "blue"]:
            if color_name == "red":
                # Red has two HSV ranges
                lower1 = np.array(COLOR_RANGES["red1"][0])
                upper1 = np.array(COLOR_RANGES["red1"][1])
                lower2 = np.array(COLOR_RANGES["red2"][0])
                upper2 = np.array(COLOR_RANGES["red2"][1])

                mask1 = cv2.inRange(hsv, lower1, upper1)
                mask2 = cv2.inRange(hsv, lower2, upper2)
                mask = cv2.bitwise_or(mask1, mask2)
                color_masks.append(mask)
            else:
                lower = np.array(COLOR_RANGES[color_name][0])
                upper = np.array(COLOR_RANGES[color_name][1])
                mask = cv2.inRange(hsv, lower, upper)
                color_masks.append(mask)

            
            #mask = np.maximum.reduce(color_masks)
            
            cv2.imshow("mask", mask)

            contours, hierarchy = cv2.findContours(
                mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )

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
                    size_label = classify_area(area)

                    if size_label:
                        group_sizes.append(size_label)
                        group_centers.append((cx, cy))

                    # Draw contour + centroid
                    cv2.drawContours(img, [contour], -1, color, 1)
                    cv2.circle(img, (cx, cy), 3, color, -1)

                # -----------------------------
                # CLASSIFY DIGIT
                # -----------------------------

                group_items = list(zip(group_sizes, group_centers))

                group_items.sort(key=lambda x: x[1][0])

                sorted_sizes = [size for size, _ in group_items]

                digit = classify_group(group_sizes)

                #total_digits.append(digit)
                #print(total_digits)
                # Compute group label position (average centroid)
                if group_centers:
                    avg_x = int(sum(p[0] for p in group_centers) / len(group_centers))
                    avg_y = int(sum(p[1] for p in group_centers) / len(group_centers))
                    
                    label = str(digit) if digit is not None else "?"


                    if color_name == "red":
                        if len(groups) > 2:
                            if len(total_digits) > 3:
                                #print(total_digits)
                                print()
                                value = 0
                                for digit_idx in range(len(total_digits) - 1):
                                    value = total_digits[digit_idx] * total_digits[digit_idx + 1]
                            else:
                                value = 0
                            label = str(digit) + f": {value}" if digit is not None else "?"
                        else:
                            print()
                            #print(group_sizes)
                            #if len(total_digits) > 1:
                             #   print(total_digits)
                              #  value = total_digits[0] * total_digits[1]
                            #else:
                             #   value = 0
                            #label = str(digit) + f": VALUE OF COIN = {value}" if digit is not None else "?"
                    elif color_name == "blue":
                        if len(groups) > 3:
                            label = str(digit) + f": G_{g_idx % 3}" if digit is not None else "?"
                        else:
                            label = str(digit) + f": G_{g_idx}" if digit is not None else "?"
                    else:
                        if len(groups) > 3:
                            label = str(digit) + f": G_{g_idx % 3}" if digit is not None else "?"
                        else:
                            label = str(digit) + f": G_{g_idx}" if digit is not None else "?"
                        



                    cv2.putText(
                        img,
                        label,
                        (avg_x, avg_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        color,
                        2,
                        cv2.LINE_AA
                    )

        
        # FPS calculation
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time)
        prev_time = curr_time

        print("COINT AMOUNT:", coint_amount)
        coint_amount = 0
        color_masks = []
        total_digits = []
        cv2.putText(
            img,
            f"FPS: {fps:.2f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            2,
            cv2.LINE_AA
        )

        # Display
        cv2.imshow("basler live feed", resize(img, 1))

        key = cv2.waitKey(1)
        if key == 27:  # ESC
            break

    grabResult.Release()

# Cleanup
camera.StopGrabbing()
cv2.destroyAllWindows()
