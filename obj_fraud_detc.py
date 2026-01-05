

import cv2
import numpy as np

title = '-'*50 
print(f"{title:>20}\n{'Fraud Detection':>30}\n{title:>20}")


MINAREA = 15000

def clean_img(img):
    # Convert to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Green background range
    lower_green = np.array([35, 80, 40])
    upper_green = np.array([85, 255, 255])

    # Mask green background
    green_mask = cv2.inRange(hsv, lower_green, upper_green)

    # Invert mask -> object becomes white
    object_mask = cv2.bitwise_not(green_mask)

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    object_mask = cv2.morphologyEx(object_mask, cv2.MORPH_CLOSE, kernel)
    object_mask = cv2.morphologyEx(object_mask, cv2.MORPH_OPEN, kernel)

    return object_mask


def find_contours(img, mask):
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    vis = img.copy()

    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area < MINAREA:
            continue
        

        x, y, w, h = cv2.boundingRect(cnt)

        cv2.drawContours(vis, [cnt], -1, (0, 255, 0), 2)
        #cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 255, 0), 2)

        #rotated rectangle
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int32(box) 
        cv2.drawContours(vis,[box],0,(0,0,255),2) #red box

        cv2.putText(
            vis,
            f"Area={int(area)}",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )
        print(f"Contour {i}: Area = {area}, BBox = ({x}, {y}, {w}, {h})")

    #print(f"Detected contours: {len(contours)}")
    return vis, contours



def classify_object_color2(img, mask, contours, min_area=MINAREA):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # HSV ranges + drawing colors (BGR)
    color_ranges = {
        "red":    ([0, 70, 50], [10, 255, 255], (0, 0, 255)),
        "red2":   ([170, 70, 50], [180, 255, 255], (0, 0, 255)),
        #"green":  ([35, 40, 40], [85, 255, 255], (0, 255, 0)),
        "blue":   ([90, 50, 50], [130, 255, 255], (255, 0, 0)),
        "yellow": ([20, 100, 100], [35, 255, 255], (0, 255, 255)),
        #"orange": ([10, 100, 100], [20, 255, 255], (0, 165, 255)),
        #"purple": ([130, 50, 50], [160, 255, 255], (255, 0, 255)),
        #"white":  ([0, 0, 200], [180, 50, 255], (255, 255, 255)),
        "black":  ([0, 0, 0], [180, 255, 50], (50, 50, 50)),
    }

    vis = img.copy()
    all_results = []

    for obj_id, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        # Mask for this contour
        contour_mask = np.zeros(mask.shape, dtype=np.uint8)
        cv2.drawContours(contour_mask, [cnt], -1, 255, -1)
        contour_mask = cv2.bitwise_and(contour_mask, mask)

        total_pixels = cv2.countNonZero(contour_mask)
        if total_pixels == 0:
            continue

        color_stats = {}

        for name, (lower, upper, draw_color) in color_ranges.items():
            lower = np.array(lower)
            upper = np.array(upper)

            color_mask = cv2.inRange(hsv, lower, upper)
            color_mask = cv2.bitwise_and(color_mask, contour_mask)

            count = cv2.countNonZero(color_mask)
            if count < 300:  # noise filter
                continue

            if name == "red2":
                color_stats["red"] = color_stats.get("red", 0) + count
                continue

            color_stats[name] = count

            # Compute centroid
            ys, xs = np.where(color_mask > 0)
            cx = int(xs.mean())
            cy = int(ys.mean())

            percent = (count / total_pixels) * 100

            # Draw centroid + label
            cv2.circle(vis, (cx, cy), 7, draw_color, -1)
            cv2.putText(
                vis,
                f"Obj {obj_id}: {name} {percent:.1f}%",
                (cx + 8, cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                draw_color,
                2
            )

        sorted_colors = sorted(color_stats.items(), key=lambda x: x[1], reverse=True)

        all_results.append({
            "object_id": obj_id,
            "area": area,
            "colors": sorted_colors,
            "cnt": cnt
        })

        print(f"\nObject {obj_id} | Area: {int(area)}")
        for c, cnt_px in sorted_colors:
            print(f"  {c}: {(cnt_px / total_pixels) * 100:.2f}%")

    return all_results, vis

def is_fraud(object_result, vis):
    """
    object_result:
    {
        'object_id': int,
        'area': float,
        'colors': [('black', px), ('red', px), ...],
        'cnt': cnt
    }
    """

    area = object_result["area"]
    colors = object_result["colors"]
    cnt = object_result["cnt"]
    x, y, w, h = cv2.boundingRect(cnt)


    # Remove black from consideration
    non_black = [(c, cnt) for c, cnt in colors if c != "black"]

    if not non_black:
        return True, "No color markings detected", vis

    # Dominant marking color
    dominant_color = max(non_black, key=lambda x: x[1])[0]

    # --- RULE SET ---
    rules = [
        {"min": 120000, "max": 200000, "color": "red"},
        {"min": 210000, "max": 270000, "color": "yellow"},
        {"min": 280000, "max": 380000, "color": "blue"},
    ]

    for rule in rules:
        # Compute centroid
        
        if rule["min"] <= area <= rule["max"]:
            if dominant_color == rule["color"]:
                
                return False, "Valid object", vis
            else:
                cv2.putText(
                vis,
                f"FRAUD",
                (x-10,y-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (0, 0, 255),
                2)
                return True, (
                    f"Expected {rule['color']} markings, "
                    f"found {dominant_color}"
                ), vis

    cv2.putText(
                vis,
                f"FRAUD",
                (x-10,y-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (0, 0, 255),
                2)
    return True, (f"Area does not match any known product size :{area}"), vis


def resize(img):
    return cv2.resize(img, (int(img.shape[1]//3), int(img.shape[0]//3)))

# -------------------------
# MAIN
# -------------------------
import time

prev_time = time.time()

def draw_fps(img):
    global prev_time

    curr_time = time.time()
    fps = 1.0 / (curr_time - prev_time)
    prev_time = curr_time

    h, w = img.shape[:2]

    cv2.putText(
        img,
        f"FPS: {fps:.2f}",
        (w - 180, 40),          # top-right
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2,
        cv2.LINE_AA
    )

    return img

def main(img):
    mask = clean_img(img)
    vis, contours = find_contours(img, mask)
    colors, color_vis = classify_object_color2(img, mask, contours)
    # No objects/colors detected: return annotated visualization instead of crashing
    if not colors:
        cv2.putText(
            vis,
            "No object detected",
            (40, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            2,
        )
        return vis

    fraud_vis = vis
    for obj in colors:
        try:
            fraud, reason, fraud_vis = is_fraud(obj, vis)
            print(f"Object {obj['object_id']}: {'FRAUD' if fraud else 'Not fraud'} – {reason}")
        except Exception as e:
            print(f"Error processing object {obj['object_id']}: {e}")

    fraud_vis = draw_fps(fraud_vis if fraud_vis is not None else vis)
    return fraud_vis

'''img = cv2.imread("imgs/Blue.jpg")
cv2.imshow("live",resize(main(img)))
cv2.waitKey(0)'''
'''img = cv2.imread("imgs/Blue.jpg")

mask = clean_img(img)
vis, contours = find_contours(img, mask)
colors, color_vis = classify_object_color2(img, mask, contours)

#print(colors)

for obj in colors:
    fraud, reason = is_fraud(obj)

    status = "FRAUD :(" if fraud else "Not fraud :)"
    print(f"Object {obj['object_id']}: {status} – {reason}")


cv2.imshow("Original Image", resize(img))
#cv2.imshow("Mask (object)", resize(mask))
cv2.imshow("Detected Object", resize(vis))
cv2.imshow("Color Classification", resize(color_vis))
cv2.waitKey(0)'''
