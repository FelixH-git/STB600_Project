

import argparse
import json
import os

import cv2
import numpy as np

from obj_fraud_detc import clean_img, find_contours, classify_object_color2


DEFAULT_RULES = [
    {"min": 120000, "max": 200000, "color": "red"},
    {"min": 210000, "max": 270000, "color": "yellow"},
    {"min": 280000, "max": 380000, "color": "blue"},
]


def resize_to_width(img, target_w=1280):
    h, w = img.shape[:2]
    if w <= target_w:
        return img
    scale = target_w / float(w)
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def dominant_non_black(colors):
    # colors is a sorted list of tuples [(name, count), ...], desc by count
    for name, cnt in colors:
        if name != "black":
            return name, cnt
    return None, 0


def annotate_objects(base_img, results):
    vis = base_img.copy()

    for obj in results:
        cnt = obj.get("cnt")
        area = int(obj.get("area", 0))
        colors = obj.get("colors", [])

        dom_name, dom_cnt = dominant_non_black(colors)

        if cnt is not None:
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int32(box)
            cv2.drawContours(vis, [box], 0, (0, 0, 255), 2)

            x, y, w, h = cv2.boundingRect(cnt)
            label = f"Area={area} | Dom={dom_name if dom_name else 'n/a'}"
            cv2.putText(vis, label, (x, max(0, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 220, 50), 2)

    return vis


def suggest_rules(results):
    # Aggregate area ranges by dominant color
    per_color = {}
    for obj in results:
        colors = obj.get("colors", [])
        area = float(obj.get("area", 0))
        dom, _ = dominant_non_black(colors)
        if not dom:
            continue
        rng = per_color.setdefault(dom, {"min": None, "max": None})
        rng["min"] = area if rng["min"] is None else min(rng["min"], area)
        rng["max"] = area if rng["max"] is None else max(rng["max"], area)

    # Convert to rules list with ints
    rules = []
    for color, rng in per_color.items():
        rules.append({"min": int(rng["min"]), "max": int(rng["max"]), "color": color})

    # Keep stable order (red, yellow, blue) if present
    order = {"red": 0, "yellow": 1, "blue": 2}
    rules.sort(key=lambda r: order.get(r["color"], 99))
    return rules


def overlay_rules(vis, rules, origin=(20, 30)):
    x, y = origin
    cv2.putText(vis, "Suggested Rules:", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    y += 28
    for r in rules:
        line = f"{r['color']}: {r['min']}..{r['max']}"
        cv2.putText(vis, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)
        y += 24
    return vis


def main():
    parser = argparse.ArgumentParser(description="Calibrate fraud area rules by visualizing detected objects.")
    parser.add_argument("--image", "-i", default="live_imgs2/basler_capture_0.jpg", help="Path to a sample frame")
    parser.add_argument("--save", "-s", default="calibrated_rules.json", help="Where to save suggested rules (press 's')")
    args = parser.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {args.image}")

    # Detect objects and classify colors
    mask = clean_img(img)
    find_vis, contours = find_contours(img, mask)
    results, color_vis = classify_object_color2(img, mask, contours)

    if not results:
        vis = img.copy()
        cv2.putText(vis, "No object detected", (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        vis = resize_to_width(vis)
        cv2.imshow("Calibration - Annotated", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    # Build annotated view and suggestions
    annotated = annotate_objects(img, results)
    rules = suggest_rules(results)
    annotated = overlay_rules(annotated, rules)

    # Compose side-by-side view
    left = resize_to_width(annotated, 900)
    right = resize_to_width(color_vis, 900)
    h = max(left.shape[0], right.shape[0])
    if left.shape[0] != h:
        left = cv2.copyMakeBorder(left, 0, h - left.shape[0], 0, 0, cv2.BORDER_CONSTANT)
    if right.shape[0] != h:
        right = cv2.copyMakeBorder(right, 0, h - right.shape[0], 0, 0, cv2.BORDER_CONSTANT)
    combo = np.hstack([left, right])

    print("Suggested rules (based on current image):")
    for r in rules:
        print(f"  - color={r['color']}, min={r['min']}, max={r['max']}")
    if not rules:
        print("  (No dominant colors found)")

    cv2.imshow("Calibration - Annotated | Color Classification", combo)
    print("Press 's' to save suggested rules, or 'q'/ESC to exit.")
    while True:
        k = cv2.waitKey(0) & 0xFF
        if k in (27, ord('q')):  # ESC or q
            break
        if k == ord('s'):
            with open(args.save, "w", encoding="utf-8") as f:
                json.dump(rules if rules else DEFAULT_RULES, f, indent=2)
            print(f"Saved rules to {os.path.abspath(args.save)}")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

