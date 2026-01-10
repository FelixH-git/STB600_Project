import cv2
import numpy as np
import time
import math
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

# ==================== CONSTANTS ====================
MINAREA = 15000
KERNEL_SIZE = (9, 9)
NOISE_THRESHOLD = 300
DIST_THRESHOLD = 45

# HSV color ranges with BGR display colors
COLOR_RANGES = {
    "red": ([0, 50, 50], [10, 255, 255], (0, 0, 0)),
    "red2": ([160, 50, 60], [180, 255, 255], (0, 0, 0)),
    "blue": ([80, 50, 50], [130, 255, 255], (0, 0, 0)),
    "yellow": ([20, 100, 100], [35, 255, 255], (0,0,0)),
    "black": ([0, 0, 0], [180, 255, 50], (50, 50, 50)),
}

# Digit detection color ranges
DIGIT_COLOR_RANGES = {
    "blue": ([90, 50, 110], [150, 255, 255]),
    "red1": ([0, 100, 40],   [10, 255, 255]),
    "red2": ([170, 100, 40], [180, 255, 255]),
    "yellow": ([20, 100, 100], [35, 255, 255])
}

# Product validation rules


VALIDATION_RULES = [
    {"min": 15000, "max": 17000, "color": "red"},
    {"min": 21000, "max": 29000, "color": "yellow"},
    {"min": 29500, "max": 35000, "color": "blue"},
]

# Digit size classification
DIGIT_SIZE_RANGES = {
    "SMALL": (100, 190),
    "MEDIUM": (330, 500),
    "BIG": (500, 700),
}

# Star Wars numbering system
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

GROUP_COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255),
    (255, 255, 0), (255, 0, 255), (0, 255, 255),
]

# ==================== DATA CLASSES ====================
@dataclass
class ObjectResult:
    object_id: int
    area: float
    colors: List[Tuple[str, int]]
    contour: np.ndarray
    bbox: Tuple[int, int, int, int]
    is_fraud: bool = False
    fraud_reason: str = ""
    detected_digits: List[int] = None
    
    def __post_init__(self):
        if self.detected_digits is None:
            self.detected_digits = []

# ==================== UTILITY CLASSES ====================
class FPSCounter:
    def __init__(self):
        self.prev_time = time.time()
    
    def get_fps(self) -> float:
        curr_time = time.time()
        fps = 1.0 / (curr_time - self.prev_time)
        self.prev_time = curr_time
        return fps

class GeometryUtils:
    @staticmethod
    def euclidean_distance(p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])
    
    @staticmethod
    def classify_digit_size(contour: np.ndarray) -> Optional[str]:
        area = cv2.contourArea(contour)
        if 3000 < area < 50:
            return None 

        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = max(w, h) / max(1, min(w, h))
        print('Ratio: ',aspect_ratio)

        if 1 <= aspect_ratio < 1.3:
            return "SMALL"

        if 1.3 <= aspect_ratio < 2.7:
            return "MEDIUM"
        
        if 2.7 <= aspect_ratio < 4:
            return "BIG"
        return None
    
    @staticmethod
    def classify_digit_group(size_labels: List[str]) -> Optional[int]:
        size_labels_sorted = sorted(size_labels)
        for digit, pattern in STAR_WARS_NUMBERS.items():
            if sorted(pattern) == size_labels_sorted:
                return digit
        return None

# ==================== MAIN DETECTOR ====================
class IntegratedFraudDetector:
    def __init__(self, min_area=MINAREA, enable_digit_counting=True, resize_scale=0.3):
        self.min_area = min_area
        self.enable_digit_counting = enable_digit_counting
        self.resize_scale = resize_scale
        self.fps_counter = FPSCounter()
        self.geometry = GeometryUtils()
        self._print_header()
    
    @staticmethod
    def _print_header():
        title = '-' * 50
        print(f"{title:>20}\n{'Integrated Fraud Detection & Digit Counter':>45}\n{title:>20}")
    
    # ========== IMAGE PREPROCESSING ==========
    def clean_image(self, img: np.ndarray) -> np.ndarray:
        """Remove green background and return object mask."""
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        green_mask = cv2.inRange(hsv, np.array([35, 80, 40]), np.array([85, 255, 255]))
        object_mask = cv2.bitwise_not(green_mask)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, KERNEL_SIZE)
        object_mask = cv2.morphologyEx(object_mask, cv2.MORPH_CLOSE, kernel)
        object_mask = cv2.morphologyEx(object_mask, cv2.MORPH_OPEN, kernel)
       
        
        return object_mask
    
    def remove_green_background(self, img: np.ndarray) -> np.ndarray:
        """Remove green background for digit detection."""
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask_green = cv2.inRange(hsv, np.array([35, 50, 50]), np.array([85, 255, 255]))
        mask_inv = cv2.bitwise_not(mask_green)
        return cv2.bitwise_and(img, img, mask=mask_inv)
    
    # ========== CONTOUR DETECTION ==========
    def find_contours(self, img: np.ndarray, mask: np.ndarray) -> List:
        """Find and filter contours above minimum area."""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return [cnt for cnt in contours if cv2.contourArea(cnt) >= self.min_area]
    
    def draw_contours(self, img: np.ndarray, contours: List) -> np.ndarray:
        """Draw contours and bounding boxes."""
        vis = img.copy()
        for i, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            x, y, w, h = cv2.boundingRect(cnt)
            
            cv2.drawContours(vis, [cnt], -1, (0, 255, 0), 2)
            
            rect = cv2.minAreaRect(cnt)
            box = np.int32(cv2.boxPoints(rect))
            cv2.drawContours(vis, [box], 0, (0, 0, 255), 2)
            
            cv2.putText(vis, f"Area={int(area)}", (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            print(f"Contour {i}: Area = {area}, BBox = ({x}, {y}, {w}, {h})")
        
        return vis
    
    # ========== COLOR CLASSIFICATION ==========
    def _get_color_mask(self, hsv: np.ndarray, contour_mask: np.ndarray, 
                        color_name: str) -> Tuple[np.ndarray, int]:
        """Get mask and pixel count for a specific color."""
        lower, upper, _ = COLOR_RANGES[color_name]
        color_mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        color_mask = cv2.bitwise_and(color_mask, contour_mask)
        return color_mask, cv2.countNonZero(color_mask)
    
    def classify_colors(self, img: np.ndarray, mask: np.ndarray, 
                       contours: List) -> List[ObjectResult]:
        """Classify colors in detected objects."""
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        results = []
        
        for obj_id, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            
            contour_mask = np.zeros(mask.shape, dtype=np.uint8)
            cv2.drawContours(contour_mask, [cnt], -1, 255, -1)
            contour_mask = cv2.bitwise_and(contour_mask, mask)
            
            total_pixels = cv2.countNonZero(contour_mask)
            if total_pixels == 0:
                continue
            
            color_stats = {}
            
            for color_name in COLOR_RANGES.keys():
                _, count = self._get_color_mask(hsv, contour_mask, color_name)
                
                if count < NOISE_THRESHOLD:
                    continue
                
                if color_name == "red2":
                    color_stats["red"] = color_stats.get("red", 0) + count
                    continue
                
                color_stats[color_name] = count
            
            sorted_colors = sorted(color_stats.items(), key=lambda x: x[1], reverse=True)
            bbox = cv2.boundingRect(cnt)
            
            results.append(ObjectResult(obj_id, area, sorted_colors, cnt, bbox))
            
            print(f"\nObject {obj_id} | Area: {int(area)}")
            for c, cnt_px in sorted_colors:
                print(f"  {c}: {(cnt_px / total_pixels) * 100:.2f}%")
        
        return results
    
    # ========== FRAUD DETECTION ==========
    def check_fraud(self, obj: ObjectResult) -> Tuple[bool, str]:
        """Check if object is fraudulent based on area and color rules."""
        non_black_colors = [(c, cnt) for c, cnt in obj.colors if c != "black"]
        
        if not non_black_colors:
            return True, "No color markings detected"
        
        # Create color presence dict
        color_dict = {c: cnt for c, cnt in non_black_colors}
        
        # Check against validation rules
        for rule in VALIDATION_RULES:
            if rule["min"] <= obj.area <= rule["max"]:
                expected_color = rule["color"]
                
                # Check if expected color is present (even if not dominant)
                if expected_color in color_dict:
                    return False, "Valid object"
                
                # If expected color not present, it's fraud
                dominant_color = max(non_black_colors, key=lambda x: x[1])[0]
                return True, f"Expected {expected_color} markings, found {dominant_color}"
        
        return True, f"Area {obj.area:.0f} does not match any known product size"
    
    def get_expected_color_for_area(self, area: float) -> Optional[str]:
        """Get expected color based on area range."""
        for rule in VALIDATION_RULES:
            if rule["min"] <= area <= rule["max"]:
                return rule["color"]
        return None
    
    # ========== DIGIT COUNTING ==========
    def _get_digit_mask(self, hsv_roi: np.ndarray, color_name: str) -> np.ndarray:
        """Get color mask for digit detection."""
        if color_name == "red":
            lower1, upper1 = np.array(DIGIT_COLOR_RANGES["red1"][0]), np.array(DIGIT_COLOR_RANGES["red1"][1])
            lower2, upper2 = np.array(DIGIT_COLOR_RANGES["red2"][0]), np.array(DIGIT_COLOR_RANGES["red2"][1])
            mask1 = cv2.inRange(hsv_roi, lower1, upper1)
            mask2 = cv2.inRange(hsv_roi, lower2, upper2)
            return cv2.bitwise_or(mask1, mask2)
        else:
            lower, upper = np.array(DIGIT_COLOR_RANGES[color_name][0]), np.array(DIGIT_COLOR_RANGES[color_name][1])
            return cv2.inRange(hsv_roi, lower, upper)
    
    def _group_centroids(self, centroids: List[Tuple[int, Tuple[int, int]]]) -> List[List[int]]:
        """Group nearby centroids based on distance threshold."""
        groups = []
        for idx, (_, c) in enumerate(centroids):
            placed = False
            for group in groups:
                _, ref_c = centroids[group[0]]
                if self.geometry.euclidean_distance(c, ref_c) < DIST_THRESHOLD:
                    group.append(idx)
                    placed = True
                    break
            if not placed:
                groups.append([idx])
        return groups 
    
    def count_digits_in_object(self, img_no_green: np.ndarray, img: np.ndarray,
                               obj_contour_scaled: np.ndarray, color_name: str) -> Tuple[np.ndarray, List[int]]:
        """Count digits within a single object's bounding region."""
        obj_mask = np.zeros(img_no_green.shape[:2], dtype=np.uint8)
        cv2.drawContours(obj_mask, [obj_contour_scaled], -1, 255, -1)
        
        x, y, w, h = cv2.boundingRect(obj_contour_scaled)
        padding = 20
        x1, y1 = max(0, x - padding), max(0, y - padding)
        x2, y2 = min(img.shape[1], x + w + padding), min(img.shape[0], y + h + padding)
        
        roi = img[y1:y2, x1:x2].copy()
        roi_mask = obj_mask[y1:y2, x1:x2]

        roi_blur = cv2.medianBlur(roi, 3)
        hsv_roi = cv2.cvtColor(roi_blur, cv2.COLOR_BGR2HSV)
        mask = self._get_digit_mask(hsv_roi, color_name)
        mask = cv2.bitwise_and(mask, roi_mask)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


        h, s, v = cv2.split(hsv_roi)
        mask[s < 100] = 0



        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        


        centroids = []
        for i, contour in enumerate(contours):
            cv2.drawContours(mask, [contour], -1, (255,0,0), 1)

            cv2.imshow("mask", mask)
            area = cv2.contourArea(contour)
            if 80 < area < 3000:
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                    centroids.append((i, (cx, cy)))
        
        groups = self._group_centroids(centroids)
        
        detected_digits = []
        for g_idx, group in enumerate(groups):
            color = GROUP_COLORS[g_idx % len(GROUP_COLORS)]
            group_sizes = []
            group_centers = []
            
            for centroid_idx in group:
                contour_i, (cx, cy) = centroids[centroid_idx]
                contour = contours[contour_i]
                area = cv2.contourArea(contour)
                size_label = self.geometry.classify_digit_size(contour)
                
                if size_label:
                    group_sizes.append(size_label)
                    group_centers.append((cx, cy))
                
                cv2.drawContours(roi, [contour], -1, color, 1)
                cv2.circle(roi, (cx, cy), 3, color, -1)
            
            digit = self.geometry.classify_digit_group(group_sizes)
            if digit is not None and group_centers:
               
                
                avg_x = int(sum(p[0] for p in group_centers) / len(group_centers))
                avg_y = int(sum(p[1] for p in group_centers) / len(group_centers))
                
                axis, origin = self.get_object_axis(obj_contour_scaled)
                group_center = np.array([avg_x + x1, avg_y + y1], dtype=np.float32)
                projection = np.dot(group_center - origin, axis)
                detected_digits.append((digit, projection))
                cv2.putText(roi, f"{digit}", (avg_x, avg_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
        
        img[y1:y2, x1:x2] = roi
        return img, detected_digits
    

    def get_object_axis(self, contour: np.ndarray):
        """
        Returns unit vector representing object's main axis
        """
        data = contour.reshape(-1, 2).astype(np.float32)
        mean = np.mean(data, axis=0)
        data -= mean

        _, _, vt = np.linalg.svd(data)
        axis = vt[0]  # principal direction (long axis)

        axis /= np.linalg.norm(axis)
        return axis, mean

    # ========== VISUALIZATION ==========
    def draw_fraud_marker(self, img: np.ndarray, obj: ObjectResult):
        """Draw fraud marker on image."""
        cnt_scaled = (obj.contour * self.resize_scale).astype(np.int32)
        x, y, w, h = cv2.boundingRect(cnt_scaled)
        
        cv2.putText(img, "FRAUD", ((x + w) // 2, (y + h) // 2),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        rect = cv2.minAreaRect(cnt_scaled)
        box = np.int32(cv2.boxPoints(rect))
        cv2.drawContours(img, [box], 0, (0, 0, 255), 2)
    
    def draw_fps(self, img: np.ndarray) -> np.ndarray:
        """Draw FPS counter on image."""
        fps = self.fps_counter.get_fps()
        h, w = img.shape[:2]
        cv2.putText(img, f"FPS: {fps:.2f}", (w - 180, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
        return img
    
    def compute_object_value(self, detected_digits, color_name: str) -> Optional[int]:
        if not detected_digits:
            return None

        # Sort top → down
        detected_digits = sorted(detected_digits, key=lambda x: x[1])
        digits = [d for d, _ in detected_digits]

        if color_name == "red":
            value = 1
            for d in digits:
                value *= d
            return value

        elif color_name == "blue":
            return int("".join(map(str, digits)))

        elif color_name == "yellow":
            return int("".join(map(str, digits))) * 10

        return None

    
    # ========== MAIN PROCESSING ==========
    def process_frame(self, img: np.ndarray, show_contours: bool = False) -> np.ndarray:
        """Main processing pipeline for a single frame."""
        # Step 1: Detect objects
        mask = self.clean_image(img)
        contours = self.find_contours(img, mask)
        
        if show_contours:
            vis = self.draw_contours(img, contours)
        
        
        # Step 2: Classify colors
        objects = self.classify_colors(img, mask, contours)
        
        # if not objects:
        #     #img = cv2.resize(img, None, fx=self.resize_scale, fy=self.resize_scale)
        #     cv2.putText(img, "No object detected", (40, 60),
        #                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        #     return self.draw_fps(img)
        
        # Step 3: Prepare resized image
        #img = cv2.resize(img, None, fx=self.resize_scale, fy=self.resize_scale)
        img_no_green = self.remove_green_background(img)
        
        # Step 4: Process each object
        valid_count = 0
        fraud_count = 0
        
        for obj in objects:
            is_fraud, reason = self.check_fraud(obj)
            obj.is_fraud = is_fraud
            obj.fraud_reason = reason
            
            if is_fraud:
                fraud_count += 1
                print(f"✗ Object {obj.object_id}: FRAUD - {reason}")
                self.draw_fraud_marker(img, obj)
            else:
                valid_count += 1
                print(f"✓ Object {obj.object_id}: VALID - {reason}")
                
                if self.enable_digit_counting:
                    color_to_count = self.get_expected_color_for_area(obj.area)
                    if color_to_count:
                        cnt_scaled = (obj.contour * self.resize_scale).astype(np.int32)
                        img, digits = self.count_digits_in_object(
                            img_no_green, img, cnt_scaled, color_to_count
                        )
                        obj.detected_digits = digits

                        value = self.compute_object_value(digits, color_to_count)

                        if value is not None:
                            cnt_scaled = (obj.contour * self.resize_scale).astype(np.int32)
                            x, y, w, h = cv2.boundingRect(cnt_scaled)

                            cv2.putText(
                                img,
                                f"Value: {value}",
                                (x, y - 15),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.9,
                                (255, 255, 255),
                                2,
                                cv2.LINE_AA
                            )

                            print(f"   Final value ({color_to_count}): {value}")

        
        #Summary
        print(f"\n{'='*50}")
        print(f"Total objects: {len(objects)}")
        print(f"Valid objects: {valid_count}")
        print(f"Fraud objects: {fraud_count}")
        print(f"{'='*50}\n")
        
        return self.draw_fps(img)

# ==================== CONVENIENCE FUNCTIONS ====================
def process_image(img: np.ndarray, enable_digit_counting=True, show_contours=False) -> np.ndarray:
    """Process single image with fraud detection and optional digit counting."""
    detector = IntegratedFraudDetector(enable_digit_counting=enable_digit_counting, resize_scale=1.0)
    return detector.process_frame(img, show_contours)

def resize_image(img: np.ndarray, scale: float = 1/3) -> np.ndarray:
    """Resize image by scale factor."""
    return cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)))

