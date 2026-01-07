import cv2
import numpy as np
import argparse
import os


def resize(img, scale=0.33):
    h, w = img.shape[:2]
    return cv2.resize(img, (int(w * scale), int(h * scale)))


def acquire_image(capture=False, src_path='live_imgs/basler_capture_2.jpg', save_to='live_imgs/captured.jpg'):
    """Stage 1: Acquire image from file or webcam (optional). Returns BGR image."""
    if capture:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError('Cannot open camera')
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise RuntimeError('Failed to capture from camera')
        cv2.imwrite(save_to, frame)
        return frame
    else:
        img = cv2.imread(src_path)
        if img is None:
            raise FileNotFoundError(f'Image not found: {src_path}')
        return img


def preprocess(img):
    """Stage 2: Denoise and do morphological opening. Returns BGR and HSV versions."""
    # Denoise: median then Gaussian
    den = cv2.medianBlur(img, 5)
    den = cv2.GaussianBlur(den, (5, 5), 0)

    # Morphological opening in BGR converted to gray for structure
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    gray = cv2.cvtColor(den, cv2.COLOR_BGR2GRAY)
    opened = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)

    # Keep denoised color image and its HSV
    hsv = cv2.cvtColor(den, cv2.COLOR_BGR2HSV)
    return den, hsv, opened


def get_color_ranges():
    # HSV ranges for common colors. May need tuning for your lighting/camera.
    return {
        'red': [((0, 70, 50), (10, 255, 255)), ((170, 70, 50), (180, 255, 255))],
        'green': [((36, 25, 25), (86, 255, 255))],
        'blue': [((94, 80, 2), (126, 255, 255))],
        'yellow': [((15, 100, 100), (35, 255, 255))]
    }


def segment_by_color(hsv, color_ranges=None):
    """Stage 3: Return dict of binary masks per color and combined mask."""
    if color_ranges is None:
        color_ranges = get_color_ranges()
    masks = {}
    combined = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for cname, ranges in color_ranges.items():
        mask_total = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lo, hi in ranges:
            lo = np.array(lo, dtype=np.uint8)
            hi = np.array(hi, dtype=np.uint8)
            mask = cv2.inRange(hsv, lo, hi)
            mask_total = cv2.bitwise_or(mask_total, mask)
        # clean up small noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_total = cv2.morphologyEx(mask_total, cv2.MORPH_OPEN, kernel)
        mask_total = cv2.morphologyEx(mask_total, cv2.MORPH_CLOSE, kernel)
        masks[cname] = mask_total
        combined = cv2.bitwise_or(combined, mask_total)
    # Optional: further clean combined
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)))
    return masks, combined


def extract_features(original_bgr, masks, combined_mask, min_area=100):
    """Stage 4: Find contours and compute features and per-contour color overlaps."""
    # find contours on combined mask
    cnts, hierarchy = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    features = []
    h, w = combined_mask.shape[:2]
    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        x, y, cw, ch = cv2.boundingRect(cnt)
        M = cv2.moments(cnt)
        if M.get('m00', 0) != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
        else:
            cx, cy = x + cw // 2, y + ch // 2

        # create mask for this single contour
        contour_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(contour_mask, [cnt], -1, 255, -1)

        # compute overlap with each color mask
        overlaps = {}
        for cname, cmask in masks.items():
            overlap = cv2.bitwise_and(cmask, contour_mask)
            overlaps[cname] = int(np.count_nonzero(overlap))

        # mean color inside contour (BGR)
        mean_val = cv2.mean(original_bgr, mask=contour_mask)  # returns (b,g,r,alpha)

        features.append({
            'contour': cnt,
            'area': area,
            'bbox': (x, y, cw, ch),
            'centroid': (cx, cy),
            'overlaps': overlaps,
            'mean_bgr': (mean_val[2], mean_val[1], mean_val[0])  # r,g,b for readability
        })
    return features


def classify_and_annotate(img, features, out_path='live_imgs/annotated.jpg'):
    """Stage 5: Classify each feature by color (highest overlap) and annotate the image."""
    annotated = img.copy()
    summary = []
    for f in features:
        overlaps = f['overlaps']
        # pick label with max overlap
        label = 'unknown'
        if overlaps:
            label = max(overlaps.items(), key=lambda kv: kv[1])[0]
            if overlaps[label] == 0:
                label = 'unknown'

        x, y, w, h = f['bbox']
        cx, cy = f['centroid']
        area = f['area']

        # draw box and centroid
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.circle(annotated, (cx, cy), 4, (0, 0, 255), -1)
        cv2.putText(annotated, f'{label} {int(area)}', (x, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        summary.append({'label': label, 'area': area, 'centroid': (cx, cy), 'bbox': (x, y, w, h)})

    cv2.imwrite(out_path, annotated)
    return annotated, summary


def main():
    parser = argparse.ArgumentParser(description='Pipeline: acquisition -> preprocess -> segment -> features -> classify')
    parser.add_argument('--capture', action='store_true', help='Capture from webcam instead of loading file')
    parser.add_argument('--src', default='live_imgs/basler_capture_2.jpg', help='Source image path')
    parser.add_argument('--out', default='live_imgs/annotated.jpg', help='Output annotated image path')
    args = parser.parse_args()

    # Stage 1
    img = acquire_image(capture=args.capture, src_path=args.src)

    # Stage 2
    den, hsv, opened = preprocess(img)

    # Stage 3
    color_ranges = get_color_ranges()
    masks, combined = segment_by_color(hsv, color_ranges=color_ranges)

    # Stage 4
    features = extract_features(den, masks, combined)

    # Stage 5
    annotated, summary = classify_and_annotate(img, features, out_path=args.out)

    # Display small preview and print summary
    print('Detected objects:', len(summary))
    for i, s in enumerate(summary, 1):
        print(f"{i}: label={s['label']}, area={int(s['area'])}, centroid={s['centroid']}")

    cv2.imshow('annotated', resize(annotated))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
import cv2
import numpy as np

def resize(img):
    return cv2.resize(img, (int(img.shape[1]//3), int(img.shape[0]//3)))


img = cv2.imread("live_imgs/basler_capture_2.jpg",cv2.IMREAD_GRAYSCALE)

def sobel(img):
    '''

        similar to prewitt but the kernel here gives extra wieghts to the
        center pixel row making it better than prewitt 

        The center row/column has weight 2 
    '''

    img_gaus = cv2.medianBlur(img, 3, 0)
    #cv2.imshow('median', resize(img4_gaus))


    ''' this kernel is aparantly used internally in open cv
    x_axis= np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    y_axisy = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
    '''
    sobel_x = cv2.Sobel(img_gaus, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img_gaus, cv2.CV_64F, 0, 1, ksize=3)

    abs_sobel_x = cv2.convertScaleAbs(sobel_x)
    abs_sobel_y = cv2.convertScaleAbs(sobel_y)

    sobel_combined = cv2.addWeighted(abs_sobel_x, 0.5, abs_sobel_y, 0.5, 0)

    """cv2.imshow('Sobel X', resize(abs_sobel_x))
    cv2.imshow('Sobel Y', resize(abs_sobel_y))
    cv2.imshow('Sobel Combined', resize(sobel_combined))"""

   
    return sobel_combined

def Opening(img):

    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #_, thresh = cv2.threshold(img, cv2.MORPH_OPEN,   cv2.THRESH_BINARY)
  
    kernel = np.ones((5,5), np.uint8)
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    return opening


#cv2.imshow('original', resize(img))


'''first_vis = sobel(img)
cv2.imshow('sobel', resize(first_vis))'''
'''img_copy = img.copy()
open = Opening(img)
vis = sobel(open)

_, th = cv2.threshold(vis, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

#find contours
contours, hierarchy = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print("Number of contours found:", len(contours))

from tests.findall import agglomerative_cluster1




for i, cnt  in enumerate(contours):

    cv2.drawContours(img_copy, [cnt], 0, (255,0,0), 2) #contours in blue

cv2.imshow('contours', resize(img_copy))
cv2.waitKey(0)'''