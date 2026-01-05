import cv2
import numpy as np

def split_contour_into_four(img, contour):
    x, y, w, h = cv2.boundingRect(contour)

    mid_x = x + w // 2
    mid_y = y + h // 2

    # Draw rectangles
    cv2.rectangle(img, (x, y), (mid_x, mid_y), (255, 0, 0), 2)       # TL
    cv2.rectangle(img, (mid_x, y), (x + w, mid_y), (0, 255, 0), 2)  # TR
    cv2.rectangle(img, (x, mid_y), (mid_x, y + h), (0, 0, 255), 2)  # BL
    cv2.rectangle(img, (mid_x, mid_y), (x + w, y + h), (255,255,0), 2) # BR

    return img



def draw_split_contours(img, split_contours):
    colors = [
        (255, 0, 0),   # blue
        (0, 255, 0),   # green
        (0, 0, 255),   # red
        (255, 255, 0)  # cyan
    ]

    for i, c in enumerate(split_contours):
        if c is not None:
            cv2.drawContours(img, [c], -1, colors[i], 2)

    return img

def split_contour_ladder(img, contour, steps=4):
    x, y, w, h = cv2.boundingRect(contour)

    step_h = h // steps

    for i in range(steps):
        y1 = y + i * step_h
        y2 = y + (i + 1) * step_h if i < steps - 1 else y + h

        cv2.rectangle(
            img,
            (x, y1),
            (x + w, y2),
            (0, 255 - i*40, 255),
            2
        )

    return img


def split_contour_ladder_longest_side(img, contour, steps=4):
    x, y, w, h = cv2.boundingRect(contour)

    if w > h:
        # Vertical ladder (long side = width)
        step = w // steps
        for i in range(steps):
            x1 = x + i * step
            x2 = x + (i + 1) * step if i < steps - 1 else x + w

            cv2.rectangle(
                img,
                (x1, y),
                (x2, y + h),
                (0, 255, 0),
                2
            )
    else:
        # Horizontal ladder (long side = height)
        step = h // steps
        for i in range(steps):
            y1 = y + i * step
            y2 = y + (i + 1) * step if i < steps - 1 else y + h

            cv2.rectangle(
                img,
                (x, y1),
                (x + w, y2),
                (255, 0, 0),
                2
            )

    return img

def detect_objects(img): 
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    equalized_img = cv2.equalizeHist(gray_img)
    blurred_img = cv2.GaussianBlur(equalized_img, (9,9), 0)
    edges = cv2.Canny(blurred_img, 90, 180)
    ret, thresh = cv2.threshold(edges, 127, 255, 0)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
    print("contours after findContours: %s" % len(contours))
    vis = img.copy()
    for c in contours:
        rect = cv2.boundingRect(c)
        x, y, w, h = rect

        rec = cv2.minAreaRect(c)
        box = cv2.boxPoints(rec)
        box = np.int32(box) 
        cv2.drawContours(vis,[box],0,(0,0,255),2) #red box
    
    return vis
    
def resize(img):
    return cv2.resize(img, (int(img.shape[1]//3), int(img.shape[0]//3)))

img = cv2.imread("live_imgs/basler_capture_7.jpg")

vis = detect_objects(img)

cv2.imshow('git', resize(vis) )
cv2.waitKey(0)