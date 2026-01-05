import cv2
import numpy as np

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
        '''if area < MINAREA:
            continue'''
        

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
        BBox = ()

    #print(f"Detected contours: {len(contours)}")
    return vis, contours

'''img_gaus = cv2.GaussianBlur(img, (3,3), 0)
    t_lower = 50  
    t_upper = 150 
    edge = cv2.Canny(img_gaus, t_lower, t_upper)
'''
def find(img):

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    equalized_img = cv2.equalizeHist(gray_img)
    blurred_img = cv2.GaussianBlur(equalized_img, (9,9), 0)
    edges = cv2.Canny(blurred_img, 0, 255)
    ret, thresh = cv2.threshold(edges, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    cv2.imshow("tresh", resize(thresh))
    cv2.waitKey(0)
    kernel = np.ones((5,5), np.uint8)

    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    cv2.imshow("opening", opening)
    cv2.waitKey(0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
    print("contours after findContours: %s" % len(contours))

    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)

        # skip very small contours (noise)
        '''if area < 100:
            continue'''

        x, y, w, h = cv2.boundingRect(cnt)

        # draw filled contour in green (optional)
        cv2.drawContours(img, [cnt], -1, (0, 255, 0), 2)

        '''# rotated rectangle (drawn in red)
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int32(box)
        cv2.drawContours(img, [box], 0, (0, 0, 255), 2)  # red box'''
    
    return img

def Opening(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
  
    kernel = np.ones((5,5), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    cv2.imshow("opening", resize(opening))
    cv2.waitKey(0)

def resize(img):
    return cv2.resize(img, (int(img.shape[1]//3), int(img.shape[0]//3)))




img = cv2.imread("live_imgs/basler_capture_7.jpg")

fin = find(img)

#Opening(img)

'''mask = clean_img(img)
vis, contours = find_contours(img, mask)
fin = find(img)
cv2.imshow('basler live feed', resize(vis))
cv2.imshow('git', resize(fin))
cv2.waitKey(0)'''