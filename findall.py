import cv2
import numpy

def calculate_contour_distance(contour1, contour2): 
    x1, y1, w1, h1 = cv2.boundingRect(contour1)
    c_x1 = x1 + w1/2
    c_y1 = y1 + h1/2

    x2, y2, w2, h2 = cv2.boundingRect(contour2)
    c_x2 = x2 + w2/2
    c_y2 = y2 + h2/2

    return max(abs(c_x1 - c_x2) - (w1 + w2)/2, abs(c_y1 - c_y2) - (h1 + h2)/2)

def merge_contours(contour1, contour2):
    return numpy.concatenate((contour1, contour2), axis=0)

def agglomerative_cluster(contours, threshold_distance=40.0):
    current_contours = contours
    while len(current_contours) > 1:
        min_distance = None
        min_coordinate = None

        for x in range(len(current_contours)-1):
            for y in range(x+1, len(current_contours)):
                distance = calculate_contour_distance(current_contours[x], current_contours[y])
                if min_distance is None:
                    min_distance = distance
                    min_coordinate = (x, y)
                elif distance < min_distance:
                    min_distance = distance
                    min_coordinate = (x, y)

        if min_distance < threshold_distance:
            index1, index2 = min_coordinate
            ind1 = current_contours.index(index1) 
            ind2 = current_contours.index(index2)
            what = merge_contours(current_contours.index(index1), current_contours.index(index2))
            del current_contours[index2]
        else: 
            break

    return current_contours
def agglomerative_cluster1(contours, threshold_distance=40.0):
    current_contours = list(contours)  # 👈 FIX

    while len(current_contours) > 1:
        min_distance = None
        min_coordinate = None

        for x in range(len(current_contours) - 1):
            for y in range(x + 1, len(current_contours)):
                distance = calculate_contour_distance(
                    current_contours[x],
                    current_contours[y]
                )

                if min_distance is None or distance < min_distance:
                    min_distance = distance
                    min_coordinate = (x, y)

        if min_distance < threshold_distance:
            index1, index2 = min_coordinate

            current_contours[index1] = merge_contours(
                current_contours[index1],
                current_contours[index2]
            )
            del current_contours[index2]
        else:
            break

    return current_contours

import numpy as np
def detect_objects(img): 
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    equalized_img = cv2.equalizeHist(gray_img)
    blurred_img = cv2.GaussianBlur(equalized_img, (9,9), 0)
    edges = cv2.Canny(blurred_img, 90, 180)
    ret, thresh = cv2.threshold(edges, 127, 255, 0)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
    print("contours after findContours: %s" % len(contours))

    #contours = take_biggest_contours(contours)
    #print("contours after take_biggest_contours: %s" % len(contours))

    contours = agglomerative_cluster1(contours)
    print("contours after agglomerative_cluster: %s" % len(contours))

    vis = img.copy()

    objects = []
    for c in contours:
        rect = cv2.boundingRect(c)
        x, y, w, h = rect

        rec = cv2.minAreaRect(c)
        box = cv2.boxPoints(rec)
        box = np.int32(box) 
        cv2.drawContours(vis,[box],0,(0,0,255),2) #red box
        
        # ignore small contours
        if w < 20 and h < 20:
            print("dropping rect due to small size", rect)
            continue

        objects.append(rect)

   

    return objects, vis

'''def resize(img):
    return cv2.resize(img, (int(img.shape[1]//3), int(img.shape[0]//3)))

img = cv2.imread("live_imgs/basler_capture_7.jpg")

obj, vis = detect_objects(img)

cv2.imshow('git', resize(vis) )
cv2.waitKey(0)
'''