import cv2
import numpy as np

from obj_fraud_detc import main

def resize(img):
    return cv2.resize(img, (int(img.shape[1]//3), int(img.shape[0]//3)))


img = cv2.imread("live_imgs/basler_capture_7.jpg")

def canny(img):
    '''
    mulsti-staged edge detector
    Gaussain blur -> gradient calculation -> non-maximum suppression -> double thresholding -> edge tracking by hysteresis
        '''


    img_gaus = cv2.GaussianBlur(img, (3,3), 0)
    t_lower = 50  
    t_upper = 150 
    edge = cv2.Canny(img_gaus, t_lower, t_upper)
    return edge

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

#print(img.shape)
#template = img[767:837, 46: 108 ]

vis = main(img)
        # Display the resulting frame
cv2.imshow('basler live feed', resize(vis))
cv2.waitKey(0)



'''cleanimg= clean_img(img)

cv2.imshow("original", resize(img))
cv2.imshow("clean",resize(cleanimg))
cv2.waitKey(0)
'''

'''
vis = canny(img)

contours, _ = cv2.findContours(
        vis, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )


# convert the single-channel edge image back to BGR so colored drawings are visible
vis_color = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    
cv2.drawContours(vis_color, contours, -1, (0, 255, 0), 2)

cv2.imshow("original", resize(img))
cv2.imshow("contour", resize(vis_color))
cv2.waitKey(0)
'''