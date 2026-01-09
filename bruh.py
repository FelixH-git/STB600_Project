'''
A simple Program for grabbing video from Basler camera and converting it to OpenCV img.
Tested on Basler acA1300-200uc (USB3, Linux 64bit, Python 3.5)
https://www.geeksforgeeks.org/clahe-histogram-eqalization-opencv/
'''

from pypylon import pylon
import cv2
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor

TEMPLATE_DIR = "templates"
MATCH_THRESHOLD = 0.7  # adjust as needed

# Load templates
templates = []
for filename in os.listdir(TEMPLATE_DIR):
    path = os.path.join(TEMPLATE_DIR, filename)
    if path.lower().endswith((".png", ".jpg", ".jpeg")):
        t_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        _, t_thresh = cv2.threshold(t_img, 30, 255, cv2.THRESH_OTSU)
        templates.append((filename, t_thresh))

# Connect to the first available camera
camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())

# Grab continuously (video) with minimal delay
camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
converter = pylon.ImageFormatConverter()
converter.OutputPixelFormat = pylon.PixelType_BGR8packed
converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

def resize(img, scale):
    return cv2.resize(img, (int(img.shape[1] / scale), int(img.shape[0] / scale)))

def match_template(template, thresh):
    """Run template matching and return top-left and bottom-right coordinates."""
    match = cv2.matchTemplate(thresh, template[1], cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(match)
    w, h = template[1].shape[::-1]
    top_left = max_loc
    bottom_right = (max_loc[0] + w, max_loc[1] + h)
    return [top_left, bottom_right]

count = 1
tmp = []

while camera.IsGrabbing():
    grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

    if grabResult.GrabSucceeded():
        # Access the image data
        image = converter.Convert(grabResult)
        img = image.GetArray()
        
        gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray_frame, 30, 255, cv2.THRESH_OTSU)

        # Multithread template matching every 100 frames
        if count > 0 and count % 1 == 0:
            tmp = []  # reset previous matches
            for template in templates:
                results = match_template(template, thresh)
                
                min_val, max_val, min_locl, max_loc = cv2.minMaxLoc(results)

        # Draw rectangles for all matched templates
        for value in tmp:
            cv2.rectangle(thresh, value[0], value[1], 255, 2)

        # Display the thresholded frame
        cv2.imshow("match", resize(thresh, 3))


        # Exit on ESC key
        if cv2.waitKey(1) == 27:
            break

    grabResult.Release()

# Releasing the resource
camera.StopGrabbing()
cv2.destroyAllWindows()
