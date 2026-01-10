import cv2
import numpy as np

img = cv2.imread("symbol.jpg")
hsv = cv2.cvtColor(ref, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv, (15,80,80), (40,255,255))

cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnt_ref = max(cnts, key=cv2.contourArea)

np.save("symbol_contour.npy", cnt_ref)

