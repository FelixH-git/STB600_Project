'''
A simple Program for grabing video from basler camera and converting it to opencv img.
Tested on Basler acA1300-200uc (USB3, linux 64bit , python 3.5)
https://www.geeksforgeeks.org/clahe-histogram-eqalization-opencv/
'''
from pypylon import pylon
import cv2
import numpy as np
#from obj_fraud_detc import main
from liveCNN import main




def resize(img):
    return cv2.resize(img, (int(img.shape[1]//3), int(img.shape[0]//3)))




# conecting to the first available camera
camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())

# Grabing Continusely (video) with minimal delay
camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly) 
converter = pylon.ImageFormatConverter()

# converting to opencv bgr format
converter.OutputPixelFormat = pylon.PixelType_BGR8packed
converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
img_id=0
while camera.IsGrabbing():
    grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

    if grabResult.GrabSucceeded():
        # Access the image data
        image = converter.Convert(grabResult)
        img = image.GetArray()
        gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        vis = main(img)
        # Display the resulting frame
        cv2.imshow('basler live feed', resize(vis))
        
        #cv2.imshow('title', combined_img)
        k = cv2.waitKey(1)
        if k == 27:
            break
        elif k == 32:
            cv2.imwrite(f'live_imgs/basler_capture_{img_id}.jpg', img)
            img_id += 1
    grabResult.Release()
    
# Releasing the resource    
camera.StopGrabbing()

cv2.destroyAllWindows()