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

# img = cv2.imread("756.png")

# img_resized = cv2.resize(img, None, fx=0.3, fy=0.3)


# epic = process_image(img_resized)

# cv2.imshow("epic", epic)

# cv2.waitKey(0)



# # conecting to the first available camera
camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())

# Grabing Continusely (video) with minimal delay
camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly) 
converter = pylon.ImageFormatConverter()

# converting to opencv bgr format
converter.OutputPixelFormat = pylon.PixelType_BGR8packed
converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
img_id=0




#epic = IntegratedFraudDetector()

while camera.IsGrabbing():
    grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

    if grabResult.GrabSucceeded():
        # Access the image data
        image = converter.Convert(grabResult)
        img = image.GetArray()
        gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        

        epic = process_image(img_resized, show_contours=True)
        # Display the resulting frame
        cv2.imshow('basler live feed', epic)
        
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