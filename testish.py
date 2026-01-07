import cv2
import numpy as np

def resize(img):
    return cv2.resize(img, (int(img.shape[1]//3), int(img.shape[0]//3)))




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

for i in range(0,1):
    print(f"Processing image {i} \n")
    img = cv2.imread(f"live_imgs/basler_capture_{i}.jpg")
    img_copy = img.copy()

    open = Opening(img)
    vis = sobel(open)

    cv2.imshow('sobel', resize(vis))
    

    grey = cv2.cvtColor(vis, cv2.COLOR_BGR2GRAY)

    _, th = cv2.threshold(grey, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    #find contours
    contours, hierarchy = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print("Number of contours found:", len(contours))

    nr = 0
    for i, cnt  in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area < 300:
            continue

        cv2.drawContours(img_copy, [cnt], 0, (0,0,255), 2) #contours in red
        nr += 1
        print("Contour %d: Area = %f" % (i, area) )

    print("Number of contours after area filtering:", nr)
    cv2.imshow('contours', resize(img_copy))
    cv2.waitKey(0)