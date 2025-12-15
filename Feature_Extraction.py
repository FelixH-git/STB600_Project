#functions to remove green pixels from an image
import cv2
import numpy as np

def resize(img, scale):
    return cv2.resize(img, (int(img.shape[1]/scale),int(img.shape[0]/scale)))

def Remove_Green(img_path, name):
    image = cv2.imread(img_path)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 80, 40])
    upper_green = np.array([85, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    mask_inv = cv2.bitwise_not(mask)
    result = cv2.bitwise_and(image, image, mask=mask_inv)
    
    tmp = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(tmp, 0, 255, cv2.THRESH_BINARY)
    b, g, r = cv2.split(result)
    rgba = [b, g, r, thresh]
    dst = cv2.merge(rgba, 4)
    cv2.imwrite(fr"Coins\no_green_{name}_coin.png", dst)



def Find_Color_in_Coin(img_path, name, lower_color, upper_color):
    image = cv2.imread(img_path)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    lower_yellow = np.array(lower_color)
    upper_yellow = np.array(upper_color)
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    result = cv2.bitwise_and(image, image, mask=mask)
    cv2.imwrite(rf"Coins\{name}_parts.png", result)


def Opening(img_path, name):
    image = cv2.imread(img_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
    cv2.imshow("thresh", resize(thresh, 4))
    cv2.waitKey(0)
    kernel = np.ones((5,5), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    cv2.imwrite(rf"Coins\opening_result_{name}.png", opening)

###### Dilate might not be necessary after opening ####

def Dilate(img_path, name):
    image = cv2.imread(img_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
    cv2.imshow("thresh", resize(thresh, 4))
    cv2.waitKey(0)
    kernel = np.ones((5,5), np.uint8)
    dilation = cv2.dilate(thresh, kernel, iterations=1)
    cv2.imwrite(rf"Coins\dilation_result_{name}.png", dilation)

######################################################################

if __name__ == "__main__":
    colors = ["Yellow", "Blue", "Red"]
    img_path = r"Images\Red.jpg"

    # for color in colors:
    #     Remove_Green(fr"Images\{color}.jpg", color)

    #     print(f"Saved Image {color} coin without green background.")
    

    #Find_Color_in_Coin(r"Coins\no_green_Yellow_coin.png", "Yellow", [0, 97, 24], [36, 255, 255])

    # Find_Color_in_Coin(r"Coins\no_green_Red_coin.png", "Red", [0, 97, 24], [36, 255, 255])
    # Opening(r"Coins\Red_parts.png", "Red")
    
    Find_Color_in_Coin(r"Coins\no_green_Blue_coin.png", "Blue", [216, 77, 24], [255, 255, 255])
    Opening(r"Coins\Blue_parts.png", "Blue")
    
    
    #Opening(r"Coins\Yellow_parts.png", "Yellow")