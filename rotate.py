import cv2
import numpy as np
import math
import matplotlib.pyplot as plt



def long_edge_vertical_angle(rect):
    (_, _), (w, h), angle = rect
    if w > h:
        return -angle
    else:
        return -(angle - 90)

def rotate_and_crop(img, cnt):
    rect = cv2.minAreaRect(cnt)
    angle = long_edge_vertical_angle(rect)
    center = rect[0]

    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        img, M, img.shape[1::-1],
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )

    box = cv2.boxPoints(rect)
    box = np.int32(box)

    ones = np.ones((4, 1))
    pts = np.hstack([box, ones])
    pts = (M @ pts.T).T.astype(int)

    x, y, w, h = cv2.boundingRect(pts)
    x = max(0, x)
    y = max(0, y)
    x2 = min(rotated.shape[1], x + w)
    y2 = min(rotated.shape[0], y + h)

    return rotated[y:y2, x:x2]

def upsidedown(coin, mode, blobs):


    strong = [((cx, cy), size)
              for (cx, cy), size in blobs
              if size in ("MEDIUM", "BIG")]
    
    if strong:
        print("Stroing")
        xs = [cx for (cx, cy), _ in strong]
        center_x = sum(xs) / len(xs)

        # MEDIUM/BIG should be on the LEFT
        on_right = sum(cx > center_x for cx in xs)

        return on_right < len(xs) / 2

    elif mode == "blue":
        gray = cv2.cvtColor(coin, cv2.COLOR_BGR2GRAY)
        _, dark = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY_INV)
        h = dark.shape[0]
        slices = [
            dark[0:h//4, :],
            dark[h//4:h//2, :],
            dark[h//2:3*h//4, :],
            dark[3*h//4:h, :]
        ]
        plt.imshow(dark, cmap="gray")
        plt.title("Dark mask rotate")
        plt.axis("off")
        plt.show()
        scores = [cv2.countNonZero(s) for s in slices]
        
        for i, score in enumerate(scores):
            print(f"Slice {i}: {score}")
            
        # If bottom quarter has more dark pixels than top, the coin is upside down
        if scores[0] < scores[3]:
            coin = cv2.rotate(coin, cv2.ROTATE_180)
            return True
        
    elif mode == "yellow":
        return False
            

        
    return False



    



def main():

    img = cv2.imread("Blueup.png")  
    out = process_image(img)

    cv2.imshow("result", out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()