import cv2
import numpy as np

def resize(img):
    return cv2.resize(img, (int(img.shape[1]//3), int(img.shape[0]//3)))


img = cv2.imread("live_imgs/basler_capture_7.jpg", cv2.IMREAD_GRAYSCALE)

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

vis = canny(img)

contours, _ = cv2.findContours(
        vis, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

# convert the single-channel edge image back to BGR so colored drawings are visible
vis_color = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)

        # skip very small contours (noise)
        '''if area < 100:
            continue'''

        x, y, w, h = cv2.boundingRect(cnt)

        # draw filled contour in green (optional)
        cv2.drawContours(vis_color, [cnt], -1, (0, 255, 0), 2)

        # rotated rectangle (drawn in red)
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int32(box)
        cv2.drawContours(vis_color, [box], 0, (0, 0, 255), 2)  # red box

        # compute area of the rotated rectangle (width * height)
        rect_w, rect_h = rect[1][0], rect[1][1]
        rect_area = float(rect_w * rect_h)

        # polygon area from the 4 corner points (should be similar)
        poly_area = float(cv2.contourArea(box.astype(np.float32)))

        # draw rectangle area text just above the bounding box
        text_pos = (x, y - 10 if y - 10 > 10 else y + 10)
        cv2.putText(
            vis_color,
            f"rect_area={rect_area:.1f}",
            text_pos,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA
        )

        # also print values to console for debugging
        print(f"Contour {i}: contour_area={area:.1f}, rect_area={rect_area:.1f}, poly_area={poly_area:.1f}, BBox = ({x}, {y}, {w}, {h})")

        cv2.imshow("contour", resize(vis_color))
        cv2.waitKey(0)

print(f"Detected contours: {len(contours)}")

'''
silver 

Contour 2: contour_area=4224.5, rect_area=4504.8, poly_area=4502.0, BBox = (837, 767, 108, 46)
Contour 3: contour_area=14.5, rect_area=4777.2, poly_area=4789.0, BBox = (628, 759, 109, 46)
Contour 4: contour_area=1710.0, rect_area=1845.9, poly_area=1810.0, BBox = (417, 752, 44, 44)
Contour 5: contour_area=1601.0, rect_area=1763.0, poly_area=1763.0, BBox = (840, 705, 42, 44)
Contour 6: contour_area=1624.5, rect_area=1764.0, poly_area=1764.0, BBox = (695, 699, 43, 43)
Contour 7: contour_area=1644.5, rect_area=1763.0, poly_area=1763.0, BBox = (631, 696, 42, 44)
Contour 8: contour_area=1721.5, rect_area=1892.0, poly_area=1892.0, BBox = (419, 690, 45, 44)
Contour 9: contour_area=4529.5, rect_area=13431.5, poly_area=13389.0, BBox = (236, 688, 145, 101)

'''