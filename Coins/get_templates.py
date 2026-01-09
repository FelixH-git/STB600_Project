import cv2

def crop_box(image_path, x_start, y_start, x_end, y_end):
    img = cv2.imread(image_path)
    
    if img is None:
        print("Error: Could not load image.")
        return

    # 1. Validate: Ensure the crop is within image bounds
    # and start coordinates are smaller than end coordinates
    height, width = img.shape[:2]
    
    # Clamp coordinates to image dimensions just in case
    x_start = max(0, x_start)
    y_start = max(0, y_start)
    x_end = min(width, x_end)
    y_end = min(height, y_end)

    # 2. Perform the Crop
    # Slicing syntax: img[ y_start : y_end , x_start : x_end ]
    cropped_img = img[y_start:y_end, x_start:x_end]

    # 3. Check if crop is valid (not empty)
    if cropped_img.size == 0:
        print("Error: Resulting crop is empty. Check your coordinates.")
        return

    # 4. Show Result
    #cv2.imshow("Original", img)
    cv2.imshow("Cropped Box", cropped_img)
    
    cv2.waitKey(0)
    cv2.imwrite(r"C:\Users\biggi\Desktop\STB600_Project\Coins\5_MASK.png", cropped_img)
    cv2.destroyAllWindows()

# --- Usage ---
# We want a box starting at (100, 100) and ending at (400, 400)
# Top-Left: (100, 100)
# Bottom-Right: (400, 400)
crop_box(r'C:\Users\biggi\Desktop\STB600_Project\Coins\opening_result_Red.png', x_start=1127, y_start=1158, x_end=1262, y_end=1316)