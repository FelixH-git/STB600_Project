import cv2
import numpy as np
import random
import os
class Generate_Data:
    def __init__(self):
        # 1. Load Background
        self.bg_path = r"Backround.jpg" # Make sure this matches your file extension (.png/.jpg)
        self._backround = cv2.imread(self.bg_path)
        
        if self._backround is None:
            raise FileNotFoundError(f"Could not load {self.bg_path}")
            
        h, w = self._backround.shape[:2]
        print(f"--- Background Size: {w}x{h} ---")
        
        # This mask tracks where coins already exist (White = Occupied, Black = Empty)
        self._occupied_mask = np.zeros((h, w), dtype=np.uint8)

        # 2. Load Coins (Isolate the coin, but don't damage pixel data)
        self._yellow_coin = self._extract_coin_content(r"Coins/no_green_Yellow_coin.png")
        self._blue_coin = self._extract_coin_content(r"Coins/no_green_Blue_coin.png")
        self._red_coin = self._extract_coin_content(r"Coins/no_green_Red_coin.png")

    def _extract_coin_content(self, path):
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"ERROR: Could not load {path}")
            return None

        # === CREATE ALPHA FROM BLACK BACKGROUND ===
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Anything NOT near-black becomes visible
        _, alpha = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)

        # Clean small noise
        kernel = np.ones((3, 3), np.uint8)
        alpha = cv2.morphologyEx(alpha, cv2.MORPH_OPEN, kernel)

        # Merge into BGRA
        b, g, r = cv2.split(img)
        img_bgra = cv2.merge((b, g, r, alpha))

        # === FIND TIGHT ROI USING ALPHA ===
        points = cv2.findNonZero(alpha)
        if points is None:
            return img_bgra

        x, y, w, h = cv2.boundingRect(points)
        coin_roi = img_bgra[y:y+h, x:x+w]

        print(f"Loaded {path}: ROI Found -> {w}x{h}")
        return coin_roi


    def rotate_image(self, image, angle):
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)

        # 1. Rotation Matrix
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # 2. Calculate New Canvas Size (so corners don't get cut off)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))

        # 3. Adjust Matrix for Translation
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]

        # 4. Perform Rotation
        rotated = cv2.warpAffine(image, M, (new_w, new_h), 
                                flags=cv2.INTER_LINEAR, 
                                borderMode=cv2.BORDER_CONSTANT, 
                                borderValue=(0,0,0,0))
        return rotated

    def add_coin_randomly(self, coin_img_original, max_attempts=200):
            if coin_img_original is None: return None # Changed to None

            bg_h, bg_w = self._backround.shape[:2]

            for attempt in range(max_attempts):
                # 1. Rotate the coin
                angle = random.randint(0, 360)
                coin_img = self.rotate_image(coin_img_original, angle)
                
                h, w = coin_img.shape[:2]
                
                # If coin is physically larger than background, we can't place it
                if w >= bg_w or h >= bg_h:
                    continue 

                # 2. Generate the "Hitbox" (Mask)
                if coin_img.shape[2] == 4:
                    _, coin_mask = cv2.threshold(coin_img[:, :, 3], 10, 255, cv2.THRESH_BINARY)
                else:
                    coin_mask = 255 * np.ones((h, w), dtype=np.uint8)

                # 3. Pick a random spot
                x = random.randint(0, bg_w - w)
                y = random.randint(0, bg_h - h)

                # 4. CHECK COLLISIONS
                bg_mask_region = self._occupied_mask[y:y+h, x:x+w]
                intersection = cv2.bitwise_and(bg_mask_region, coin_mask)


                if np.any(intersection):
                    continue 
                
                # 5. Success! Place image and update mask
                self._place_pixels(coin_img, x, y)
                self._occupied_mask[y:y+h, x:x+w] = cv2.bitwise_or(bg_mask_region, coin_mask)

                # cv2.imshow("occupied_mask", self._occupied_mask)
                # cv2.waitKey(0)

                # --- CHANGE: Return the Bounding Box ---
                # Returns: (x_min, y_min, width, height) in absolute pixels
                return (x, y, w, h)

            print("Failed to place coin (No valid spot found).")
            return None # Changed to None

    def _place_pixels(self, coin_img, x, y):
        h, w = coin_img.shape[:2]
        bg_slice = self._backround[y:y+h, x:x+w]
        coin_bgr = coin_img[:, :, :3]
        
        # Standard Alpha Blending
        if coin_img.shape[2] == 4:
            alpha = coin_img[:, :, 3] / 255.0
            alpha_inv = 1.0 - alpha
            
            for c in range(3):
                bg_slice[:, :, c] = (alpha * coin_bgr[:, :, c] + 
                                     alpha_inv * bg_slice[:, :, c])
        else:
            self._backround[y:y+h, x:x+w] = coin_bgr

    def save_image(self, filename):
        cv2.imwrite(filename, self._backround)
        # Optional: Save the mask to verify overlap logic works
        # cv2.imwrite("Debug_Map.png", self._occupied_mask)
        print(f"Saved {filename}")

if __name__ == "__main__":
    gen = Generate_Data()
    
    # Try to place multiple coins
    coins = [gen._yellow_coin, gen._blue_coin, gen._red_coin]
    

    def Create_Dataset(n_images, train_ratio=0.8):
        # 1. Setup Directory Structure
        base_dir = "dataset"
        sub_dirs = ["images/train", "images/test", "labels/train", "labels/test"]
        
        for sub in sub_dirs:
            os.makedirs(os.path.join(base_dir, sub), exist_ok=True)

        print(f"Generating {n_images} images...")

        for i in range(n_images):
            # 2. Reset the background for the new image
            # Make sure to reload the clean background every time
            gen._backround = cv2.imread("Backround.jpg", cv2.IMREAD_UNCHANGED)
            
            # 3. Decide if this image is for Training or Testing
            is_train = random.random() < train_ratio
            split_folder = "train" if is_train else "test"
            
            # List to hold all labels for this specific image
            image_labels = []
            
            # 4. Place random coins
            coin_amount = random.randint(1, 5)
            
            for _ in range(coin_amount):
                # Pick a random coin image from your list
                coin_idx = random.randint(0, len(coins) - 1)
                bbox = gen.add_coin_randomly(coins[coin_idx], max_attempts=1)
                
                if bbox is not None:
                    x, y, w, h = bbox
                    bg_h, bg_w = gen._backround.shape[:2]

                    # --- CALCULATE YOLO COORDINATES ---
                    center_x = x + (w / 2)
                    center_y = y + (h / 2) # FIXED: was y + (y/2)

                    norm_center_x = center_x / bg_w
                    norm_center_y = center_y / bg_h
                    norm_width = w / bg_w
                    norm_height = h / bg_h

                    # Format: class_id center_x center_y width height
                    # Assuming class_id is always 0 for 'coin'
                    label_line = f"0 {norm_center_x:.6f} {norm_center_y:.6f} {norm_width:.6f} {norm_height:.6f}"
                    image_labels.append(label_line)

            # 5. Save the Image and the Label File
            # We use 'i' for the filename (e.g., 0.png, 1.png)
            filename_base = str(i)
            
            # Save Image
            img_path = os.path.join(base_dir, "images", split_folder, f"{filename_base}.png")
            cv2.imwrite(img_path, gen._backround)
            
            # Save Labels (One .txt file per image, containing all coin lines)
            txt_path = os.path.join(base_dir, "labels", split_folder, f"{filename_base}.txt")
            with open(txt_path, "w") as f:
                f.write("\n".join(image_labels))
                
            if i % 10 == 0:
                print(f"Generated {i}/{n_images} images...")

        print("Dataset generation complete!")

    # Run it
    Create_Dataset(n_images=10)
    # count = 0
    # for coin in coins:
    #     if gen.add_coin_randomly(coin):
    #         count += 1
            
    # print(f"Total coins placed: {count}/{len(coins)}")