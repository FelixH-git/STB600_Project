from pypylon import pylon
import cv2
import numpy as np
import os

# ------------------ CONFIG ------------------
TEMPLATE_DIR = r"C:\Users\biggi\Desktop\STB600_Project\Templates"
MATCH_THRESHOLD = 0.7
DISPLAY_SCALE = 3

# ------------------ UTILS ------------------
def resize(img, scale):
    return cv2.resize(
        img,
        (int(img.shape[1] / scale), int(img.shape[0] / scale))
    )

def load_templates(folder):
    templates = {}
    for file in os.listdir(folder):
        if file.endswith(".png") or file.endswith(".jpg"):
            path = os.path.join(folder, file)
            tmpl = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            _, tmpl = cv2.threshold(tmpl, 10, 255, cv2.THRESH_BINARY)
            templates[file] = tmpl
    return templates

# ------------------ IMAGE PROCESSING ------------------
def remove_green_and_open(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_green = np.array([35, 80, 40])
    upper_green = np.array([85, 255, 255])

    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    mask_inv = cv2.bitwise_not(green_mask)

    no_green = cv2.bitwise_and(frame, frame, mask=mask_inv)

    gray = cv2.cvtColor(no_green, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY)
    cv2.imshow("gray", resize(thresh, 3))
    

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None, None

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)

    coin = no_green[y:y+h, x:x+w]
    coin_gray = cv2.cvtColor(coin, cv2.COLOR_BGR2GRAY)
    _, coin_thresh = cv2.threshold(coin_gray, 0, 255, cv2.THRESH_BINARY)

    kernel = np.ones((9, 9), np.uint8)
    opening = cv2.morphologyEx(coin_thresh, cv2.MORPH_OPEN, kernel)

    cv2.imshow("opening", opening)


    return opening, (x, y, w, h)

# ------------------ TEMPLATE MATCHING ------------------
def match_template(binary_coin, template):
    if binary_coin.shape[0] < template.shape[0] or \
       binary_coin.shape[1] < template.shape[1]:
        return None, 0.0

    result = cv2.matchTemplate(
        binary_coin, template, cv2.TM_CCOEFF_NORMED
    )

    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    h, w = template.shape

    return (max_loc, (w, h)), max_val

# ------------------ BASLER CAMERA ------------------
camera = pylon.InstantCamera(
    pylon.TlFactory.GetInstance().CreateFirstDevice()
)
camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

converter = pylon.ImageFormatConverter()
converter.OutputPixelFormat = pylon.PixelType_BGR8packed
converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

# ------------------ LOAD TEMPLATES ------------------
templates = load_templates(TEMPLATE_DIR)

print("[INFO] Templates loaded:", list(templates.keys()))
print("[INFO] Press ESC to exit")

# ------------------ MAIN LOOP ------------------
while camera.IsGrabbing():
    grabResult = camera.RetrieveResult(
        5000, pylon.TimeoutHandling_ThrowException
    )

    if grabResult.GrabSucceeded():
        image = converter.Convert(grabResult)
        frame = image.GetArray()

        opening, bbox = remove_green_and_open(frame)

        live_vis = resize(frame, DISPLAY_SCALE)

        best_match = None
        best_score = 0
        best_template = None

        if opening is not None and bbox is not None:
            x0, y0, _, _ = bbox
            
            for name, tmpl in templates.items():
                match_info, score = match_template(opening, tmpl)

                if score > best_score:
                    best_score = score
                    best_match = match_info
                    best_template = name

            # Draw BEST template bounding box
            if best_match is not None and best_score >= MATCH_THRESHOLD:
                (mx, my), (tw, th) = best_match

                # Map to full-frame coordinates
                fx1 = x0 + mx
                fy1 = y0 + my
                fx2 = fx1 + tw
                fy2 = fy1 + th

                # Map to live feed coordinates
                cv2.rectangle(
                    live_vis,
                    (fx1 // DISPLAY_SCALE, fy1 // DISPLAY_SCALE),
                    (fx2 // DISPLAY_SCALE, fy2 // DISPLAY_SCALE),
                    (0, 255, 0),
                    2
                )

            # Show TOP 3 scores
            scores = []
            for name, tmpl in templates.items():
                _, score = match_template(opening, tmpl)
                scores.append((name, score))

            scores.sort(key=lambda x: x[1], reverse=True)
            top3 = scores[:3]

            y_text = 30
            for i, (name, score) in enumerate(top3):
                cv2.putText(
                    live_vis,
                    f"{i+1}. {name}: {score:.2f}",
                    (20, y_text),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0) if i == 0 else (255, 255, 255),
                    2
                )
                y_text += 30

        cv2.imshow("Live Feed", live_vis)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    grabResult.Release()

# ------------------ CLEANUP ------------------
camera.StopGrabbing()
cv2.destroyAllWindows()
