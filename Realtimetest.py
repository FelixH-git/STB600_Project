from pypylon import pylon
import cv2
import numpy as np

# ------------------ CONFIG ------------------
TEMPLATE_PATH = r"C:\Users\biggi\Desktop\STB600_Project\Coins\9_MASK.png"
MATCH_THRESHOLD = 0.65

# ------------------ LOAD TEMPLATE ------------------
template = cv2.imread(TEMPLATE_PATH, cv2.IMREAD_GRAYSCALE)
_, template = cv2.threshold(template, 10, 255, cv2.THRESH_BINARY)
th, tw = template.shape

# ------------------ UTILS ------------------
def resize(img, scale):
    return cv2.resize(
        img,
        (int(img.shape[1] / scale), int(img.shape[0] / scale))
    )

# ------------------ IMAGE PROCESSING ------------------
def remove_green_and_open(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_green = np.array([35, 80, 40])
    upper_green = np.array([85, 255, 255])

    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    mask_inv = cv2.bitwise_not(green_mask)

    no_green = cv2.bitwise_and(frame, frame, mask=mask_inv)

    gray = cv2.cvtColor(no_green, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)

    coin = no_green[y:y+h, x:x+w]
    coin_gray = cv2.cvtColor(coin, cv2.COLOR_BGR2GRAY)
    _, coin_thresh = cv2.threshold(coin_gray, 10, 255, cv2.THRESH_BINARY)

    kernel = np.ones((25, 25), np.uint8)
    opening = cv2.morphologyEx(coin_thresh, cv2.MORPH_OPEN, kernel)

    return opening

# ------------------ TEMPLATE MATCHING ------------------
def match_template(binary_coin):
    if binary_coin.shape[0] < th or binary_coin.shape[1] < tw:
        return None

    result = cv2.matchTemplate(
        binary_coin, template, cv2.TM_CCOEFF_NORMED
    )

    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    if max_val >= MATCH_THRESHOLD:
        return max_loc, max_val

    return None

# ------------------ BASLER CAMERA ------------------
camera = pylon.InstantCamera(
    pylon.TlFactory.GetInstance().CreateFirstDevice()
)

camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

converter = pylon.ImageFormatConverter()
converter.OutputPixelFormat = pylon.PixelType_BGR8packed
converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

print("[INFO] Press ESC to exit")

# ------------------ MAIN LOOP ------------------
while camera.IsGrabbing():
    grabResult = camera.RetrieveResult(
        5000, pylon.TimeoutHandling_ThrowException
    )

    if grabResult.GrabSucceeded():
        image = converter.Convert(grabResult)
        frame = image.GetArray()

        opening = remove_green_and_open(frame)

        cv2.imshow("Live Feed", resize(frame, 3))

        if opening is not None:
            vis = cv2.cvtColor(opening, cv2.COLOR_GRAY2BGR)

            match = match_template(opening)
            if match:
                (x, y), score = match
                cv2.rectangle(
                    vis,
                    (x, y),
                    (x + tw, y + th),
                    (0, 255, 0),
                    2
                )
                cv2.putText(
                    vis,
                    f"Match: {score:.2f}",
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2
                )

            cv2.imshow("Opening + Template", resize(vis, 2))

        if cv2.waitKey(1) == 27:
            break

    grabResult.Release()

# ------------------ CLEANUP ------------------
camera.StopGrabbing()
cv2.destroyAllWindows()
