import time
import cv2
import numpy as np
import pyautogui

TEMPLATE_PATH = 'button.png'
MATCH_THRESHOLD = 0.8

# Load the template in color, then convert to grayscale
template_color = cv2.imread(TEMPLATE_PATH, cv2.IMREAD_COLOR)
if template_color is None:
    raise FileNotFoundError(f'Cannot find template file: {TEMPLATE_PATH}')
template_gray = cv2.cvtColor(template_color, cv2.COLOR_BGR2GRAY)
h, w = template_gray.shape

print('Starting loop: detecting and attempting to click the button every second… (Press Ctrl+C to exit)')

try:
    while True:
        # Take screenshot, convert to BGR, then to grayscale
        screenshot = pyautogui.screenshot()
        img_bgr = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # Template matching (grayscale)
        res = cv2.matchTemplate(img_gray, template_gray, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)

        if max_val >= MATCH_THRESHOLD:
            cx = max_loc[0] + w // 2
            cy = max_loc[1] + h // 2
            print(f'Button detected (match score {max_val:.2f}), clicking at ({cx}, {cy})')
            pyautogui.click(cx, cy)
        else:
            print(f'Button not detected (highest match {max_val:.2f})')

        time.sleep(1)

except KeyboardInterrupt:
    print('Stopped.')
