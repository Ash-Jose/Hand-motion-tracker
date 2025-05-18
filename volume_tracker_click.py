import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='google.protobuf.symbol_database')

import cv2
import mediapipe as mp
import pyautogui
from math import sqrt
import time

pyautogui.FAILSAFE = False

# Initialize variables
x1 = y1 = x2 = y2 = x3 = y3 = 0
last_click_time = 0
click_cooldown = 1  # seconds

# Set up the webcam
webcam = cv2.VideoCapture(0)
if not webcam.isOpened():
    print("Error: Webcam not accessible.")
    exit()

# Initialize MediaPipe hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                       min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Get screen size
screen_width, screen_height = pyautogui.size()

# Constants
control_area_scaling = 1.3
mouse_speed_factor = 1.4

def calculate_distance(x1, y1, x2, y2):
    return sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

while True:
    ret, image = webcam.read()
    if not ret:
        break

    image = cv2.flip(image, 1)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_image)
    frame_height, frame_width, _ = image.shape

    if result.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(result.multi_hand_landmarks):
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            handedness = result.multi_handedness[idx].classification[0].label

            coords = {}
            for id, lm in enumerate(hand_landmarks.landmark):
                x = int(lm.x * frame_width)
                y = int(lm.y * frame_height)
                coords[id] = (x, y)

            if 4 in coords and 8 in coords and 12 in coords:
                x1, y1 = coords[8]   # index
                x2, y2 = coords[4]   # thumb
                x3, y3 = coords[12]  # middle

                dist_thumb_index = calculate_distance(x1, y1, x2, y2)
                dist_thumb_middle = calculate_distance(x2, y2, x3, y3)

                if handedness == "Right":
                    if dist_thumb_middle > 70:
                        # Mouse control
                        screen_x = screen_width * (hand_landmarks.landmark[8].x - 0.5) * control_area_scaling + screen_width / 2
                        screen_y = screen_height * (hand_landmarks.landmark[8].y - 0.5) * control_area_scaling + screen_height / 2
                        curr_x, curr_y = pyautogui.position()
                        pyautogui.moveTo(curr_x + (screen_x - curr_x) * mouse_speed_factor,
                                         curr_y + (screen_y - curr_y) * mouse_speed_factor)
                    else:
                        # Volume control
                        if dist_thumb_index > 40:
                            pyautogui.press("volumeup")
                        elif dist_thumb_index < 20:
                            pyautogui.press("volumedown")
                        cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 3)

                elif handedness == "Left":
                    if dist_thumb_index < 30:
                        current_time = time.time()
                        if current_time - last_click_time > click_cooldown:
                            pyautogui.click()
                            last_click_time = current_time
                        cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 3)

    # Resize output window for larger view
    scaled_image = cv2.resize(image, None, fx=1.5, fy=1.5)
    cv2.imshow("Hand Gesture Control", scaled_image)

    if cv2.waitKey(10) & 0xFF == 27:
        break

webcam.release()
cv2.destroyAllWindows()
