import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='google.protobuf.symbol_database')

import cv2
import mediapipe as mp
import pyautogui
from math import sqrt
pyautogui.FAILSAFE = False

# Initialize variables
x1 = y1 = x2 = y2 = x3 = y3 = 0

# Set up the webcam
webcam = cv2.VideoCapture(0)

# Initialize MediaPipe hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Get screen size for mouse control
screen_width, screen_height = pyautogui.size()

# Define the control area scaling factor (reduce the control area)
control_area_scaling = 1.3

# Define the mouse speed factor (increase the speed)
mouse_speed_factor = 1.4

while True:
    ret, image = webcam.read()
    if not ret:
        break

    # Flip the image horizontally for a later selfie-view display
    image = cv2.flip(image, 1)

    # Convert the BGR image to RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image and find hands
    result = hands.process(rgb_image)

    # Get the frame dimensions
    frame_height, frame_width, _ = image.shape

    # Draw hand landmarks and calculate distances
    if result.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(result.multi_hand_landmarks):
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            handedness = result.multi_handedness[idx].classification[0].label

            for id, landmark in enumerate(hand_landmarks.landmark):
                x = int(landmark.x * frame_width)
                y = int(landmark.y * frame_height)
                if id == 8:  # Index finger tip
                    cv2.circle(image, (x, y), 8, (0, 255, 255), 3)
                    x1 = x
                    y1 = y
                if id == 4:  # Thumb tip
                    cv2.circle(image, (x, y), 8, (255, 255, 0), 3)
                    x2 = x
                    y2 = y
                if id == 12:  # Middle finger tip
                    cv2.circle(image, (x, y), 8, (0, 255, 0), 3)
                    x3 = x
                    y3 = y

            # Calculate the distance between index finger tip and thumb tip
            dist_thumb_index = sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) // 4

            # Calculate the distance between thumb tip and middle finger tip
            dist_thumb_middle = sqrt((x2 - x3) ** 2 + (y2 - y3) ** 2)

            if handedness == "Right":
                # Right hand controls mouse movement
                if dist_thumb_middle > 70:
                    # Reduce the control area by scaling the hand coordinates
                    screen_x = screen_width * (hand_landmarks.landmark[8].x - 0.5) * control_area_scaling + screen_width / 2
                    screen_y = screen_height * (hand_landmarks.landmark[8].y - 0.5) * control_area_scaling + screen_height / 2
                    # Increase the mouse movement speed
                    current_mouse_x, current_mouse_y = pyautogui.position()
                    pyautogui.moveTo(current_mouse_x + (screen_x - current_mouse_x) * mouse_speed_factor, 
                                     current_mouse_y + (screen_y - current_mouse_y) * mouse_speed_factor)
                else:
                    # Adjust the system volume based on the distance between thumb and index finger
                    if dist_thumb_index > 15:
                        pyautogui.press("volumeup")
                    else:
                        pyautogui.press("volumedown")
                    cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 5)

            elif handedness == "Left":
                # Left hand for click action
                if dist_thumb_middle < 70:
                    pyautogui.click()
                    cv2.line(image, (x1, y1), (x3, y3), (0, 0, 255), 5)

    # Display the image
    cv2.imshow("Hand volume control and Mouse Tracker", image)

    # Exit on pressing 'ESC'
    if cv2.waitKey(10) & 0xFF == 27:
        break

# Release the webcam and destroy all windows
webcam.release()
cv2.destroyAllWindows()
