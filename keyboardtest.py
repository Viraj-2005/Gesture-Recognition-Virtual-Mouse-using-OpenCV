import cv2
import mediapipe as mp
import pyautogui

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

mp_drawing = mp.solutions.drawing_utils

screen_width, screen_height = pyautogui.size()

cap = cv2.VideoCapture(0)

prev_x = None
prev_y = None

while cap.isOpened():
    ret, frame = cap.read()

    frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

            index_tip = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            thumb_tip = landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            pinky_mcp = landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]

            x, y = int(index_tip.x * screen_width), int(index_tip.y * screen_height)

            if prev_x is not None and prev_y is not None:
                dx = x - prev_x
                dy = y - prev_y

                if abs(dx) > abs(dy): 
                    if dx > 50:  
                        pyautogui.press('right')
                    elif dx < -50:  
                        pyautogui.press('left')
                else:  
                    if dy > 50:  
                        pyautogui.press('down')
                    elif dy < -50:  
                        pyautogui.press('up')

            prev_x = x
            prev_y = y

            distance_thumb_pinky = ((thumb_tip.x - pinky_mcp.x) ** 2 + (thumb_tip.y - pinky_mcp.y) ** 2) ** 0.5

            if distance_thumb_pinky < 0.1:  
                pyautogui.press('space')

    cv2.imshow("Gesture Recognition", frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
