import cv2
import mediapipe as mp

#init mediapipes
hands_model = mp.solutions.hands
drawing = mp.solutions.drawing_utils
hands = hands_model.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)


video_capture = cv2.VideoCapture(0)
cv2.namedWindow('Hand Land Marking', cv2.WINDOW_GUI_NORMAL)

while video_capture.isOpened():
    ret, frame = video_capture.read()
    if not ret:
        break
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            drawing.draw_landmarks(frame, hand_landmarks, hands_model.HAND_CONNECTIONS)
    
    cv2.imshow('Hand Land Marking', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()