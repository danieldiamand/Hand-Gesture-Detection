import cv2
import mediapipe as mp
import time
import GestureFrame as gestureFrame
import Gesture as gesture

#init mediapipes
hands_model = mp.solutions.hands
drawing = mp.solutions.drawing_utils
hands = hands_model.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)


video_capture = cv2.VideoCapture(0)
cv2.namedWindow('Hand Land Marking', cv2.WINDOW_GUI_NORMAL)

startTime = time.time()


isRecording = False

frame = gestureFrame.GestureFrame(None, None)
recording = gesture.GestureRecording()
aGesture = gesture.Gesture("wave", True)

while video_capture.isOpened():
    frameStart = time.time()
    ret, frame = video_capture.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            drawing.draw_landmarks(frame, hand_landmarks, hands_model.HAND_CONNECTIONS)
    
    cv2.imshow('Hand Land Marking', frame)



    if (isRecording):
        frame = gestureFrame.GestureFrame(results.multi_hand_landmarks, results.multi_handedness)
        recording.addFrame(frame)
   
    

    if key == ord('q'):
        aGesture.saveAsJson("wave.json")
        break
    if results.multi_hand_landmarks:
        if key == ord('a'):
            if isRecording == True:
                isRecording = False
                aGesture.addRecording(recording)
                recording = gesture.GestureRecording()
            else:
                isRecording = True


    elapsedFrame = time.time() - frameStart
    time.sleep(max(0, FRAME_TIME - elapsedFrame))
video_capture.release()
cv2.destroyAllWindows()