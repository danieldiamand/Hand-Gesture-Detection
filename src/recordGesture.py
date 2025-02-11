import mediapipe as mp
import cv2
import time

from Gesture import Gesture, GestureRecording
from GestureFrame import GestureFrame

from constants import FRAME_TIME

COUNT_DOWN = 3

def recordGesture(name, filepath, oneHanded = False):
    """
    Tool for recordings gestures
    press a to start and stop recording
    press q to quit and save
    """
    handsModel = mp.solutions.hands
    drawing = mp.solutions.drawing_utils
    hands = handsModel.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_hands=2)

    videoCapture = cv2.VideoCapture(0)
    cv2.namedWindow('Gesture Capture', cv2.WINDOW_GUI_NORMAL)

    gesture = Gesture(name, oneHanded=oneHanded)
    recording = GestureRecording()

    isRecording = False
    startTime = time.time()

    while videoCapture.isOpened():
        #capture video and landmark using mediapipes
        frameStart = time.time()
        ret, frame = videoCapture.read()
        if not ret:
            break

        rgbFrame =  cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgbFrame)

        #visualize landmarking
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                drawing.draw_landmarks(frame, hand_landmarks, handsModel.HAND_CONNECTIONS)
        
        cv2.imshow('Gesture Capture', frame)

        #record frame
        if isRecording:
            frame = GestureFrame(results.multi_hand_landmarks, results.multi_handedness)
            recording.addFrame(frame)

        #check for keyboard inputs
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            gesture.saveAsJson(filepath)
            break
        if results.multi_hand_landmarks:
            if key == ord('a'):
                if isRecording:
                    gesture.addRecording(recording)
                    recording = GestureRecording()
                    print("STOPPED recording")
                    isRecording = False
                else:
                    print("Recording in: " + str(COUNT_DOWN))
                    for i in range(COUNT_DOWN, 0, -1):
                        time.sleep(1)
                        print(i)
                    print("STARTED recording!")
                    isRecording = True
        
        #maintain FPS
        elapsedFrame = time.time() - frameStart
        time.sleep(max(0, FRAME_TIME - elapsedFrame))
    
    videoCapture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    recordGesture("how_are_you", "how_are_youq.json", oneHanded=False)