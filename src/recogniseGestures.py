import mediapipe as mp
import cv2
import time

from Gesture import Gesture, GestureRecording
from GestureFrame import GestureFrame

from constants import FPS, FRAME_TIME

SECOUNDS_ANALYZED = 2 #how many secounds of capture is analyzed and compared with gestures

class MultiGestures:
    def __init__(self):
        self.gestures = []
    
    def loadGesture(self, filepath):
        gesture = Gesture.loadFromJson(filepath)
        self.gestures.append(gesture)
    
    def bestMatch(self, comparedRecording):
        bestGesture = None
        bestScore = float("inf")
        for gesture in self.gestures:
            score = gesture.compareAll(comparedRecording)
            if score < bestScore:
                bestGesture = gesture
                bestScore = score
        
        return bestGesture.name, bestScore

def recogniseGestures(multiGesture):
    handsModel = mp.solutions.hands
    drawing = mp.solutions.drawing_utils
    hands = handsModel.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_hands=2)

    videoCapture = cv2.VideoCapture(0)
    cv2.namedWindow('Gesture Capture', cv2.WINDOW_GUI_NORMAL)

    recording = GestureRecording(maxLength=FPS*SECOUNDS_ANALYZED)

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

        #find best match
        frame = GestureFrame(results.multi_hand_landmarks, results.multi_handedness)
        recording.addFrame(frame)

        bestName, bestScore = multiGesture.bestMatch(recording)
        
        print(bestName, bestScore)

        #check for keyboard inputs
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        
        #maintain FPS
        elapsedFrame = time.time() - frameStart
        time.sleep(max(0, FRAME_TIME - elapsedFrame))
    

if __name__ == "__main__":
    multiGesture = MultiGestures()
    multiGesture.loadGesture("hello.json")
    multiGesture.loadGesture("how_are_you.json")
    recogniseGestures(multiGesture)
