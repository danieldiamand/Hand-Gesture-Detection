import numpy as np

MAX_HAND_DIST = 5

class GestureFrame:
    def __init__(self, multiLandmarks, multiHandedness):
 
        #Angles between each finger:thumb and index, index and middle, ...
        self.leftBetweenFinger = None
        self.rightBetweenFinger = None

        #Angles between each knuckle of each finger, each thumb knuckle, each index knuckle, ...
        self.leftEachFinger = None
        self.rightEachFinger = None

        #Angles between hand and horizon [1,0,0]
        self.leftHorizon = None
        self.rightHorizon = None
        
        #Distance between hands relative to size of hands, max's at 5.
        self.handDist = MAX_HAND_DIST
    
        if multiLandmarks is None:
            self.noOfHands = 0
            return
        
        self.noOfHands = len(multiLandmarks)

        self.leftLandmark = None
        self.rightLandmark = None

        if (self.noOfHands == 1):
            if (multiHandedness[0].classification[0].label == "left"):
                self.leftLandmark = self.formatLandmarks(multiLandmarks[0])
            else:
                self.rightLandmark = self.formatLandmarks(multiLandmarks[0])
        
        if (self.noOfHands == 2):
            if (multiHandedness[0].classification[0].label == "left"):
                self.leftLandmark = self.formatLandmarks(multiLandmarks[0])
            else:
                self.rightLandmark = self.formatLandmarks(multiLandmarks[0])
            if (multiHandedness[1].classification[0].label == "left"):
                self.leftLandmark = self.formatLandmarks(multiLandmarks[1])
            else:
                self.rightLandmark = self.formatLandmarks(multiLandmarks[1])

        if (self.noOfHands > 2):
            raise ValueError("More than 2 hands in gesture!") 
                

        #o/w must be 2 landmarked hands
        self.leftBetweenFinger = self.calcBetweenFingers(self.leftLandmark)
        self.rightBetweenFinger = self.calcBetweenFingers(self.rightLandmark)
        self.leftEachFinger = self.calcEachFinger(self.leftLandmark)
        self.rightEachFinger = self.calcEachFinger(self.rightLandmark)
        self.leftHorizon = self.calcHorizon(self.leftLandmark)
        self.rightHorizon = self.calcHorizon(self.rightLandmark)
        self.handDist = self.calcHandDist(self.leftLandmark, self.rightLandmark)

    def formatLandmarks(self, unformattedLandmarks):
        return np.array([[lm.x, lm.y, lm.z] for lm in list(unformattedLandmarks.landmark)], dtype=np.float32)

    def calcLMAngle(self, handLandmark, p1, p2, p3): #calculate angle between 3 lareturn np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)ndmark points
        vec1 = handLandmark[p1] - handLandmark[p2]
        vec2 = handLandmark[p3] - handLandmark[p2]

        return self.calcVecAngle(vec1, vec2)
    
    def calcVecAngle(self, v1, v2):
        vec1 = v1 / np.linalg.norm(v1)
        vec2 = v1 / np.linalg.norm(v2)

        dotProd = np.clip(np.dot(vec1, vec2), -1, 1)

        return np.arccos(dotProd)
    
    def calcBetweenFingers(self, handLandmark):
        if handLandmark is None:
            return None
        return np.array([
            #angles between each finger:
            self.calcLMAngle(handLandmark, 1,0,5),
            self.calcLMAngle(handLandmark, 5,0,9),
            self.calcLMAngle(handLandmark, 19,0,13),
            self.calcLMAngle(handLandmark, 13, 0, 17)
        ])
    
    def calcEachFinger(self, handLandmark):
        if handLandmark is None:
            return None
        return np.array([
            #angles between thumb joints
            self.calcLMAngle(handLandmark, 0, 1, 2), 
            self.calcLMAngle(handLandmark, 1, 2, 3), 
            self.calcLMAngle(handLandmark, 2, 3, 4), 

            #angles between index finger joints
            self.calcLMAngle(handLandmark, 0, 5, 6), 
            self.calcLMAngle(handLandmark, 5, 6, 7),  
            self.calcLMAngle(handLandmark, 6, 7, 8), 

            #angles between middle finger joints
            self.calcLMAngle(handLandmark, 0, 9, 10),
            self.calcLMAngle(handLandmark, 9, 10, 11), 
            self.calcLMAngle(handLandmark, 10, 11, 12), 

            #angles between ring finger joints
            self.calcLMAngle(handLandmark, 0, 13, 14),  
            self.calcLMAngle(handLandmark, 13, 14, 15),  
            self.calcLMAngle(handLandmark, 14, 15, 16),  

            #angles between pinky finger joints
            self.calcLMAngle(handLandmark, 0, 17, 18),  
            self.calcLMAngle(handLandmark, 17, 18, 19),  
            self.calcLMAngle(handLandmark, 18, 19, 20),  
        ])
    
    def calcHorizon(self, handLandmark):
        if handLandmark is None:
            return None
        vec1 = handLandmark[9] - handLandmark[0] #between wrist and centre of hand
        vec2 = [1, 0 ,0] #horizon
        return self.calcVecAngle(vec1, vec2)
    
    def calcHandDist(self, handLandmark1, handLandmark2):
        if handLandmark1 is None or handLandmark2 is None:
            return MAX_HAND_DIST
        avHandSize = (np.linalg.norm(handLandmark1[9] - handLandmark1[0]) + np.linalg.norm(handLandmark2[9] - handLandmark2[0])) /2
        distBetween = np.linalg.norm(handLandmark1[0] - handLandmark2[0])
        return min(distBetween / avHandSize, MAX_HAND_DIST)
    
    def __str__(self):
        return ("No of hands:" + self.noOfHands + ", dist:" + self.getHandDist())
    

def angleDif(a ,b):
    diff = np.abs(a - b)
    return np.minimum(diff, 2 * np.pi - diff)

def dtwGestureFrameDistance(frame1, frame2, oneHandedGesture = False):

    weights = {
        "betweenFingers" : 0.3,
        "eachFinger" : 0.3,
        "horizon" : 0.15,
        "handDist" : 0.15
    }

    if (frame1.noOfHands == 0 or frame2.noOfHands == 0):
        return np.pi #max val


    leftBetweenFingersDist = leftEachFingersDist = leftHorizonDist = rightBetweenFingersDist = rightEachFingersDist = rightHorizonDist = np.pi

    if (frame1.leftLandmark is not None and frame2.leftLandmark is not None):
        leftBetweenFingersDist = np.linalg.norm(angleDif(frame1.leftBetweenFinger, frame2.leftBetweenFinger)) / len(frame1.leftBetweenFinger)
        leftEachFingersDist = np.linalg.norm(angleDif(frame1.leftEachFinger, frame2.leftEachFinger)) / len(frame1.leftEachFinger)
        leftHorizonDist = np.linalg.norm(angleDif(frame1.leftHorizon, frame2.leftHorizon))
    
    if (frame1.rightLandmark is not None and frame2.rightLandmark is not None):
        rightBetweenFingersDist = np.linalg.norm(angleDif(frame1.rightBetweenFinger, frame2.rightBetweenFinger)) / len(frame1.rightBetweenFinger)
        rightEachFingersDist = np.linalg.norm(angleDif(frame1.rightEachFinger, frame2.rightEachFinger)) / len(frame1.rightEachFinger)
        rightHorizonDist = np.linalg.norm(angleDif(frame1.rightHorizon, frame2.rightHorizon))

    if oneHandedGesture:  # If 1-handed gesture, return hand that matches best
        betweenDist = min(leftBetweenFingersDist, rightBetweenFingersDist)
        eachDist = min(leftEachFingersDist, rightEachFingersDist)
        horizonDist = min(leftHorizonDist, rightHorizonDist)
    else:  # If 2-handed gesture, average both hands
        betweenDist = (leftBetweenFingersDist + rightBetweenFingersDist) / 2
        eachDist = (leftEachFingersDist + rightEachFingersDist) / 2
        horizonDist = (leftHorizonDist + rightHorizonDist) / 2

    handDist = np.pi
    if (frame1.handDist is not None and frame2.handDist is not None):
        handDist = np.abs(frame1.handDist - frame2.handDist) / MAX_HAND_DIST * np.pi  # Ensures handDist is between 0 and pi.

    print("between: " + betweenDist + ", eachDist: " + eachDist + ", horizon: " + horizonDist + "handDist:" + handDist)

    return (
        weights["betweenFingers"] * betweenDist +
        weights["eachFinger"] * eachDist +
        weights["horizon"] * horizonDist +
        weights["handDist"] * handDist
    )

