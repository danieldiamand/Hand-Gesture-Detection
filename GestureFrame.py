import numpy as np

MAX_HAND_DIST = 5
BETWEEN_SIZE = 4
EACH_SIZE = 15

weights = {
    "betweenFingers" : 0.3,
    "eachFinger" : 0.3,
    "horizon" : 0.15,
    "handDist" : 0.15
}

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
            self.data = np.full(BETWEEN_SIZE + BETWEEN_SIZE + EACH_SIZE + EACH_SIZE + 1 + 1 + 1, np.nan)
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

        self.data = np.concatenate([self.leftBetweenFinger, self.rightBetweenFinger, self.leftEachFinger, self.rightEachFinger, [self.leftHorizon], [self.rightHorizon], [self.handDist]])

        if len(self.data) != BETWEEN_SIZE + BETWEEN_SIZE + EACH_SIZE + EACH_SIZE + 1 + 1 + 1:
            print(len(self.data))
            raise ValueError("Data incorrect shape!")
    
    def formatLandmarks(self, unformattedLandmarks):
        return np.array([[lm.x, lm.y, lm.z] for lm in list(unformattedLandmarks.landmark)], dtype=np.float32)

    def calcLMAngle(self, handLandmark, p1, p2, p3): #calculate angle between 3 lareturn np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)ndmark points
        vec1 = handLandmark[p1] - handLandmark[p2]
        vec2 = handLandmark[p3] - handLandmark[p2]

        return self.calcVecAngle(vec1, vec2)
    
    def calcVecAngle(self, v1, v2):
        vec1 = v1 / np.linalg.norm(v1)
        vec2 = v2 / np.linalg.norm(v2)

        dotProd = np.clip(np.dot(vec1, vec2), -1, 1)

        return np.arccos(dotProd)
    
    def calcBetweenFingers(self, handLandmark):
        if handLandmark is None:
            return np.full(BETWEEN_SIZE ,np.nan)
        return np.array([
            #angles between each finger:
            self.calcLMAngle(handLandmark, 1,0,5),
            self.calcLMAngle(handLandmark, 5,0,9),
            self.calcLMAngle(handLandmark, 19,0,13),
            self.calcLMAngle(handLandmark, 13, 0, 17)
        ])
    
    def calcEachFinger(self, handLandmark):
        if handLandmark is None:
            return np.full(EACH_SIZE ,np.nan)
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
            return np.nan
        vec1 = handLandmark[9] - handLandmark[0] #between wrist and centre of hand
        vec2 = [1, 0 ,0] #horizon
        return self.calcVecAngle(vec1, vec2)
    
    def calcHandDist(self, handLandmark1, handLandmark2):
        if handLandmark1 is None or handLandmark2 is None:
            return np.nan
        avHandSize = (np.linalg.norm(handLandmark1[9] - handLandmark1[0]) + np.linalg.norm(handLandmark2[9] - handLandmark2[0])) /2
        distBetween = np.linalg.norm(handLandmark1[0] - handLandmark2[0])
        return min(distBetween / avHandSize, MAX_HAND_DIST)
    
    def __str__(self):
        return ("No of hands:" + self.noOfHands + ", dist:" + self.getHandDist())
    

def angleDif(a ,b):
    diff = np.abs(a - b)
    return np.minimum(diff, 2 * np.pi - diff)

def frameDistance(frameData1, frameData2, oneHandedGesture = False):
    leftBetweenFinger1, rightBetweenFinger1, leftEachFinger1, rightEachFinger1, leftHorizon1, rightHorizon1, handDist1 = np.split(frameData1,
        [BETWEEN_SIZE, BETWEEN_SIZE+BETWEEN_SIZE, BETWEEN_SIZE+BETWEEN_SIZE+EACH_SIZE, BETWEEN_SIZE+BETWEEN_SIZE+EACH_SIZE+EACH_SIZE, BETWEEN_SIZE+BETWEEN_SIZE+EACH_SIZE+EACH_SIZE + 1, BETWEEN_SIZE+BETWEEN_SIZE+EACH_SIZE+EACH_SIZE +2])
    leftBetweenFinger2, rightBetweenFinger2, leftEachFinger2, rightEachFinger2, leftHorizon2, rightHorizon2, handDist2 = np.split(frameData2,
        [BETWEEN_SIZE, BETWEEN_SIZE+BETWEEN_SIZE, BETWEEN_SIZE+BETWEEN_SIZE+EACH_SIZE, BETWEEN_SIZE+BETWEEN_SIZE+EACH_SIZE+EACH_SIZE, BETWEEN_SIZE+BETWEEN_SIZE+EACH_SIZE+EACH_SIZE + 1, BETWEEN_SIZE+BETWEEN_SIZE+EACH_SIZE+EACH_SIZE +2])
    
    leftBetweenFingersDist = leftEachFingersDist = leftHorizonDist = rightBetweenFingersDist = rightEachFingersDist = rightHorizonDist = np.pi

    if (not np.isnan(leftBetweenFinger1[0]) and not np.isnan(leftBetweenFinger2[0])):
        leftBetweenFingersDist = np.mean(angleDif(leftBetweenFinger1, leftBetweenFinger2))
        leftEachFingersDist = np.mean(angleDif(leftEachFinger1, leftEachFinger2)) #assume we have each if we have between
        leftHorizonDist = np.abs(leftHorizon1[0] - leftHorizon2[0])
    
    if (not np.isnan(rightBetweenFinger1[0]) and not np.isnan(rightBetweenFinger2[0])):
        rightBetweenFingersDist = np.mean(angleDif(rightBetweenFinger1, rightBetweenFinger2))
        rightEachFingersDist = np.mean(angleDif(rightEachFinger1, rightEachFinger2)) 
        rightHorizonDist = angleDif(rightHorizon1[0], rightHorizon2[0])
    
    betweenDist = eachDist = horizonDist = handDist = np.pi 

    if oneHandedGesture:  # If 1-handed gesture, return hand that matches best
        betweenDist = min(leftBetweenFingersDist, rightBetweenFingersDist)
        eachDist = min(leftEachFingersDist, rightEachFingersDist)
        horizonDist = min(leftHorizonDist, rightHorizonDist)
    else:  # If 2-handed gesture, average both hands
        betweenDist = (leftBetweenFingersDist + rightBetweenFingersDist) / 2
        eachDist = (leftEachFingersDist + rightEachFingersDist) / 2
        horizonDist = (leftHorizonDist + rightHorizonDist) / 2

    if (not np.isnan(handDist1[0]) and not np.isnan(handDist2)):
        handDist = np.abs(handDist1[0] - handDist2[0]) / MAX_HAND_DIST * np.pi  # Ensures handDist is between 0 and pi.

    return (
        weights["betweenFingers"] * betweenDist +
        weights["eachFinger"] * eachDist +
        weights["horizon"] * horizonDist +
        weights["handDist"] * handDist
    )
