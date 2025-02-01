import numpy as np

MAX_HAND_DIST = 5

class GestureFrame:
    def __init__(self, multiLandmarks):
        self.noOfHands = len(multiLandmarks)

        #Angles between each finger:thumb and index, index and middle, ...
        self.hand1BetweenFinger = []
        self.hand2BetweenFinger = []

        #Angles between each knuckle of each finger, each thumb knuckle, each index knuckle, ...
        self.hand1EachFinger = []
        self.hand2EachFinger = []

        #Angles between hand and horizon [1,0,0]
        self.hand1Horizon = -1
        self.hand2Horizon = -1
        
        #Distance between hands relative to size of hands, max's at 5.
        self.handDist = MAX_HAND_DIST
    
        if (self.noOfHands == 0):
            return
        if (self.noOfHands == 1):
            self.hand1BetweenFinger = self.hand2BetweenFinger = self.calcBetweenFingers(multiLandmarks[0])
            self.hand1EachFinger = self.hand2EachFinger = self.calcEachFinger(multiLandmarks[0])
            self.hand1Horizon = self.hand2Horizon = self.calcHorizon(multiLandmarks[0])
            return
        if (self.noOfHands != 2):
            raise ValueError("More than 2 hands in gesture!") 
        
        #o/w must be 2 landmarked hands
        self.hand1BetweenFinger = self.calcBetweenFingers(multiLandmarks[0])
        self.hand2BetweenFinger = self.calcBetweenFingers(multiLandmarks[1])
        self.hand1EachFinger = self.calcEachFinger(multiLandmarks[0])
        self.hand2EachFinger = self.calcEachFinger(multiLandmarks[1])
        self.hand1Horizon = self.calcHorizon(multiLandmarks[0])
        self.hand2Horizon = self.calcHorizon(multiLandmarks[1])
        self.handDist = self.calcHandDist(multiLandmarks[0], multiLandmarks[1])
    
    def getHand1Between(self, swap = False):
        return self.hand2BetweenFinger if swap else self.hand1BetweenFinger
    
    def getHand2Between(self, swap = False):
        return self.hand1BetweenFinger if swap else self.hand2BetweenFinger
    
    def getHand1Each(self, swap = False):
        return self.hand2EachFinger if swap else self.hand1EachFinger
    
    def getHand2Each(self, swap = False):
        return self.hand1EachFinger if swap else self.hand2EachFinger
    
    def getHand1Horizon(self, swap = False):
        return self.hand2Horizon if swap else self.hand1Horizon
    
    def getHand2Horizon(self, swap = False):
        return self.hand1Horizon if swap else self.hand2Horizon
    
    def getHandDist(self):
        return self.handDist


    def calcLMAngle(self, handLandmark, p1, p2, p3): #calculate angle between 3 landmark points
        vec1 = handLandmark[p1] - handLandmark[p2]
        vec2 = handLandmark[p3] - handLandmark[p2]

        return self.calcVecAngle(vec1, vec2)
    
    def calcVecAngle(self, v1, v2):
        vec1 = v1 / np.linalg.norm(v1)
        vec2 = v1 / np.linalg.norm(v2)

        dotProd = np.clip(np.dot(vec1, vec2), -1, 1)

        return np.arccos(dotProd)
    
    def calcBetweenFingers(self, handLandmark):
        return np.array([
            #angles between each finger:
            self.calcLMAngle(handLandmark, 1,0,5),
            self.calcLMAngle(handLandmark, 5,0,9),
            self.calcLMAngle(handLandmark, 19,0,13),
            self.calcLMAngle(handLandmark, 13, 0, 17)
        ])
    
    def calcEachFinger(self, handLandmark): 
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
        vec1 = handLandmark[9] - handLandmark[0] #between wrist and centre of hand
        vec2 = [1, 0 ,0] #horizon
        return self.calcVecAngle(vec1, vec2)
    
    def calcHandDist(self, handLandmark1, handLandmark2):
        avHandSize = (np.linalg.norm(handLandmark1[9] - handLandmark1[0]) + np.linalg.norm(handLandmark2[9] - handLandmark2[0])) /2
        distBetween = np.linalg.norm(handLandmark1[0] - handLandmark2[0])
        return min(distBetween / avHandSize, 5)
    

def angleDif(a ,b):
    diff = np.abs(a - b)
    return np.minimum(diff, 2 * np.pi - diff)

def dtwGestureFrameDistance(frame1, frame2):

    weights = {
        "betweenFingers" : 0.3,
        "eachFinger" : 0.3,
        "horizon" : 0.15,
        "handDist" : 0.15
    }

    def frameDistance(frame1, frame2, swap):
        if (frame1.noOfHands == 0 or frame2.noOfHands == 0):
            return 3.14 #max val

        hand1BetweenFingersDist = np.linalg.norm(angleDif(frame1.getHand1Between(),frame2.getHand1Between(swap=swap))) / len(frame1.getHand1Between())
        hand2BetweenFingersDist = np.linalg.norm(angleDif(frame1.getHand2Between(), frame2.getHand2Between(swap=swap))) / len(frame1.getHand2Between())
        betweenDist = (hand1BetweenFingersDist + hand2BetweenFingersDist) / 2

        hand1EachFingersDist = np.linalg.norm(angleDif(frame1.getHand1Each(), frame2.getHand1Each(swap=swap))) / len(frame1.getHand1Each())
        hand2EachFingersDist = np.linalg.norm(angleDif(frame1.getHand2Each(), frame2.getHand2Each(swap=swap))) / len(frame1.getHand2Each())
        eachDist = (hand1EachFingersDist + hand2EachFingersDist) / 2


        hand1HorizonDist = np.linalg.norm(angleDif(frame1.getHand1Horizon(), frame2.getHand1Horizon(swap=swap)))
        hand2HorizonDist = np.linalg.norm(angleDif(frame1.getHand2Horizon(), frame2.getHand2Horizon(swap=swap)))
        horizonDist = (hand1HorizonDist + hand2HorizonDist) / 2

        handDist = np.abs(frame1.getHandDist() - frame2.getHandDist()) / MAX_HAND_DIST * 3.14 #Ensures handDist is between 0 and pi.

        return (weights["betweenFingers"] * betweenDist + weights["eachFinger"] * eachDist + weights["horizon"] * horizonDist + weights["handDist"] * handDist)
    
    return min(frameDistance(frame1, frame2, False), frameDistance(frame1, frame2, True))
