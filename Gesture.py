import numpy as np

class Gesture:
    def __init__(self, multiLandmarks):
        self.hand1BetweenFingerAngles = []
        self.hand2BetweenFingerAngles = []

        self.hand1EachFingerAngles = []
        self.hand2EachFingerAngles = []

        self.hand1HorizonAngle = -1
        self.hand2HorizonAngle = -1

        self.betweenHandDist = -1 #relative to size of hands


        if (len(multiLandmarks) == 0):
            return
        if (len(multiLandmarks) == 1): #if only one hand, store it twice so at least some matching can be done.
            handLandmark = np.array(multiLandmarks[0])
            self.hand1BetweenFingerAngles = self.hand2BetweenFingerAngles = self.self.calculateAnglesBetweenFingers(handLandmark)
            self.hand1EachFingerAngles = self.hand2EachFingerAngles = self.self.calculateAnglesEachFinger(handLandmark)
            self.hand1HorizonAngle = self.hand2HorizonAngle = self.calculateHorizonAngle(handLandmark)
            return
        if (len(multiLandmarks) > 2):
            raise ValueError("More than 2 hands in gesture!")
        
        #ideal case: there are 2 landmarked hands.
        hand1Landmark = np.array(multiLandmarks[0])
        hand2Landmark = np.array(multiLandmarks[1])

        self.hand1BetweenFingerAngles = self.calculateAnglesBetweenFingers(hand1Landmark)
        self.hand2BetweenFingerAngles = self.calculateAnglesBetweenFingers(hand2Landmark)

        self.hand1EachFingerAngles = self.calculateAnglesEachFinger(hand1Landmark)
        self.hand2EachFingerAngles = self.calculateAnglesEachFinger(hand2Landmark)

        self.hand1HorizonAngle = self.calculateHorizonAngle(hand1Landmark)
        self.hand2HorizonAngle = self.calculateHorizonAngle(hand2Landmark)

        self.betweenHandDist = self.calculateBetweenHandDistance(hand1Landmark, hand2Landmark)



    def calculateAnglesBetweenFingers(self, handLandmark):
        return np.array([
            #angles between each finger:
            self.calculateAngle(handLandmark, 1,0,5),
            self.calculateAngle(handLandmark, 5,0,9),
            self.calculateAngle(handLandmark, 19,0,13),
            self.calculateAngle(handLandmark, 13, 0, 17)
        ])

    def calculateAnglesEachFinger(self, handLandmark): 
        return np.array([
            #angles between thumb joints
            self.calculateAngle(handLandmark, 0, 1, 2), 
            self.calculateAngle(handLandmark, 1, 2, 3), 
            self.calculateAngle(handLandmark, 2, 3, 4), 

            #angles between index finger joints
            self.calculateAngle(handLandmark, 0, 5, 6), 
            self.calculateAngle(handLandmark, 5, 6, 7),  
            self.calculateAngle(handLandmark, 6, 7, 8), 

            #angles between middle finger joints
            self.calculateAngle(handLandmark, 0, 9, 10),
            self.calculateAngle(handLandmark, 9, 10, 11), 
            self.calculateAngle(handLandmark, 10, 11, 12), 

            #angles between ring finger joints
            self.calculateAngle(handLandmark, 0, 13, 14),  
            self.calculateAngle(handLandmark, 13, 14, 15),  
            self.calculateAngle(handLandmark, 14, 15, 16),  

            #angles between pinky finger joints
            self.calculateAngle(handLandmark, 0, 17, 18),  
            self.calculateAngle(handLandmark, 17, 18, 19),  
            self.calculateAngle(handLandmark, 18, 19, 20),  
        ])
    
    def calculateAngleVec(self, v1, v2):
        vec1 = vec1 / np.linalg.norm(v1)
        vec2 = vec2 / np.linalg.norm(v2)

        dotProd = np.clip(np.dot(vec1, vec2), -1, 1)

        return np.degrees(np.arccos(dotProd))

    def calculateAngle(self, handLandmark, p1, p2, p3):

        vec1 = handLandmark[p1] - handLandmark[p2]
        vec2 = handLandmark[p3] - handLandmark[p2]

        return self.calculateAngleVec(vec1, vec2)
    
    def calculateHorizonAngle(self, handLandmark):
        vec1 = handLandmark[9] - handLandmark[0] #between wrist and centre of hand
        vec2 = [1, 0 ,0] #horizon

        return self.calculateAngleVec(vec1, vec2)
    
    def calculateBetweenHandDistance(self, handLandmark1, handLandmark2):
        avHandSize = (np.linalg.norm(handLandmark1[12] - handLandmark1[0]) + np.linalg.norm(handLandmark2[12] - handLandmark2[0])) /2
        distBetween = np.linalg.norm(handLandmark1[0] - handLandmark2[0])
        return distBetween / avHandSize
        