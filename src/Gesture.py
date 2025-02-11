from fastdtw import fastdtw
import numpy as np
import GestureFrame as gestureFrame
from multiprocessing import Pool
import multiprocessing as mp
import json

class GestureRecording:
    def __init__(self, maxLength=None):
        self.recording = np.array([]) #stores eachs frames data attribute, an array of np.arrays for computation
        self.rawFrames = [] #stores frames as array of gesture frames for storing/reading from json
        self.maxLength = maxLength

    def addFrame(self, gestureFrame):
        self.rawFrames.append(gestureFrame)
        if len(self.recording) == 0:
            self.recording = gestureFrame.data.reshape(1, -1)
        self.recording =  np.vstack((self.recording, gestureFrame.data))

        if self.maxLength is not None:
              if len(self.recording) > self.maxLength:
                 self.recording = self.recording[-self.maxLength:]
                 self.rawFrames.pop(0)
        
    def toDict(self):
        return {
            "rawFrames" : [frame.toDict() for frame in self.rawFrames]
        }
    
    @classmethod
    def fromDict(cls, data):
        recording = cls()
        recording.rawFrames = [gestureFrame.GestureFrame.fromDict(frameData) for frameData in data["rawFrames"]]
        if recording.rawFrames:
            recording.recording = np.vstack([frame.data for frame in recording.rawFrames])
        return recording
def compareRecordings(recording1, recording2, oneHandedGesture = False):
    def customDistance(frame1, frame2):
        return gestureFrame.frameDistance(frame1, frame2, oneHandedGesture=oneHandedGesture)
    distance, _ = fastdtw(recording1, recording2, dist=customDistance)
    return distance

class Gesture:
    def __init__(self, name, oneHanded = False):
        self.name = name
        self.oneHanded = oneHanded
        self.recordings = []
        
    def addRecording(self, newRecording):#a recording is gesture frame's data
        self.recordings.append(newRecording)
    
    def compareAll(self, comparedRecording):
        min = float("inf")
        for recording in self.recordings:
            dist = compareRecordings(recording.recording, comparedRecording.recording)
            if dist < min:
                min = dist
        return dist
    
    def toDict(self):
        return {
            "name": self.name,
            "oneHanded": self.oneHanded,
            "recordings": [recording.toDict() for recording in self.recordings]
        }
    
    @classmethod
    def fromDict(cls, data):
        gesture = cls(data["name"], data["oneHanded"])
        gesture.recordings = [GestureRecording.fromDict(recData) for recData in data["recordings"]]
        return gesture
    
    def saveAsJson(self, filepath):
        with open(filepath, "w") as f:
            json.dump(self.toDict(), f)
    
    @classmethod
    def loadFromJson(cls, filepath):
        with open(filepath, "r") as f:
            data = json.load(f)
            return cls.fromDict(data)