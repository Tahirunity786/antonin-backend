import os
import numpy as np
from utils.Logger.Logger import Logger
from ShoulderCase.fitSphere import fitSphere
from ShoulderCase.Humerus.Humerus import Humerus
from ShoulderCase.Humerus.Humerus import landmarksBelongToCorrectShoulder
import pickle

class HumerusAuto(Humerus):
    """
    To be used with ShoulderAuto data.
    """
    def __init__(self, shoulder):
        super().__init__(shoulder)
        self.autoLandmarksPath = os.path.join(self.shoulder.SCase.dataPythonPath(),
                                              f"humerusLandmarksAuto{self.shoulder.side}.pkl")

    def loadData(self):
        """
        Call methods that can be run after the ShoulderCase object has
        been instantiated.
        """
        if self.hasAutoLandmarks():
            success = Logger.timeLogExecution("Humerus load landmarks (slicer): ",
                      lambda self : self.loadAutoLandmarks(), self)
        else:
            success = Logger.timeLogExecution("Humerus load landmarks: ",
                      lambda message : raise_(Exception(message)), "No landmarks file found"
            )
        return success

    def morphology(self):
        """
        Call methods that can be run after loadData() methods has been run
        by all ShoulderCase objects.
        """
        success = Logger.timeLogExecution("Humerus center and radius: ",
                      lambda self : self.measureCenterAndRadius(), self)
        success = success and Logger.timeLogExecution("Insertions' ring: ",
                      lambda self : self.measureInsertionsRing(), self)
        return success

    def hasAutoLandmarks(self):
        """
        Check if the landmarks file exists.
        """
        return os.path.exists(self.autoLandmarksPath)

    def loadAutoLandmarks(self):
        """
        LOAD Load 5 humeral head landmarks
        """
        landmarks = self.getAutoLandmarks()

        assert landmarksBelongToCorrectShoulder(landmarks, self.shoulder), "Loaded Auto landmarks belong to the other shoulder."

        self.landmarks["landmarks"] = landmarks

        return 1

    def getAutoLandmarks(self):

        assert os.path.isfile(self.autoLandmarksPath), "No Auto landmarks found."

        with open(self.autoLandmarksPath, 'rb') as f:
            loadedLandmarks = pickle.load(f)

        return np.array(list(loadedLandmarks.values())).squeeze()

    def measureCenterAndRadius(self):
        """
        By fitting a sphere on humeral head landmarks
        """
        landmarksToFit = self.getAutoLandmarks()
        center, self.radius, _, _ = fitSphere(landmarksToFit)
        self.center = center.T

    def measureInsertionsRing(self):
        """
        To be completed
        """
        pass



def raise_(ex):
    raise ex
