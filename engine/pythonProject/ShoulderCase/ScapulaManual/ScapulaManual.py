from ShoulderCase.GlenoidManual.GlenoidManual import GlenoidManual
from ShoulderCase.Scapula.Scapula import Scapula
from ShoulderCase.loadStl import loadStl
import os
import numpy as np
import pandas as pd

class ScapulaManual(Scapula):
    """
    To be used with ShoulderManual data.
    The load() method requires a specific implementation.
    Instanciate a GlenoidManual object.
    """
    def __init__(self, shoulder):
        super().__init__(shoulder)
        self.glenoid = GlenoidManual(self)

    def loadGroovePoints(self):
        SCase = self.shoulder.SCase

        filename = os.path.join(SCase.dataAmiraPath(),
                                "ScapulaGrooveLandmarks"+SCase.id4c+".landmarkAscii")
        self.groove= np.array(pd.read_table(filename,
                                       skiprows=[i for i in range(14)],
                                       sep=" ",
                                       header=None).iloc[:, :-1])
        self.groove = self.getSortedGrooveLateralToMedial()


    def loadLandmarks(self):
        """
        LOAD Load segmented surface and landmnarks
        Load the segmented scapula surface (if exist) and the
        scapula landmarks (6 scapula, 5 groove, 5 pillar) from
        amira directory.
        """
        SCase = self.shoulder.SCase
        filename = os.path.join(SCase.dataAmiraPath(),
                                "ScapulaLandmarks"+SCase.id4c+".landmarkAscii")
        landmarks = np.array(pd.read_table(filename,
                                       skiprows=[i for i in range(14)],
                                       sep=" ",
                                       header=None).iloc[:,:-1])
        self.angulusInferior = landmarks[0, :]
        self.trigonumSpinae = landmarks[1, :]
        self.processusCoracoideus = landmarks[2, :]
        self.acromioClavicular = landmarks[3, :]
        self.angulusAcromialis = landmarks[4, :]
        self.spinoGlenoidNotch = landmarks[5, :]

    def loadPillarPoints(self):
        SCase = self.shoulder.SCase
        filename = os.path.join(SCase.dataAmiraPath(),
                                "ScapulaPillarLandmarks"+SCase.id4c+".landmarkAscii")
        self.pillar= np.array(pd.read_table(filename,
                                       skiprows=[i for i in range(14)],
                                       sep=" ",
                                       header=None).iloc[:,:-1])

    def loadSurface(self):
        SCase = self.shoulder.SCase
        filename = os.path.join(SCase.dataAmiraPath(),
                                "scapula_"+SCase.id4c+".stl")

        points, faces, *_ = loadStl(filename, 1)
        self.surface["points"] = points
        self.surface["faces"] = faces
        self.segmentation = "M"
