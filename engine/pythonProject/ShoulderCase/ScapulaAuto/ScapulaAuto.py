from ShoulderCase.Scapula.Scapula import Scapula
from ShoulderCase.GlenoidAuto.GlenoidAuto import GlenoidAuto
from ShoulderCase.LandmarksExporter.LandmarksExporter import LandmarksExporter
from ShoulderCase.loadPly import loadPly
import os
import pickle

class ScapulaAuto(Scapula):
    """
    To be used with ShoulderAuto data.
    The load() method requires a specific implementation.
    Instantiate a GlenoidAuto object.
    """
    def __init__(self, shoulder):
        super().__init__(shoulder)
        self.glenoid = GlenoidAuto(self)

    def exportLandmarksToAmiraFolder(self):
        scapulaLandmarks = LandmarksExporter()
        scapulaLandmarks.addLandmarks(self.angulusInferior)
        scapulaLandmarks.addLandmarks(self.trigonumSpinae)
        scapulaLandmarks.addLandmarks(self.processusCoracoideus)
        scapulaLandmarks.addLandmarks(self.acromioClavicular)
        scapulaLandmarks.addLandmarks(self.angulusAcromialis)
        scapulaLandmarks.addLandmarks(self.spinoGlenoidNotch)

        if not os.path.isdir(self.shoulder.SCase.dataAmiraPath()):
            os.mkdir(self.shoulder.SCase.dataAmiraPath())

        filename = "AutoScapulaLandmarks" + self.shoulder.SCase.id4c + ".landmarkAscii"
        scapulaLandmarks.exportAmiraFile(self.shoulder.SCase.dataAmiraPath(), filename)

    def exportPillarPointsToAmiraFolder(self):
        scapulaLandmarks = LandmarksExporter()
        scapulaLandmarks.addLandmarks(self.pillar)

        if not os.path.isdir(self.shoulder.SCase.dataAmiraPath()):
            os.mkdir(self.shoulder.SCase.dataAmiraPath())

        filename = "AutoScapulaPillarLandmarks" + self.shoulder.SCase.id4c + ".landmarkAscii"
        scapulaLandmarks.exportAmiraFile(self.shoulder.SCase.dataAmiraPath(), filename)

    def exportGroovePointsToAmiraFolder(self):
        scapulaLandmarks = LandmarksExporter.LandmarksExporter()
        scapulaLandmarks.addLandmarks(self.groove)

        if not os.path.isdir(self.shoulder.SCase.dataAmiraPath()):
            os.mkdir(self.shoulder.SCase.dataAmiraPath())

        filename = "AutoScapulaGrooveLandmarks" + self.shoulder.SCase.id4c + ".landmarkAscii"
        scapulaLandmarks.exportAmiraFile(self.shoulder.SCase.dataAmiraPath(), filename)

    def loadGroovePoints(self):
        """
        LOAD Load segmented surface and landmnarks
        Load the segmented scapula surface (if exist) and the
        scapula landmarks (6 scapula, 5 groove, 5 pillar) from
        amira directory.
        """
        SCase = self.shoulder.SCase
        filename = os.path.join(SCase.dataPythonPath(),
                                "scapulaLandmarksAuto"+self.shoulder.side+".pkl")

        with open(filename, 'rb') as f:
            ScapulaLandmarks = pickle.load(f)
        #remove squeez
        self.groove = ScapulaLandmarks["groove"]
        self.groove = self.getSortedGrooveLateralToMedial()

    def loadLandmarks(self):
        """
        LOAD Load segmented surface and landmnarks
        Load the segmented scapula surface (if exist) and the
        scapula landmarks (6 scapula, 5 groove, 5 pillar) from
        amira directory.
        """
        SCase = self.shoulder.SCase

        filename = os.path.join(SCase.dataPythonPath(), f"scapulaLandmarksAuto{self.shoulder.side}.pkl")
        with open(filename, "rb") as f:
            ScapulaLandmarks = pickle.load(f)
        self.angulusInferior = ScapulaLandmarks["angulusInferior"]
        self.trigonumSpinae = ScapulaLandmarks["trigonumSpinae"]
        self.processusCoracoideus = ScapulaLandmarks["processusCoracoideus"]
        self.acromioClavicular = ScapulaLandmarks["acromioClavicular"]
        self.angulusAcromialis = ScapulaLandmarks["angulusAcromialis"]
        self.spinoGlenoidNotch = ScapulaLandmarks["spinoGlenoidNotch"]

    def loadPillarPoints(self):
        """
        LOAD Load segmented surface and landmnarks
        Load the segmented scapula surface (if exist) and the
        scapula landmarks (6 scapula, 5 groove, 5 pillar) from
        amira directory.
        """

        SCase = self.shoulder.SCase
        filename = os.path.join(SCase.dataPythonPath(),
                                "scapulaLandmarksAuto"+self.shoulder.side+".pkl")
        with open(filename, 'rb') as f:
            ScapulaLandmarks = pickle.load(f)

        self.pillar = ScapulaLandmarks["pillar"]

    def loadSurface(self):
        SCase = self.shoulder.SCase
        filename = os.path.join(SCase.dataPythonPath(),
                                "scapulaSurfaceAuto"+self.shoulder.side+".ply")
        points, faces = loadPly(filename)
        self.surface["points"] = points
        self.surface["faces"] = faces
        self.segmentation = "A"
