import os
from ShoulderCase.Humerus.Humerus import Humerus
from ShoulderCase.RotatorCuff.RotatorCuff import RotatorCuff
from ShoulderCase.ScapulaAuto.ScapulaAuto import ScapulaAuto
from ShoulderCase.ScapulaManual.ScapulaManual import ScapulaManual
from ShoulderCase.HumerusAuto.HumerusAuto import HumerusAuto
from ShoulderCase.HumerusManual.HumerusManual import HumerusManual
from ShoulderCase.AutoFE.AutoFEAbaqus import AutoFEAbaqus
from utils.Logger.Logger import Logger
from utils.recursiveMethodCall import recursiveMethodCall
from getConfig import getConfig


class Shoulder:
    """
    Contains all the shoulder parts (bones) and their measurements.

    Can be used to plot an overview of the case.
    """

    def __init__(self, SCase, shoulderSide, landmarksAcquisition):
        self.side = shoulderSide
        self.landmarksAcquisition = landmarksAcquisition
        self.SCase = SCase
        if not os.path.isdir(self.dataPath()):
            os.makedirs(self.dataPath())

        #self.rotatorCuff = RotatorCuff.RotatorCuff(self)
        if landmarksAcquisition == "auto":
            self.scapula = ScapulaAuto(self)
        elif landmarksAcquisition == "manual":
            self.scapula = ScapulaManual(self)

        if landmarksAcquisition == "auto":
            self.humerus = HumerusAuto(self)
        elif landmarksAcquisition == "manual":
            self.humerus = HumerusManual(self)

        config = getConfig()["runMeasurements"]
        if config["sliceRotatorCuffMuscles"] or config["segmentRotatorCuffMuscles"]:
            self.rotatorCuff = RotatorCuff(self)

        if getConfig()["runFE"]:
            self.FE = AutoFEAbaqus(self)

        self.CTSCan = ""
        self.comment = ""
        self.hasMeasurement = ""

    def hasMeasurements(self):
        self.hasMeasurement = not self.isempty()

    def isempty(self):
        return self.scapula.isempty()

    def explicitSide(self):
        explicitSide = self.side
        explicitSide = explicitSide.replace("R", "right")
        explicitSide = explicitSide.replace("L", "left")
        return explicitSide

    def dataPath(self):
        return os.path.join(self.SCase.dataPythonPath(),
                            "shoulders",
                            self.explicitSide(),
                            self.landmarksAcquisition)

    def loadData(self):
         Logger.newDelimitedSection("Load data")
         Logger.logn("")
         recursiveMethodCall(self, "loadData", [self.SCase])
         Logger.closeSection()
         self.hasMeasurements()

    def morphology(self):
         Logger.newDelimitedSection("Morphology")
         Logger.logn("")
         recursiveMethodCall(self, "morphology", [self.SCase])
         Logger.closeSection()

    def measureFirst(self):
         Logger.newDelimitedSection("First measurements")
         Logger.logn("")
         recursiveMethodCall(self, "measureFirst", [self.SCase])
         Logger.closeSection()

    def measureSecond(self):
         Logger.newDelimitedSection("Second  measurements")
         Logger.logn("")
         recursiveMethodCall(self, "measureSecond", [self.SCase])
         Logger.closeSection()

    def measureThird(self):
         Logger.newDelimitedSection("Third  measurements")
         Logger.logn("")
         recursiveMethodCall(self, "measureThird", [self.SCase])
         Logger.closeSection()

    def measureDensity(self):
        self.scapula.glenoid.measureDensity()

    def measureMuscles(self):
        self.rotatorCuff.sliceAndSegment()
        self.rotatorCuff.measure("rotatorCuffMatthieu", "autoMatthieu")

    def runFE(self):
        self.FE.runAbaqus()
