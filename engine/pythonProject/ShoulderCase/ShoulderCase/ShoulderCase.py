import os
import pickle
import pandas as pd
import numpy as np
import plotly
import glob
from getConfig import getConfig
from ShoulderCase.shoulder.Shoulder import Shoulder
from ShoulderCase.Patient.Patient import Patient
from ShoulderCase.SCaseIDParser.SCaseIDParser import SCaseIDParser
from ShoulderCase.getTabulatedProperties import getTabulatedProperties
from pydicom import dcmread
#from ShoulderCase.ShoulderCasePlotter.ShoulderCasePlotter import ShoulderCasePlotter
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go

class ShoulderCase:
    """
    Properties and methods associated to the SoulderCase object.
    """

    def __init__(self, SCaseID, dataCTPath):
        self.diagnosis  = []
        self.treatment  = []
        self.outcome    = []
        self.study      = []
        self.comment    = []

        # SCaseID validation
        rawSCaseID = SCaseIDParser(SCaseID)
        assert rawSCaseID.isValidID(), "The input argument is not a valid SCaseID."
        self.id = SCaseID # id of the shoulder case, as it appears in the database (Pnnn)
        self.id4c = rawSCaseID.getIDWithNumberOfDigits(3) # SCaseID with P/N followed by 3 digits --> 4 char

        # path attribution
        self.dataCTPath = dataCTPath["pathWithPython"]
        self.CTKernel = dataCTPath["pathWithPython"].split("_")[3]
        self.allCTPaths = dataCTPath

        # Folder creation
        if not(os.path.isdir(self.dataPythonPath())):
            os.mkdir(self.dataPythonPath())

        # Initialisation
        self.patient = Patient(self)
        self.shoulders = {
            "left":{
                "auto":Shoulder(self, "L", "auto"),
                "manual":Shoulder(self, "L", "manual")
                },
            "right":{
                "auto":Shoulder(self, "R", "auto"),
                "manual":Shoulder(self, "R", "manual")
                }
            }

    def dataPath(self):
        return os.path.split(self.dataCTPath)

    def dataPythonPath(self):
        return os.path.join(self.dataCTPath, getConfig()["landmarkAndSurfaceFilesFolder"])

    def dataDicomPath(self):
        return os.path.join(self.dataCTPath, "dicom")

    def dataAmiraPath(self):
        return os.path.join(self.dataCTPath, "amira")

    def dataSlicerPath(self):
        return os.path.join(self.dataCTPath, "slicer")

    def savePython(self):

        # save SCase to python file
        dir_ = self.dataPythonPath()
        # create direction if not exist
        try:
            if not(os.path.isdir(dir_)): # create directory if it does not exist
                os.mkdir(dir_)
        except:
            print("Error creating the python directory \n") # should be in log

        # save SCase in python directory, in a file named SCaseCNNN.pkl
        filename = os.path.join(self.dataPythonPath(), "SCase.pkl")
        dataCTPath = self.dataCTPath
        allCTPaths = self.allCTPaths
        try:
            # Delete CT path which must be set when loading/creating the
            # SCase to avoid messing with paths on different systems.
            self.dataCTPath = ""
            self.allCTPaths = ""
            with open(filename, "wb") as pklFile:
                pickle.dump(self, pklFile)

        except:
            print("error creating SCase python file \n") # should be in log

        self.dataCTPath = dataCTPath
        self.allCTPaths = allCTPaths

    def saveExcel(self):
        return 1

    def saveSQL(self):
        return 1

    def getDataFrameOfData(self):
        # This method export all of the measurements and data of a
        # SCase to a pandas dataframe, which then can be used to export all of the
        # measurements as csv or excel file

        metadataSCase = getTabulatedProperties(self)
        metadataPatient = getTabulatedProperties(self.patient, prefix="patient", parentObjs=[self])

        # Right auto shoulder
        rightAutoData = getTabulatedProperties(self.shoulders["right"]["auto"], recursive=True, prefix="shoulder",
                                               arrays1x3AreCoordinatesXYZ=True,
                                               excludedProperties = ["surface", "landmarks", "points"],
                                               parentObjs=[self])

        rightAutoDataFrame = {}
        for d in [metadataSCase, metadataPatient, rightAutoData]:
            rightAutoDataFrame.update(d)
        rightAutoDataFrame = pd.DataFrame(rightAutoDataFrame, index=[0])

        # Right manual shoulder
        rightManualData = getTabulatedProperties(self.shoulders["right"]["manual"], recursive=True, prefix="shoulder",
                                                 arrays1x3AreCoordinatesXYZ=True,
                                                 excludedProperties = ["surface", "landmarks", "points"],
                                                 parentObjs=[self])

        rightManualDataFrame = {}
        for d in [metadataSCase, metadataPatient, rightManualData]:
            rightManualDataFrame.update(d)
        rightManualDataFrame = pd.DataFrame(rightManualDataFrame, index=[0])

        # Left auto shoulder
        leftAutoData = getTabulatedProperties(self.shoulders["left"]["auto"], recursive=True, prefix="shoulder",
                                              arrays1x3AreCoordinatesXYZ=True,
                                              excludedProperties=["surface", "landmarks", "points"],
                                              parentObjs=[self])

        leftAutoDataFrame = {}
        for d in [metadataSCase, metadataPatient, leftAutoData]:
            leftAutoDataFrame.update(d)
        leftAutoDataFrame = pd.DataFrame(leftAutoDataFrame, index=[0])

        # Left manual shoulder
        leftManualData = getTabulatedProperties(self.shoulders["left"]["manual"], recursive=True, prefix="shoulder",
                                                arrays1x3AreCoordinatesXYZ=True,
                                                excludedProperties=["surface", "landmarks", "points"],
                                                parentObjs=[self])
        leftManualDataFrame = {}
        for d in [metadataSCase, metadataPatient, leftManualData]:
            leftManualDataFrame.update(d)
        leftManualDataFrame = pd.DataFrame(leftManualDataFrame, index=[0])

        output = pd.concat([rightAutoDataFrame, rightManualDataFrame,
                           leftAutoDataFrame,  leftManualDataFrame], axis=0, ignore_index=True)

        return output

    def getSmoothDicomInfo(self):
        dicomFiles = os.listdir(os.path.join(self.getSmoothDicomPath(),'*.dcm'))
        assert dicomFiles, 'No dicom file found there %s' % self.getSmoothDicomPath()
        output = dcmread(os.path.realpath(dicomFiles))
        return output

    def getSmoothDicomPath(self):
        SCaseSmoothCTFolders = []
        smoothKernels = ["STANDARD", "DETAIL", "A", "B", "FC13", "B25s", "B26s", "B31s", "I31s"]
        for kernel in smoothKernels:
            if self.id[0] == "P":
                SCaseSmoothCTFolder = glob.glob(
                    os.path.join(f"{os.sep}".join(self.dataCTPath.split(os.sep)[:-1]), f"CT_*_shoulder_{kernel}_*_preop")
                )
                if SCaseSmoothCTFolder:
                    SCaseSmoothCTFolders.append(SCaseSmoothCTFolder[0])
            elif self.id[0] == "N":
                SCaseSmoothCTFolder = glob.glob(
                    os.path.join(f"{os.sep}".join(self.dataCTPath.split(os.sep)[:-1]), f"CT_*_shoulder_{kernel}_*")
                )
                if SCaseSmoothCTFolder:
                    SCaseSmoothCTFolders.append(SCaseSmoothCTFolder[0])

        smoothDicomPath = ""
        if len(SCaseSmoothCTFolders) > 1:
            for SCaseSmoothCTFolder in SCaseSmoothCTFolders:
                if "STANDARD" in SCaseSmoothCTFolder:
                    smoothDicomPath = SCaseSmoothCTFolder

        if smoothDicomPath == "":
            smoothDicomPath = SCaseSmoothCTFolders[0]

        smoothDicomPath = os.path.join(smoothDicomPath,"dicom")
        assert os.path.isdir(smoothDicomPath), 'No smooth dicom for this SCase. %s is not a valid folder' % smoothDicomPath
        return smoothDicomPath

    def plot(self,
             side,
             auto_normal,
             plot_landmarks=True,
             landmarksColor="black",
             scapulaSurfaceColor="rgb(244, 235, 188)",
             lightingFeatures=dict(ambient=0.5, diffuse=0.5, roughness = 0.9, fresnel=0.2, specular=0.6),
             lightPosition=dict(x=0, y = 0, z=0),
             plot_glenoid_surface=True,
             glenoidSurfaceColor="rgb(250, 0, 250)",
             plot_humeral_head=True,
             humeralHeadColor="fall",
             pathToSavePlot=os.getcwd()):
        #ShoulderCasePlotter(self, *args)
        landmarksFig = self.shoulders[side][auto_normal].scapula.plotLandmarks(landmarksColor)
        scapulaFig = self.shoulders[side][auto_normal].scapula.plotSurface(scapulaSurfaceColor, lightingFeatures, lightPosition)
        glenoidFig = self.shoulders[side][auto_normal].scapula.glenoid.plot(glenoidSurfaceColor)
        coorSysFig = self.shoulders[side][auto_normal].scapula.coordSys.plot(False)
        if plot_landmarks and plot_glenoid_surface and plot_humeral_head:
            try:
                humeralHead = self.shoulders[side][auto_normal].humerus.plot(humeralHeadColor)
                fig = go.Figure(
                        data=landmarksFig.data + scapulaFig.data + glenoidFig.data + coorSysFig.data + humeralHead.data)
            except:
                fig = go.Figure(data=landmarksFig.data + scapulaFig.data + glenoidFig.data + coorSysFig.data )

        elif plot_landmarks and plot_glenoid_surface and not plot_humeral_head:
            fig = go.Figure(data=landmarksFig.data + scapulaFig.data + glenoidFig.data + coorSysFig.data)

        elif plot_landmarks and not plot_glenoid_surface and not plot_humeral_head:
            fig = go.Figure(data=landmarksFig.data + scapulaFig.data + coorSysFig.data)

        elif not plot_landmarks and not plot_glenoid_surface and not plot_humeral_head:
            fig = go.Figure(data=scapulaFig.data + coorSysFig.data)

        fig.update_layout(
            paper_bgcolor="rgba(255, 255, 255, 0.8)",
            plot_bgcolor="rgba(255, 255, 255, 0.8)",
            title=self.id + "-" + side,
            font=dict(
                family="Courier New, monospace",
                size=18
            )
        )

        fig.update_layout(scene=dict(
            xaxis=dict(
                backgroundcolor="white",
                gridcolor="white",
                showbackground=False,
                zerolinecolor="white",
                visible=False),
            yaxis=dict(
                backgroundcolor="rgb(230, 200,230)",
                gridcolor="white",
                showbackground=False,
                zerolinecolor="white",
                visible=False),
            zaxis=dict(
                backgroundcolor="rgb(230, 230,200)",
                gridcolor="white",
                showbackground=False,
                zerolinecolor="white",
                visible=False))
        )

        fig.show()
        fig.write_html(os.path.join(pathToSavePlot, f"{self.id}Plot.html"))

        return fig

    def plotManualAutoDifference(self, shoulderSide, normLimitForOutliers):
        """
        Plot the vectors between auto and manual points.

         input: normLimitForOutliers

        The vector is green if below the value given to normLimitForOutliers,
        otherwise it is red.
        """
        # initialisation
        manual = self.shoulders.shoulderSide.manual
        manualPoints = np.array([])

        auto = self.shoulders.shoulderSide.auto
        autoPoints = np.array([])

        # find maximum number of common groove points
        minGroovePoints = min(manual.scapula.groove.shape, auto.scapula.groove.shape)
        if (minGroovePoints > 0):
            manualPoints = manual.scapula.groove[:minGroovePoints,:]
            autoPoints = auto.scapula.groove[:minGroovePoints,:]

        # create points arrays
        manualPoints = np.concatenate([manualPoints,
                        manual.scapula.angulusInferior,
                        manual.scapula.trigonumSpinae,
                        manual.scapula.processusCoracoideus,
                        manual.scapula.acromioClavicular,
                        manual.scapula.angulusAcromialis,
                        manual.scapula.spinoGlenoidNotch,
                        manual.scapula.coordSys.origin,
                        manual.scapula.glenoid.center])

        autoPoints = np.concatenate([manualPoints,
                        auto.scapula.angulusInferior,
                        auto.scapula.trigonumSpinae,
                        auto.scapula.processusCoracoideus,
                        auto.scapula.acromioClavicular,
                        auto.scapula.angulusAcromialis,
                        auto.scapula.spinoGlenoidNotch,
                        auto.scapula.coordSys.origin,
                        auto.scapula.glenoid.center])

        # measure norm difference
        manualAutoDifference = np.linalg.norm(autoPoints-manualPoints,2,2)

        # plot the difference vectors
        for i in range(manualAutoDifference.shape[0]):
            if manualAutoDifference[i] > normLimitForOutliers:
                # outlier
                color = 'red'
            else:
                # valid point
                color = 'green'
            differenceLine = np.concatenate([manualPoints[i,:], autoPoints[i,:]])

            Axes3D.plot(differenceLine[:,0],
                                        differenceLine[:,1],
                                        differenceLine[:,2],
                                        color=color, lw=3)
