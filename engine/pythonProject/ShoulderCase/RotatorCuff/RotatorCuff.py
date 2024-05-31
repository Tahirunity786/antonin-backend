import pydicom
from ShoulderCase.Muscle.Muscle import Muscle
import os
import pandas as pd
from ShoulderCase.getTabulatedProperties import getTabulatedProperties
from utils.Logger.Logger import Logger
from ShoulderCase.DicomVolumeSlicer.DicomVolumeSlicer import DicomVolumeSlicer
import numpy as np
import cv2
import pickle
from ShoulderCase.DicomVolume.readDicomVolume import readDicomVolume
from pydicom import dcmread
from scipy.ndimage import gaussian_filter
from utils.Rotations.rotation_angle import angle_of_rotation_from_vectors, axis_of_rotation_from_vectors
from utils.Slicer.SlicerControlPoint.SlicerControlPoint import SlicerControlPoint
from utils.Slicer.SlicerMarkupsExporter.SlicerMarkupsExporter import SlicerMarkupsExporter
from utils.Slicer.SlicerMarkupsLine.SlicerMarkupsLine import SlicerMarkupsLine
from utils.Vector.Vector import Vector
from getConfig import getConfig
import shutil
import subprocess
import warnings
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import UID
from copy import deepcopy
import plotly.graph_objects as go
import sys

class RotatorCuff:

    def __init__(self, shoulder):

        self.shoulder = shoulder

        self.SC = Muscle(self, "SC")
        self.SS = Muscle(self, "SS")
        self.IS = Muscle(self, "IS")
        self.TM = Muscle(self, "TM")

        self.imbalanceAngle3D = []
        self.imbalanceAngleOrientation = []
        self.imbalanceAngleAnteroPosterior = []
        self.imbalanceAngleInferoSuperior = []

        if not os.path.isdir(self.dataPath()):
            os.mkdir(self.dataPath())

        if getConfig()["runMeasurements"]["sliceRotatorCuffMuscles"]:
            if os.path.isdir(self.dataSlicePath()):
                shutil.rmtree(self.dataSlicePath())
                os.mkdir(self.dataSlicePath())
            else:
                os.mkdir(self.dataSlicePath())

        self.dicomInfo = ""

        self.obliqueSlices = []
        self.obliqueSlicesCoordinates = []

    def dataPath(self):
        return os.path.join(self.shoulder.dataPath(), "muscles")

    def dataSlicePath(self):
        return os.path.join(self.dataPath(), "oblique_slices")

    def summary(self):
        return pd.concat([getTabulatedProperties(self.SC),
                          getTabulatedProperties(self.SS),
                          getTabulatedProperties(self.IS),
                          getTabulatedProperties(self.TM)])

    def measureSecond(self):
        # Call methods that can be run after measureFirst() methods has been run by all ShoulderCase objects.
        success = Logger.timeLogExecution("Rotator cuff antero-posterior imbalance: ",
                                          lambda self: self.measureImbalanceAngles(),
                                          self)
        return success

    def createSliceMatthieu(self, numberOfSlices):
        """
        This function and its results might be referred as Matthieu algorithm and Matthieu slices in some studies.

        Images and pixel spacings are saved at:
        shoulder/dataDev/*/*/*/SCase-IPP/CT-SCase-*/python/shoulder/muscles/oblique_slice/
        """
        assert (not (self.shoulder.scapula.coordSys.isEmpty()), 'Scapula coordinate system not measured.')
        assert (numberOfSlices == 1 or numberOfSlices == 3 or numberOfSlices == 10), "Number of slices should be 1, 3 or 10"

        rcSlicer = self.loadAndNormalizeDicom()

        if numberOfSlices == 1:
            slices = self.getSlices(rcSlicer, 0) # 0 is the medialisation for the central slice
            sliceName = "obliqueSlice"
            self.saveImages(slices["forSegmentation"], slices["forMeasurements"], slices["pixelSpacings"], sliceName)
            self.saveImagesPixelSpacings(slices["pixelSpacings"], sliceName)
            self.saveImagesPixelCoordinates(slices["pixelCoordinates"], sliceName)

        elif numberOfSlices == 3:

            widthToPutMedialSlices = abs(
                self.shoulder.scapula.coordSys.express(self.shoulder.scapula.trigonumSpinae)[0, -1]
            )
            medialisationStep = widthToPutMedialSlices / 8
            laterlaSlices = [-medialisationStep]
            centerSlice = [0]
            medialSlices = [medialisationStep]
            medialisations = laterlaSlices + centerSlice + medialSlices
            sliceNames = ["Lateral", "Central", "Medial"]

            for sliceNumber, medialisation in enumerate(medialisations):
                slices = self.getSlices(rcSlicer, medialisation)
                sliceName = f"ObliqueSlice{sliceNames[sliceNumber]}"
                self.saveImages(slices["forSegmentation"], slices["forMeasurements"], sliceName)
                self.saveImagesPixelSpacings(slices["pixelSpacings"], sliceName)
                self.saveImagesPixelCoordinates(slices["pixelCoordinates"], sliceName)

        elif numberOfSlices == 10:
            widthToPutMedialSlices = abs(self.shoulder.scapula.coordSys.express(self.shoulder.scapula.trigonumSpinae)[0, -1])
            medialisationStep = widthToPutMedialSlices / 8
            laterlaSlices = [-3*medialisationStep, -2*medialisationStep, -medialisationStep]
            centerSlice = [0]
            medialSlices = list(np.arange(medialisationStep, widthToPutMedialSlices+0.1, medialisationStep))
            medialisations = laterlaSlices + centerSlice + medialSlices
            sliceNames = [f"0{i}" for i in range(1, 10)] + ["10", "11", "12"]

            for sliceNumber, medialisation in enumerate(medialisations):
                slices = self.getSlices(rcSlicer, medialisation)
                sliceName = f"_obliqueSlice_{sliceNames[sliceNumber]}"
                self.saveImages(slices["forSegmentation"], slices["forMeasurements"], sliceName)
                self.saveImagesAsDicom(slices["forMeasurements"], sliceName)
                self.saveImagesPixelSpacings(slices["pixelSpacings"], sliceName)
                self.saveImagesPixelCoordinates(slices["pixelCoordinates"], sliceName)

    def loadAndNormalizeDicom(self):
        SCase = self.shoulder.SCase

        try:
            rotatorCuffSlicer = DicomVolumeSlicer(SCase.getSmoothDicomPath())
        except:
            print(f"No soft CT for the {SCase.id} subject!")
            #rotatorCuffSlicer = DicomVolumeSlicer(SCase.dataDicomPath())
            #rotatorCuffSlicer.applyGaussianFilter()

        self.dicomInfo = rotatorCuffSlicer.dicomInfo

        # The 3D volume must be normalised for the slices not to be distorted
        rotatorCuffSlicer.setMinimalResolutionDividedBy(2)
        rotatorCuffSlicer.normaliseVolumeMain()

        return rotatorCuffSlicer

    def getSlices(self, rotatorCuffSlicer, medialisation):
        scapula = self.shoulder.scapula

        origin = scapula.coordSys.origin - medialisation * scapula.coordSys.ML
        rotatorCuffSlicer.slice(origin, scapula.coordSys.ML)

        if getConfig()["croppedRotatorCuff"]:
            # The following lines set the slice to display a 200x150 mm area around the scapula with the "Y" shape of the scapula oriented up
            rotatorCuffSlicer.orientSliceUpwardVector(origin, scapula.coordSys.IS)
            height = 200
            width = 150
            inferiorisation = 40
            rotatorCuffSlicer.crop(origin - inferiorisation * scapula.coordSys.IS, height, width)

        # Rotator cuff segmentation code requires 1024x1024 pixels images
        if rotatorCuffSlicer.sliced.shape[0] > rotatorCuffSlicer.sliced.shape[1]:
            sliceWidth = rotatorCuffSlicer.sliced.shape[0]
            aspectRatio = 1024/sliceWidth
            sliceNewHeight = int(aspectRatio*rotatorCuffSlicer.sliced.shape[1])+1
            rotatorCuffSlicer.resize((1024, sliceNewHeight))
        else:
            sliceHeight = rotatorCuffSlicer.sliced.shape[1]
            aspectRatio = 1024 / sliceHeight
            sliceNewWidth = int(aspectRatio * rotatorCuffSlicer.sliced.shape[0]) + 1
            rotatorCuffSlicer.resize((sliceNewWidth, 1024))

        rotatorCuffSlicer.addEmptyBackgroundToSlice(1024, 1024, emptyIsZero=False)

        rawSlice = rotatorCuffSlicer.sliced

        rotatorCuffSlicer.rescaleSliceToUint8()
        slice8Bit = rotatorCuffSlicer.sliced
        rotatorCuffSlicer.sliced = rawSlice

        rotatorCuffSlicer.rescaleSliceToInt16()
        slice16Bit = rotatorCuffSlicer.sliced

        slicePixelCoordinates = rotatorCuffSlicer.getSlicedPixelCoordinates()

        # Flip the slices for right and left shoulders to give similar slices
        if self.shoulder.side == "L":
            slice8Bit = np.rot90(np.flipud(slice8Bit), 2)
            slice16Bit = np.rot90(np.flipud(slice16Bit), 2)
            slicePixelCoordinates["x"] = np.rot90(np.flipud(slicePixelCoordinates["x"]), 2)
            slicePixelCoordinates["y"] = np.rot90(np.flipud(slicePixelCoordinates["y"]), 2)
            slicePixelCoordinates["z"] = np.rot90(np.flipud(slicePixelCoordinates["z"]), 2)

        slicePixelSpacings = rotatorCuffSlicer.slicedPixelSpacings

        output = {}
        output["forSegmentation"] = slice8Bit
        output["forMeasurements"] = slice16Bit
        output["pixelSpacings"] = slicePixelSpacings
        output["pixelCoordinates"] = slicePixelCoordinates

        return output

    def saveImages(self, imageForSegmentation, imageForMeasurements, sliceName):

        cv2.imwrite(
            os.path.join(self.dataSlicePath(), f"{sliceName}ForSegmentation.png"),
            imageForSegmentation
        )
        with open(os.path.join(self.dataSlicePath(), f"{sliceName}ForMeasurements.npy"), mode="wb") as f:
            np.save(f, imageForMeasurements)

    def saveImagesAsDicom(self, image, sliceName):

        dicomPath = os.path.join(self.dataSlicePath(),
                                 f"{self.shoulder.SCase.id}_{self.shoulder.side}{sliceName}.dcm")

        dicomData = deepcopy(self.dicomInfo)

        dicomData.PatientID = f"{self.shoulder.SCase.id}_{self.shoulder.side}{sliceName}"
        dicomData.PatientName = ""
        dicomData.PatientBirthDate = ""
        dicomData.PhotometricInterpretation = "MONOCHROME2"
        dicomData.RescaleIntercept = 0
        dicomData.RescaleSlope = 1
        dicomData.SamplesPerPixel = 1
        dicomData.BitsStored = 16
        dicomData.BitsAllocated = 16
        dicomData.HighBit = 15
        dicomData.PixelRepresentation = 1
        dicomData.PixelData = image.tobytes()
        dicomData.Rows = image.shape[0]
        dicomData.Columns = image.shape[1]

        dicomData.save_as(dicomPath)

    def saveImagesPixelSpacings(self, imagesPixelSpacings, sliceName):
        with open(os.path.join(self.dataSlicePath(), f"{sliceName}PixelSpacings.npy"), mode="wb") as f:
            np.save(f, imagesPixelSpacings)

    def saveImagesPixelCoordinates(self, imagesPixelCoordinates, sliceName):
        imagesPixelCoordinates["x"] = imagesPixelCoordinates["x"].tolist()
        imagesPixelCoordinates["y"] = imagesPixelCoordinates["y"].tolist()
        imagesPixelCoordinates["z"] = imagesPixelCoordinates["z"].tolist()
        with open(os.path.join(self.dataSlicePath(), f"{sliceName}PixelCoordinates.pkl"), mode="wb") as f:
            pickle.dump(imagesPixelCoordinates, f)

    def createSliceNathan(self):
        """
        This function and its results might be reffered as Nathan algorithm and Nathan slices in some studies.

        One needs to create two empty files in the current directory :
        data_degenerescence.mat (a table) and Didnotwork.mat (a%string)

        This script calculates the degeneration of the 4 muscles of the rotator cuff : SS, SC, IS, TM.
        It is composed of four different steps :
            - Load CT images (DICOM files) in order to create a 3D matrix consituted with HU (Hounsfield unit) values
            - Create a slice of this 3D matrix along a patient-specific plane.
            Do image processing and set saving format usable by automatic segmentation code (Python's code from
            ARTORG team (UNIBE))
            - Execute automatic segmentation code on the new created images
            - Apply degeneration code (from Laboratoire de biomecanique orthopedique (EPFL)) on new images with automatic
            segmentation and on image with manual segmentation (from CHUV radiologist).
            Save degeneration values under the "muscles" properties of each case in the LBO database.

        Some "try-catch" are present in order to catch the errors while keeping running the script on the rest of the database.
        They are summarized underneath :
        (DicomReadVolume) : do not manage to load the dicoms
        Nothing : Automatic segmentation does not work (generally for TM)
        * : slice function does not work
        ** : does not want to crop image
        *** : black image, don't want to save png
        """

        # Load dicom
        SCase = self.shoulder.SCase
        dataCTPath = SCase.dataCTPath

        # Create "muscles" folder (and its 4 subfolders) if it does not exist
        folderMuscle = os.path.join(dataCTPath, "muscles")

        if os.path.isdir(folderMuscle):
            pass
        else:
            os.mkdir(dataCTPath, "muscles")
            os.mkdir(folderMuscle, "IS")
            os.mkdir(folderMuscle, "SS")
            os.mkdir(folderMuscle, "SC")
            os.mkdir(folderMuscle, "TM")

        # Use folder "CT-2" (smoothed kernel) if it exists
        dataCTPath2 = dataCTPath + "2"

        if os.isdir(dataCTPath2):
            dataCTPath0 = dataCTPath2
        else:
            dataCTPath0 = dataCTPath

        dicomFolder = os.path.join(dataCTPath0, "dicom")

        # Extract the images as a 4D matrix where spatial is the positions of the Pixels in the CT coordinate system
        # Return the case's number if it does not manage to load the DICOM files
        try:
            dicomData = readDicomVolume(dicomFolder)
            V = dicomData["V"]
            patientPositions = dicomData["PatientPositions"]
            pixelSpacings = dicomData["PixelSpacings"]
        except:
            return 0

        # Read the information from the middle dicom image (all image informations are the same for all dicom files)
        # to avoid a problem if other files are located at the beginning or end of the folder
        dicomList = os.listdir(dicomFolder)
        dicomInformation = dcmread(dicomList[int(len(dicomList)/2)])

        # Take into account that the data will need to be linearly transformed from stored space to memory to
        # come back to true HU units afterwards:
        rescaleSlope = float(dicomInformation.RescaleSlope)
        rescaleIntercept = float(dicomInformation.ResRescaleIntercept)

        # To filter hard kernel images
        if not os.isdir(dataCTPath2):
            V = gaussian_filter(V, sigma=1)

        # Get local coordinate system from parent scapula object(glenoid center)
        origin = self.shoulder.scapula.coordSys.origin # do not mixup with origin_image, "origin" is the origin of the Scapula Coordinate System
        xAxis = self.shoulder.scapula.coordSys.PA
        yAxis = self.shoulder.scapula.coordSys.IS
        zAxis = self.shoulder.scapula.coordSys.ML

        # Slice from the CT data set
        # Get the position of the upper left pixel of the first image and calculate the position of each pixel in the x,y
        # and z direction starting from the "origin" pixel
        originImage = patientPositions[0, :] # coordinates of the first pixel
        xMax = originImage[0] + pixelSpacings[0, 0] * (V.shape[0] - 1)
        yMax = originImage[1] + pixelSpacings[0, 1]* (V.shape[1] - 1)
        zMax = patientPositions[-1, 2]

        # Calculate the coefficient of the linear transformation (CT-scan coordinate system (x,y,z)
        # to images coordinates system (i,j,k))
        coefficients_i = np.polyfit([0, V.shape[0]], [originImage[0], xMax], 1)
        a_i = coefficients_i[0]
        b_i = coefficients_i[1]

        coefficients_j = np.polyfit([0, V.shape[1]], [originImage[1], yMax], 1)
        a_j = coefficients_j[0]
        b_j = coefficients_j[1]

        coefficients_k = np.polyfit([0, V.shape[2]], [originImage[2], zMax], 1)
        a_k = coefficients_k[0]
        b_k = coefficients_k[1]

        pixelPosX = np.arange(0, V.shape[0])*a_i + b_i # Create the position vector along x-axis
        pixelPosY = np.arange(0, V.shape[0])*a_j + b_j
        pixelPosZ = np.arange(0, V.shape[0])*a_k + b_k

        pixSizeZ = (zMax - originImage[2]) / (V.shape[2] - 1) # Space between 2 slices

        # The oblique slice to be created is perpendicular to the z axis of the scapula CS and passes through the
        # origin point. We will apply the matlab function "slice", to create the oblique slice.
        # One must first create a slice parallel to 2 system's axis. Here we take a plane parallel to the (x,y)
        # plane, passing through the point "origin". Then, we need to rotate this plane so that it becomes normal to the
        # zAxis. V1 is the vector normal to the horizontal slice and v2 the vector normal to the final oblique slice,
        # parallel to the zAxis. We need to rotate around a vector v3 and from an angle given by "vrrotvec(v1,v2)".
        v1 = np.array([0, 0, 1])
        v2 = zAxis/np.linalg.norm(zAxis)
        theta = angle_of_rotation_from_vectors(v1, v2)
        v3 = axis_of_rotation_from_vectors(v1, v2)

        # Create the plane to be rotated
        x, y, z = np.meshgrid(pixelPosX, pixelPosY, pixelPosZ)
        s = 1000 # Extension factor of the slice in order to cut the whole matrix.Set arbitrarily
        # but sufficiently big so that it works for every patient
        PixelPosXEnlarged = np.hstack([
            originImage[0] + np.arange(-pixelSpacings[0, 0] * s, -pixelSpacings[0, 0], pixelSpacings[0, 0]),
            pixelPosX,
            xMax + np.arange(pixelSpacings[0, 0], pixelSpacings[0, 0] * s, pixelSpacings[0, 0])])
        PixelPosYEnlarged = np.hstack([
            originImage[1] + np.arange(-pixelSpacings[0, 1] * s, -pixelSpacings[0, 1], pixelSpacings[0, 1]),
            pixelPosY,
            yMax + np.arange(pixelSpacings[0, 1], pixelSpacings[0, 1] * s, pixelSpacings[0, 1])])
        # To be continued

    def exportMusclesFibresForSlicer(self):
        color = {}
        color["SC"] = "blue"
        color["SS"] = "red"
        color["IS"] = "green"
        color["TM"] = "yellow"

        for muscleName, muscle in zip(["SC", "SS", "IS", "TM"], [self.SC, self.SS, self.IS, self.TM]):
            slicerExporter = SlicerMarkupsExporter
            for i in range(muscle.centroid.shape[0]):
                fibre = SlicerMarkupsLine

                centroid = SlicerControlPoint("centroid_" + i, muscle.centroid[i, :])
                centroid.locked = True
                fibre.addControlPoint(centroid)

                forceApplicationPoint = SlicerControlPoint("contact_point_" + i, muscle.forceApplicationPoint[i, :]);
                forceApplicationPoint.locked = True
                fibre.addControlPoint(forceApplicationPoint)

                fibre.setColor(color[muscleName])
                slicerExporter.addMarkups(fibre)

            outputFileName = "_".join(["fibres", muscle.getFullName(), self.shoulder.side, ".mrk.json"])
            slicerExporter.export(os.path.join(self.shoulder.SCase.dataSlicerPath(), outputFileName))

    def measureImbalanceAngles(self):
        scapulaCS = self.shoulder.scapula.coordSys

        forceResultant = Vector(scapulaCS.origin, scapulaCS.origin) + \
            self.SC.getForceResultant() + \
            self.SS.getForceResultant() + \
            self.IS.getForceResultant() + \
            self.TM.getForceResultant()

        medioLateralAxis = Vector(scapulaCS.origin, scapulaCS.origin + scapulaCS.ML)
        inferoSuperiorAxis = Vector(scapulaCS.origin, scapulaCS.origin + scapulaCS.IS)
        anteroPosteriorAxis = Vector(scapulaCS.origin, scapulaCS.origin - scapulaCS.PA)

        self.imbalanceAngle3D = forceResultant.angle(-medioLateralAxis)*180/np.pi

        # Take medio-lateral orthogonal component
        # <=> project on scapular sagittal plane
        imbalanceVectorSagittalProjection = medioLateralAxis.orthogonalComplementTo(forceResultant)
        # superior imbalance is positive
        self.imbalanceAngleOrientation = np.sign(np.dot(imbalanceVectorSagittalProjection, inferoSuperiorAxis)) * \
                                         imbalanceVectorSagittalProjection.angle(anteroPosteriorAxis)*180/np.pi

        # Take infero-superior orthogonal component
        # <=> project on scapular axial plane
        imbalanceVectorAxialProjection = inferoSuperiorAxis.orthogonalComplementTo(forceResultant)

        # posterior imbalance is positive
        self.imbalanceAngleAnteroPosterior = np.sign(np.dot(imbalanceVectorAxialProjection, anteroPosteriorAxis)) *\
                                             imbalanceVectorAxialProjection.angle(-medioLateralAxis)*180/np.pi

        # Take antero-posterior orthogonal component
        # <=> project on scapular frontal plane
        imbalanceVectorFrontalProjection = anteroPosteriorAxis.orthogonalComplementTo(forceResultant)
        # superior imbalance is positive
        self.imbalanceAngleInferoSuperior = np.sign(np.dot(imbalanceVectorFrontalProjection, inferoSuperiorAxis)) * \
                                            imbalanceVectorFrontalProjection.angle(-medioLateralAxis)*180/np.pi

    def segmentMuscles(self):
        """
        sliceName is the part before the '_ForSegmentation.png' part of the name of the file that is sent to rcseg.
        maskName is the part before the '_Segmentation.png' part of the name of the file that is saved in the SCase folder.
        """
        self.cleanRotatorCuffSegmentationWorkspace()
        self.sendImageToRotatorCuffSegmentationWorkspace()
        self.callRotatorCuffSegmentation()
        self.saveSegmentationResults()
        self.cleanRotatorCuffSegmentationWorkspace()

    def cleanRotatorCuffSegmentationWorkspace(self):
        rotatorCuffSegmentationPath = getConfig()["muscleSegmentationModelDir"]
        cleanDirectory(os.path.join(rotatorCuffSegmentationPath, "input"))
        cleanDirectory(os.path.join(rotatorCuffSegmentationPath, "IS"))
        cleanDirectory(os.path.join(rotatorCuffSegmentationPath, "SC"))
        cleanDirectory(os.path.join(rotatorCuffSegmentationPath, "SS"))
        cleanDirectory(os.path.join(rotatorCuffSegmentationPath, "TM"))

    def sendImageToRotatorCuffSegmentationWorkspace(self):
        SCase = self.shoulder.SCase
        rotatorCuffSegmentationPath = getConfig()["muscleSegmentationModelDir"]

        if getConfig()["numberOfObliqueSlices"] == 3:
            for sliceName in [f"ObliqueSlice{i}ForSegmentation.png" for i in ["Central", "Lateral", "Medial"]]:
                imageForSegmentationPath = os.path.join(self.dataSlicePath(), sliceName)
                shutil.copy(
                    imageForSegmentationPath,
                    os.path.join(rotatorCuffSegmentationPath, "input", f"{SCase.id}{sliceName}")
                )

    def callRotatorCuffSegmentation(self):
        # Apply UNIBE method to automatically segment muscle

        muscleSegmentationModelDir = getConfig()["muscleSegmentationModelDir"]

        if getConfig()["OS"] == "windows":
            segmentationModelVenvPythonPath = os.path.join(muscleSegmentationModelDir,
                                                           "windowsenv",
                                                           "Scripts",
                                                           "python.exe")
            commandToRun = [segmentationModelVenvPythonPath,
                            "rcseg.py",
                            "segment",
                            "input"]

            cwd = os.getcwd()
            os.chdir(muscleSegmentationModelDir)
            subprocess.run(commandToRun)
            os.chdir(cwd)

        elif getConfig()["OS"] == "linux":
            pythonCommandActivateEnvironment = "source " + os.path.join(muscleSegmentationModelDir,"venv", "bin", "activate") + ";"
            pythonCommandMoveToSegmentationWorkspace = "cd " + muscleSegmentationModelDir + ";"
            pythonCommandExecuteSegmentation = os.path.join(muscleSegmentationModelDir, "rcseg.py") + " segment input" + ";"
            os.system(pythonCommandActivateEnvironment)
            os.system(pythonCommandMoveToSegmentationWorkspace)
            os.system(pythonCommandExecuteSegmentation)

    def saveSegmentationResults(self):
        SCaseID = self.shoulder.SCase.id
        rotatorCuffSegmentationPath = getConfig()["muscleSegmentationModelDir"]

        for muscleName in ["SC", "SS", "IS", "TM"]:
            if getConfig()["numberOfObliqueSlices"] == 3:
                for slice_ in ["Central", "Lateral", "Medial"]:
                    sliceName = f"ObliqueSlice{slice_}ForSegmentation.png"
                    try:
                        shutil.copy(
                            os.path.join(rotatorCuffSegmentationPath, muscleName,  f"{SCaseID}{sliceName}"),
                            os.path.join(getattr(self, f"{muscleName}").dataMaskPath(),  f"Segmented{slice_}Slice.png")
                        )
                    except Exception as e:
                        warnings.warn(e)

    def slice(self, doSlicing):
        if doSlicing:
            self.createSliceMatthieu(getConfig()["numberOfObliqueSlices"])

    def segment(self, doSegmentation):
        if doSegmentation:
            self.segmentMuscles()

    def plotObliqueSlices(self):

        side = "right" if self.shoulder.side == "R" else "left"

        scapulaFig = self.shoulder.SCase.shoulders[side]["auto"].scapula.plotSurface(
            "rgb(244, 235, 188)",
            dict(ambient=0.5, diffuse=0.5, roughness = 0.9, fresnel=0.2, specular=0.6),
            dict(x=0, y = 0, z=0)
        )
        coorSysFig = self.shoulder.SCase.shoulders[side]["auto"].scapula.coordSys.plot(False)

        obliqueSliceFigData = scapulaFig.data + coorSysFig.data


        imageFiles = [f"obliqueSlice{i}_ForMeasurements" for i in
                               ["_l3", "_l2", "_l1", "_c", "_m1", "_m2", "_m3", "_m4", "_m5", "_m6", "_m7", "_m8"]]
        coordinateFiles = [f"obliqueSlice{i}_PixelCoordinates" for i in
                               ["_l3", "_l2", "_l1", "_c", "_m1", "_m2", "_m3", "_m4", "_m5", "_m6", "_m7", "_m8"]]

        for slice in range(10):

            with open(os.path.join(self.dataSlicePath(), coordinateFiles[slice] + ".pkl"), "rb") as f:
                coordinates = pickle.load(f)

            X = np.array(coordinates["x"])
            Y = np.array(coordinates["y"])
            Z = np.array(coordinates["z"])
            x = X[np.where((~np.isnan(X)) & (~np.isnan(Y)) & (~np.isnan(Z)))]
            y = Y[np.where((~np.isnan(X)) & (~np.isnan(Y)) & (~np.isnan(Z)))]
            z = Z[np.where((~np.isnan(X)) & (~np.isnan(Y)) & (~np.isnan(Z)))]
            diff = z.shape[0] - int(np.sqrt(z.shape)) * int(np.sqrt(z.shape))
            z2 = z[:-diff].reshape((int(np.sqrt(z.shape)), int(np.sqrt(z.shape))))
            x2 = x[:-diff].reshape((int(np.sqrt(z.shape)), int(np.sqrt(z.shape))))
            y2 = y[:-diff].reshape((int(np.sqrt(z.shape)), int(np.sqrt(z.shape))))

            image = np.load(os.path.join(self.dataSlicePath(), imageFiles[slice] + ".npy"), allow_pickle=True)

            fig = go.Figure(data=go.Surface(x=x2, y=y2, z=z2,
                                            colorscale="gray",
                                            #surfacecolor=image,
                                            ))
            obliqueSliceFigData += fig.data

        fig = go.Figure(data=obliqueSliceFigData)
        fig.show()


def cleanDirectory(path):
    for filename in os.listdir(path):
        filePath = os.path.join(path, filename)
        try:
            if os.path.isfile(filePath) or os.path.islink(filePath):
                os.unlink(filePath)
            elif os.path.isdir(filePath):
                shutil.rmtree(filePath)
        except Exception as e:
            print(f"Failed to delete {filePath}. Reason: {e}")