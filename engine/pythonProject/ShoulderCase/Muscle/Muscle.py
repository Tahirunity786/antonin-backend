import os
import cv2
import glob
import pickle
import getConfig
from utils.Logger.Logger import Logger
import numpy as np
from utils.Vector.Vector import Vector
from utils.Sphere.Sphere import Sphere
from skimage.measure import label, regionprops
from ShoulderCase.MuscleMeasurer.MuscleMeasurer import MuscleMeasurer
from ShoulderCase.MaskSubdividerWithSquareGrid.MaskSubdividerWithSquareGrid import MaskSubdividerWithSquareGrid
from scipy.spatial.transform import Rotation as Rot
from utils.Plane.Plane import Plane

class Muscle:
    """
    The Muscle class is linked to a segmented muscle file (a mask)
    and to the slice it has been segmented out of.

    Then, this class can measured values linked to the PCSA and the muscle's degeneration.
    """
    def __init__(self, musclesContainer, muscleName):
        self.container = musclesContainer
        self.name = muscleName
        self.segmentationName = ""
        self.sliceName = ""
        self.PCSA = []
        self.atrophy = []
        self.fat = []
        self.osteochondroma = []
        self.degeneration = []
        self.centroid = []
        self.insertion = []
        self.forceAppilicationPoint = []
        self.forceVector = []
        self.subdivisions = []

        if not os.path.isdir(self.dataPath()):
            os.makedirs(self.dataPath())

        if not os.path.isdir(self.dataMaskPath()):
            os.makedirs(self.dataMaskPath())

    def dataPath(self):
        return os.path.join(self.container.dataPath(), self.name)

    def dataMaskPath(self):
        return os.path.join(self.dataPath(), "mask")

    def dataContourPath(self):
        return os.path.join(self.dataPath(), "contour")

    def setSliceName(self, value):
        self.sliceName = value

    def setSegmentationName(self, value):
        self.segmentationName = value

    def loadMask(self, maskSuffix):
        return cv2.imread(os.path.join(self.dataMaskPath(), self.segmentationName+"_"+maskSuffix+".png"),
                          cv2.IMREAD_UNCHANGED)

    def saveMask(self, mask, maskSuffix):
        cv2.imwrite(os.path.join(self.dataMaskPath(), self.segmentationName+"_"+maskSuffix+".png"), mask)

    def loadSlice(self, sliceSuffix):
        filePattern = os.path.join(self.container.dataSlicesPath, self.sliceName+"_"+sliceSuffix+"*")
        matchingFile = glob.glob(filePattern)
        assert matchingFile, f"File matching {filePattern} not found"
        assert len(matchingFile) <= 1, "Multiple matching files found. Include file extension in given suffix"

        if matchingFile.split(".")[-1] == ".pkl": #mat should be converted to .pkl
            with open(matchingFile[0], "rb") as f:
                loadedFile = pickle.load(f)
                ### to be continued
        elif matchingFile.split(".")[-1] == ".png":
            return cv2.imread(matchingFile[0])

    def measureFirst(self):
        """
        Call methods that can be run after morphology() methods has been run by all ShoulderCase objects.
        """
        self.setSliceName(getConfig.getConfig()["rotatorCuffSliceName"])
        self.setSegmentationName(getConfig.getConfig()["rotatorCuffSegmentationName"])
        subdivisionsResolutionX = getConfig.getConfig()["muscleSubdivisionsResolutionInMm"]["x"]
        subdivisionsResolutionY = getConfig.getConfig()["muscleSubdivisionsResolutionInMm"]["y"]
        success = Logger.timeLogExecution(
            self.name + f" subdivide segmentation {subdivisionsResolutionX} mm by {subdivisionsResolutionY} mm: ",
            lambda self: self.subdivide(subdivisionsResolutionX, subdivisionsResolutionY), self)
        success = success and Logger.timeLogExecution(
            self.name + " measure and save contour: ",
            lambda self: self.measureAndSaveContour(), self)
        success = success + Logger.timeLogExecution(
            self.name + " measure and save centroid: ",
            lambda self: self.measureAndSaveCentroid(), self)
        success = success + Logger.timeLogExecution(
            self.name + " PCSA: ",
            lambda self: self.measurePCSA(), self)
        success = success + Logger.timeLogExecution(
            self.name + " atrophy: ",
            lambda self: self.measureAtrophy(), self)
        success = success + Logger.timeLogExecution(
            self.name + " fat infiltration: ",
            lambda self: self.measureFatInfiltration(), self)
        success = success + Logger.timeLogExecution(
            self.name + " osteochondroma: ",
            lambda self: self.measureOsteochondroma(), self)
        success = success + Logger.timeLogExecution(
            self.name + " degeneration: ",
            lambda self: self.measureDegeneration(), self)
        success = success + Logger.timeLogExecution(
            self.name + " insertion: ",
            lambda self: self.measureInsertion(self.container.shoulder.humerus), self)
        success = success + Logger.timeLogExecution(
            self.name + " humeral head contact point: ",
            lambda self: self.measureHumerusContactPoint(), self)
        success = success + Logger.timeLogExecution(
            self.name + " force: ",
            lambda self: self.measureForceVector(), self)
        self.subdivisions = []
        return success

    def estimateInsertion(self):
        humerus = self.container.shoulder.humerus

        scapulaCoordSys = humerus.shoulder.scapula.coordSys
        humerusSuperior = humerus.center + humerus.radius * scapulaCoordSys.IS
        humerusAnterior = humerus.center + humerus.radius * scapulaCoordSys.PA
        humerusPosterior = humerus.center - humerus.radius * scapulaCoordSys.PA
        humerusLateral = humerus.center + humerus.radius * scapulaCoordSys.ML

        if self.getFullName() == "subscapularis":
            muscleInsertion = humerusAnterior
        elif self.getFullName() == "supraspinatus":
            muscleInsertion = humerusSuperior
        elif self.getFullName() == "infraspinatus":
            muscleInsertion = humerusPosterior
        elif self.getFullName() == "teres_minor":
            muscleInsertion = humerusPosterior
        else:
            muscleInsertion = humerusLateral

        return muscleInsertion

    def getClosestPointInSlice(self, pointCoordinates):
        slicePixelCoordinates = self.loadSlice("PixelCoordinates")

        distanceToPoint = np.srtq(
            (slicePixelCoordinates.x - pointCoordinates[0])**2 + \
            (slicePixelCoordinates.y - pointCoordinates[1])**2 + \
            (slicePixelCoordinates.z - pointCoordinates[2])**2
        )
        row, column = np.argmin(distanceToPoint) ## check the distanceToPoint.shape
        return row, column

    def getForceResultant(self):
        applicationPoint = np.mean(self.forceApplicationPoint, axis=0)
        forceResultant = Vector(applicationPoint, applicationPoint)
        for i in self.forceVector.shape[0]:
            forceResultant = forceResultant + self.forceVector[i]

        return forceResultant

    def getFullName(self):
        if self.name == "SC":
            return "subscapularis"
        elif self.name == "SS":
            return "supraspinatus"
        elif self.name == "IS":
            return "infraspinatus"
        elif self.name == "TM":
            return "teres_minor"

    def getInsertion(self):
        humerus = self.container.shoulder.humerus
        try:
            humerusLandmarks = humerus.getSlicerLandmarks()
        except:
            humerusLandmarks = []

        if "insertion_" + self.getFullName() in list(humerusLandmarks.keys()):
            muscleInsertion = humerusLandmarks["insertion_" + self.getFullName()]
        # if not segmented, teres minor insertion is set to the same point as infraspinatus insertion
        elif self.getFullName() == "teres_minor" and "insertion_infraspinatus" in list(humerusLandmarks.keys()):
            muscleInsertion = humerusLandmarks["insertion_infraspinatus"]
        # if not segmented, supraspinatus insertion is set to the articular top humeral head landmark
        elif self.getFullName() == "supraspinatus" and "articular_top" in list(humerusLandmarks.keys()):
            muscleInsertion = humerusLandmarks["articular_top"]
        else:
            muscleInsertion = self.estimateInsertion()

        humeralHead = Sphere(humerus.center, humerus.radius)
        lateralAxis = Vector(self.container.shoulder.scapula.coordSys.ML)
        muscleInsertion = lateralisePointOnSphere(muscleInsertion, humeralHead, lateralAxis, 30)

        return muscleInsertion

    def getMaskCoordinates(self, mask):
        maskIndices = np.where(mask > 0)
        slicePixelCoordinates = self.loadSlice("PixelCoordinates")
        maskCoordinates = np.array([slicePixelCoordinates.x[maskIndices],
                                    slicePixelCoordinates.y[maskIndices],
                                    slicePixelCoordinates.z[maskIndices]
                                   ])
        return maskCoordinates

    def measureAndSaveCentroid(self):
        self.centroid = []
        muscleSubdivisions = self.subdivisions
        allCentroidsMask = np.zeros(shape=(muscleSubdivisions.shape[:2]))
        for i in muscleSubdivisions.shape[2]:
            subdivisionCentroidMask = getSubdivisionCentroidMask(muscleSubdivisions[:,:, i])
            self.centroid = np.vstack((self.centroid, self.getMaskCoordinates(subdivisionCentroidMask)))
            # add current centroid to allCentroidsMask
            allCentroidsMask = np.logical_or(allCentroidsMask, subdivisionCentroidMask)
        self.saveMask(allCentroidsMask, "Centroid")

    def measureAndSaveContour(self):
        muscleSegmentation = self.subdivisions
        muscleContours = np.zeros(shape=muscleSegmentation.shape[:2])
        for i in range(muscleSegmentation.shape[2]):
            muscleContours = np.logical_or(muscleContours, muscleSegmentation[:,:,i])
        self.saveMask(muscleContours, "Contour")

    def measureAtrophy(self):
        self.atrophy = []
        for i in self.subdivisions.shape[2]:
            measurer = MuscleMeasurer(self.subdivisions[:, :, i],
                                      self.loadSlice("ForMeasurements"),
                                      self.loadSlice("PixelSpacings"))
            self.atrophy = np.vstack((self.atrophy, measurer.getRatioAtrophy()))

    def measureDegeneration(self):
        self.degeneration = []
        for i in range(self.subdivisions.shape[2]):
            measurer = MuscleMeasurer(self.subdivisions[:,:,i],
                                      self.loadSlice("ForMeasurements"),
                                      self.loadSlice("PixelSpacings"))
            self.degeneration = np.vstack((self.degeneration, measurer.getRatioDegeneration()))

    def measureFatInfiltration(self):
        self.fat = []
        for i in range(self.subdivisions.shape[2]):
            measurer = MuscleMeasurer(self.subdivisions[:, :, i],
                                      self.loadSlice("ForMeasurements"),
                                      self.loadSlice("PixelSpacings"))
            self.degeneration = np.vstack((self.fat, measurer.getRatioFat()))

    def measureForceVector(self):
        self.forceVector = []
        for i in range(self.subdivisions.shape[2]):
            subdivisionForce = Vector(self.forceAppilicationPoint[i,:], self.centroid[i,:])
            subdivisionForce = (subdivisionForce / subdivisionForce.norm()) * self.PCSA[i] * (1 - self.degeneration[i])
            self.forceVector = np.vstack((self.forceVector, subdivisionForce))

    def measureHumerusContactPoint(self):
        humerus = self.container.shoulder.humerus
        humeralHeadContactPointFinder = SphereContactPoint(Sphere(humerus.center, humerus.radius))

        self.forceAppilicationPoint = []
        # find one force application point for each subdivision's centroid
        for i in range(self.centroid.shape[0]):
            humeralHeadContactPointFinder.setAnchorPoint(self.insertion[i,:])
            humeralHeadContactPointFinder.setPolarPoint(self.centroid[i,:])
            if humeralHeadContactPointFinder.sphereIsBetweenAnchorAndPolarPoints():
                forceApplicationPoint = humeralHeadContactPointFinder.getContactPoint()
            else:
                forceApplicationPoint = self.insertion[i,:]
            self.forceApplicationPoint = np.vstack((self.forceAppilicationPoint, forceApplicationPoint))

    def measureInsertion(self, humerus):
        """
        Find and return the point for which the sidemost muscle's segmentation
        centroids are coplanar with muscle's insertion extremities and humerus center.
        This unique point insertion equivalent can then be used with existing
        methods to compute the muscle's fibres contact points.
        """
        insertionsExtremities = humerus.getInsertionsExtremitiesByMuscle[self.getFullName()].T
        insertionsExtremities = projectOnInsertionsRing(insertionsExtremities, humerus.insertionsRing)
        insertionDirection = Vector(insertionsExtremities[0,:], insertionsExtremities[1,:])

        centeroidCenter = np.mean(self.centroid)
        centroidsPositionAlongInsertion = getCentroidsPositionAlongInsertion(self.centroid, insertionDirection)

        insertion = []
        for i in range(self.centroid.shape[0]):
            insertion[i, :] = getFibreInsertion(centroidsPositionAlongInsertion[i],
                                                insertionsExtremities,
                                                humerus.insertionsRing)
        self.insertion = insertion

    def measureOsteochondroma(self):
        self.osteochondroma = []
        for i in range(self.subdivisions.shape[2]):
            measurer = MuscleMeasurer(self.subdivisions[:, :, i],
                                       self.loadSlice("ForMeasurements"),
                                       self.loadSlice("PixelSpacings"))
            self.osteochondroma = self.osteochondroma.append(measurer.getRatioOsteochondroma())
        self.osteochondroma = np.array(self.osteochondroma)

    def measurePCSA(self):
        self.PCSA = []
        for i in range(self.subdivisions.shape[2]):
            measurer = MuscleMeasurer(self.subdivisions[:, :, i],
                                      self.loadSlice("ForMeasurements"),
                                      self.loadSlice("PixelSpacings"))
            self.PCSA = self.PCSA .append(measurer.getPCSA())
        self.PCSA = np.array(self.PCSA)

    def measureUniquePointInsertionEquivalent(self, humerus):

        """
        Find and return the point for which the sidemost muscle's segmentation
        centroids are coplanar with muscle's insertion endpoints and humerus center.
        This unique point insertion equivalent can then be used with existing
        methods to compute the muscle's fibres contact points.
        """

        if self.centroid.shape[0] < 2:
            self.uniquePointInsertionEquivalent = getUniqueCentroidInsertionEquivalent(self, humerus)

        insertionEndpoints = humerus.landmarks.insertions.endpoints[self.getFullName()].T
        insertionDirection = Vector(insertionEndpoints[0, :], insertionEndpoints[1, :])

        sidemostCentroids = getSidemostCentroids(self.centroid, insertionDirection)

        sidemostPlanesNormal = []
        for i in range(sidemostCentroids.shape[0]):
            sidemostPlane = Plane()
            sidemostPlane.fit(np.vstack((humerus.center[None, :], sidemostCentroids[i,:], insertionEndpoints[i,:])))

            sidemostPlanesNormal = np.vstack((sidemostPlanesNormal, sidemostPlane.normal()))

        sidemostPlanesIntersection = Vector(np.cross(sidemostPlanesNormal[0, :], sidemostPlanesNormal[1, :]))

        humerusCenterToInsertionEndpoints = Vector(humerus.center, np.mean(insertionEndpoints, axis=0))

        humerusCenterToInsertionEquivalent = sidemostPlanesIntersection.orientToward(humerusCenterToInsertionEndpoints)

        insertionEquivalent = humerus.center + humerus.radius * humerusCenterToInsertionEquivalent.direction

        self.uniquePointInsertionEquivalent = insertionEquivalent

    def subdivide(self, xResolutionInMm, yResolutionInMm):
        subdivider = MaskSubdividerWithSquareGrid(self.loadMask("Segmentation"))
        slicePixelSpacings = self.loadSlice("PixelSpacings")
        subdivider.setXResolutionInMm(xResolutionInMm, slicePixelSpacings[0])
        subdivider.setYResolutionInMm(yResolutionInMm, slicePixelSpacings[1])
        self.subdivisions = subdivider.getMaskSubdivisions()


def lateralisePointOnSphere(point, sphere, lateralAxis, angle):
    centerToPoint = Vector(sphere.center, point)
    sphereSliceNormal = centerToPoint.cross(lateralAxis).normalised()
    complementaryAxis = lateralAxis.cross(sphereSliceNormal).normalised()
    lateralisedPoint = sphere.center + \
                       (sphere.radius * np.cos(np.degrees(angle)) * lateralAxis.normalised().direction()) + \
                       (sphere.radius * np.sin(np.degrees(angle)) * complementaryAxis.direction())

    return lateralisedPoint

def getSubdivisionCentroidMask(subdivision):
    subdivisionCentroidMask = np.zeros(shape=subdivision.shape[:2])
    subdivisionCentroidIndices = regionprops(label(subdivision))[0].centroid
    subdivisionCentroidMask[subdivisionCentroidIndices[0], subdivisionCentroidIndices[1]] = True
    return subdivisionCentroidMask

def bwmorph(input_matrix):
    output_matrix = input_matrix.copy()
    # Change. Ensure single channel
    if len(output_matrix.shape) == 3:
        output_matrix = output_matrix[:, :, 0]
    nRows,nCols = output_matrix.shape # Change
    orig = output_matrix.copy() # Need another one for checking
    for indexRow in range(0,nRows):
        for indexCol in range(0,nCols):
            center_pixel = [indexRow,indexCol]
            neighbor_array = neighbors(orig, center_pixel) # Change to use unmodified image
            if np.all(neighbor_array): # Change
                output_matrix[indexRow,indexCol] = 0

    return output_matrix

def neighbors(input_matrix,input_array):
    (rows, cols) = input_matrix.shape[:2] # New
    indexRow = input_array[0]
    indexCol = input_array[1]
    output_array = [0] * 4 # New - I like pre-allocating

    # Edit
    output_array[0] = input_matrix[(indexRow - 1) % rows,indexCol]
    output_array[1] = input_matrix[indexRow,(indexCol + 1) % cols]
    output_array[2] = input_matrix[(indexRow + 1) % rows,indexCol]
    output_array[3] = input_matrix[indexRow,(indexCol - 1) % cols]
    return output_array

def getCentroidsPositionAlongInsertion(centroids, insertionDirection):
    if centroids.shape[0] == 1:
        return 0.5
    centroidCenter = np.mean(centroids, axis=0)
    centroidsPositionAlongInsertion = []
    for i in range(centroids.shape[0]):
        centroid = Vector(centroidCenter, centroids[i, :])
        centroidsPositionAlongInsertion[i] = insertionDirection.dot(centroid)
    centroidsPositionAlongInsertion = np.array(centroidsPositionAlongInsertion)
    normalisedCentroidsPositionAlongInsertion = (centroidsPositionAlongInsertion- min(centroidsPositionAlongInsertion))/ \
                                                (np.max(centroidsPositionAlongInsertion)- np.min(centroidsPositionAlongInsertion))
    return normalisedCentroidsPositionAlongInsertion

def getFibreInsertion(fibreRelativePosition, insertionsExtremities, insertionsRing):
    insertionRingCenterToInsertionExtremities = [Vector(insertionsRing.center, insertionsExtremities[0,:]),
                                                 Vector(insertionsRing.center, insertionsExtremities[1,:])]
    insertionApertureAngle = insertionRingCenterToInsertionExtremities[1].angle(insertionRingCenterToInsertionExtremities[0])

    fibreApertureAngle = fibreRelativePosition * insertionApertureAngle
    insertionPlaneNormal = insertionRingCenterToInsertionExtremities[1].cross(insertionRingCenterToInsertionExtremities[0]).direction()

    rotationVector = Rot.from_rotvec(fibreApertureAngle @ insertionPlaneNormal)
    quaternion = rotationVector.as_quat()
    # You need to change the order of the values as the function quatrotate works with w + x + y + z notation for
    # quaternion such as matlab which is not the case in scipy
    quaternion = np.array([quaternion[3], quaternion[0], quaternion[1], quaternion[2]])
    humerusCenterToFibreInsertion = quatrotate(quaternion, insertionRingCenterToInsertionExtremities[0].vector())
    fibreInsertion = insertionsRing.center + humerusCenterToFibreInsertion
    return fibreInsertion

def quatrotate(Q, V):

    # Equivalen to quatrotate MATLAB
    # 1x4 Q
    # 1x3 V

    q0 = Q[0]
    q1 = Q[1]
    q2 = Q[2]
    q3 = Q[3]

    q11 = 1 - 2*q2**2 - 2*q3**2
    q12 = 2*((q1*q2)+(q0*q3))
    q13 = 2*((q1*q3)-(q0*q2))
    q21 = 2*((q1*q2)-q0*q3)
    q22 = 1 - 2*q1**2 - 2*q3**2
    q23 = 2*((q2*q3)+(q0*q1))
    q31 = 2*((q1*q3)+(q0*q2))
    q32 = 2*((q2*q3)-(q0*q1))
    q33 = 1 - 2*q1**2 - 2*q2**2


    DCM = np.array([[q11, q12, q13],
                    [q21, q22, q23],
                    [q31, q32, q33]])

    V = np.matmul(DCM, V.transpose())
    return V

def projectOnInsertionsRing(points, insertionsRing):
    projectedPoints = np.zeros(points.shape)
    for i in range(points.shape[0]):
        ringCenterToPoint = Vector(insertionsRing.center, points[i, :])
        projectedPoints[i, :] = insertionsRing.center + insertionsRing.radius * ringCenterToPoint.direction

    return projectedPoints

def getUniqueCentroidInsertionEquivalent(muscle, humerus):
    # In this case the unique point insertion equivalent is chosen to be in the middle of insertion endpoints
    humerusCenterToInsertionEquivalent = Vector(humerus.center,
                                                np.mean(humerus.landmarks.insertions.endpoints[muscle.getFullName()]), axis=0)
    return humerus.center + humerus.radius * humerusCenterToInsertionEquivalent.direction

def getSidemostCentroids(centroids, insertionDirection):
    # This algorithm is sufficient to determine the sidemost centroids given that all the centroids are coplanar
    centroidsPositionAlongInsertion = []
    for i in range(centroids.shape[0]):
        centroidsPositionAlongInsertion.append(np.dot(insertionDirection, centroids[i, :]))

    centroidsPositionAlongInsertion = np.array(centroidsPositionAlongInsertion)

    sidemostCentroids = []
    sidemostCentroids.append(centroids[np.where(centroidsPositionAlongInsertion == np.min(centroidsPositionAlongInsertion)), :])
    sidemostCentroids.append(centroids[np.where(centroidsPositionAlongInsertion == np.max(centroidsPositionAlongInsertion)), :])

    sidemostCentroids = np.array(sidemostCentroids)

    return sidemostCentroids





















