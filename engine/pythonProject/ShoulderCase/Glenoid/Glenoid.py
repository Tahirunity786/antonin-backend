import numpy as np
from utils.Sphere.Sphere import Sphere
from utils.Logger.Logger import Logger
from sklearn.decomposition import PCA
from utils.Plane.Plane import Plane
from utils.Rotations import rotation_angle
from utils.Vector.Vector import Vector
from ShoulderCase.fitLine import fitLine
from ShoulderCase.findShortest3DVector import findShortest3DVector
from ShoulderCase.orientVectorToward import orientVectorToward
import os
from utils.Timer.Timer import Timer
from ShoulderCase.DicomVolume.readDicomVolume import readDicomVolume
from pydicom import dcmread
from scipy.ndimage import gaussian_filter
from scipy.ndimage import binary_fill_holes
from skimage.morphology import remove_small_objects
from skimage.morphology import disk
from skimage.morphology import erosion
from ShoulderCase.projectVectorArrayOnVector import projectVectorArrayOnVector
from ShoulderCase.findLongest3DVector import findLongest3DVector
import pandas as pd
from getConfig import getConfig
import plotly.figure_factory as ff
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.offline import iplot
#import tensorflow as tf
#import tensorflow_addons as tfa
from collections import namedtuple
from ShoulderCase.project2Plane import project2Plane


class Glenoid:
    """
    Need to load surface points from Amira (or auto-segment)
    Need to set anatomical calculation, and use the scapular coord.Syst
    """
    def __init__(self, scapula):
        self.surface = {"points":[], "faces":[], "meanPoint":[]} # points of the glenoid surface load glenoid surface
        self.center = np.array([]) # Glenoid center in CT coordinate system
        self.centerLocal = {} # Glenoid center in scapula coordinate system
        self.radius = np.array([])
        self.centerLine = np.array([])
        self.posteroAnteriorLine = np.array([])
        self.inferoSuperiorLine = np.array([])
        self.depth = np.array([])
        self.width = np.array([])
        self.height = np.array([])
        self.anteroSuperiorAngle = np.array([])
        self.versionAmplitude = np.array([])
        self.versionOrientation = np.array([])
        self.version = np.array([])
        self.inclination = np.array([])
        self.retroversion = np.array([])
        self.rimInclination = np.array([])
        self.beta = np.array([])
        self.GPA = np.array([])
        self.density = np.array([])
        self.comment = []
        self.walch = np.array([])
        self.fittedSphere = Sphere()
        self.scapula = scapula

    def loadData(self):
        """
        Call methods that can be run after the ShoulderCase object has
        been instanciated.
        """
        if self.scapula.isempty():
            return False
        success = Logger().timeLogExecution(
             "Glenoid load surface: ",
             lambda self: self.loadSurface(), self)
        success = success and Logger().timeLogExecution(
             "Glenoid load Walch class: ",
             lambda self: self.readWalchData(), self)

        return success

    def morphology(self):
        """
        Call methods that can be run after loadData() methods has been run by
        all ShoulderCase objects.
        """
        success = Logger().timeLogExecution(
             "Glenoid sphere fitting: ",
             lambda self: self.fittedSphere.fitTo(self.surface["points"]), self)
        success = success and Logger().timeLogExecution(
             "Glenoid center: ",
             lambda self: self.measureCenter(), self)
        success = success and Logger().timeLogExecution(
             "Glenoid radius: ",
             lambda self: self.measureRadius(), self)
        return success

    def measureFirst(self):
        """
        Call methods that can be run after morphology() methods has been run by
        all ShoulderCase objects.
        """
        success = Logger().timeLogExecution(
             "Glenoid center local: ",
             lambda self: self.measureCenterLocal(), self)
        success = success and Logger().timeLogExecution(
             "Glenoid center line: ",
             lambda self: self.measureCenterLine(), self)
        success = success and Logger().timeLogExecution(
             "Glenoid depth: ",
             lambda self: self.measureDepth(), self)
        success = success and Logger().timeLogExecution(
             "Glenoid width and height: ",
             lambda self: self.measureWidthAndHeight(), self)
        success = success and Logger().timeLogExecution(
             "Glenoid version and inclination: ",
             lambda self: self.measureVersionAndInclination(), self)
        success = success and Logger().timeLogExecution(
             "Glenoid antero-superior angle: ",
             lambda self: self.measureAnteroSuperiorAngle(), self)
        success = success and Logger().timeLogExecution(
             "Glenoid infero-superior line: ",
             lambda self: self.measureInferoSuperiorLine(), self)
        success = success and Logger().timeLogExecution(
             "Glenoid beta angle: ",
             lambda self: self.measureBetaAngle(), self)
        success = success and Logger().timeLogExecution(
             "Glenoid postero-anterior line: ",
             lambda self: self.measurePosteroAnteriorLine(), self)
        success = success and Logger().timeLogExecution(
             "Glenoid rim inclination: ",
             lambda self: self.measureRimInclination(), self)
        success = success and Logger().timeLogExecution(
             "Glenoid GlenoPoral Angle: ",
             lambda self: self.measureGlenoPolarAngle(), self)
        return success

    def measureSecond(self):
        """
        Call methods that can be run after measureFirst() methods has been run
        by all ShoulderCase objects.
        """
        success = Logger().timeLogExecution(
             "Glenoid retroversion: ",
             lambda self: self.measureRetroversion(), self)
        return success

    def getAnatomicalExtremeRimPoints(self):
        # Return the points of the extreme points of the glenoid rim according to the
        # scapula coordinate system.
        # There might be several points returned for a given direction, most likely for
        # the medio-lateral direction.
        rimPoints = self.getRimPoints()
        coordSys = self.scapula.coordSys
        return {"lateral":selectPointsFromDotProductWithAxis(rimPoints,coordSys.ML, "max"),
                "medial":selectPointsFromDotProductWithAxis(rimPoints,coordSys.ML, "min"),
                "inferior":selectPointsFromDotProductWithAxis(rimPoints,coordSys.IS, "min"),
                "superior":selectPointsFromDotProductWithAxis(rimPoints,coordSys.IS, "max"),
                "posterior":selectPointsFromDotProductWithAxis(rimPoints,coordSys.PA, "min"),
                "anterior":selectPointsFromDotProductWithAxis(rimPoints,coordSys.PA, "max")
                }

    def getNewPCAAxis(self):
        print("SCaseID:", self.scapula.shoulder.SCase.id)
        pca = PCA()
        pcaf = pca.fit(self.surface["points"])
        pca_comp = pcaf.components_
        pcaCoeff = np.concatenate([pca_comp.T[:, 0].reshape(-1, 1),
                                   (-pca_comp.T[:, 1]).reshape(-1, 1),
                                   pca_comp.T[:, 2].reshape(-1, 1)], axis=1)
        scapulaAxes = self.scapula.coordSys.getRotationMatrix()

        heightWidthDepthScapulaAxes = scapulaAxes[:, (1, 0, 2)]

        pcaToScapulaCorrelation = np.abs(pcaCoeff.T@heightWidthDepthScapulaAxes)

        newOrder = [0]*pcaToScapulaCorrelation.shape[1]

        while np.any(pcaToScapulaCorrelation > -1000):
            row, col = np.argmax(np.max(pcaToScapulaCorrelation, axis=1)),\
                       np.argmax(np.max(pcaToScapulaCorrelation, axis=0))
            pcaToScapulaCorrelation[row, :] = -10000
            pcaToScapulaCorrelation[:, col] = -10000
            newOrder[col] = row
        return pcaCoeff[:, newOrder]

    def getRimPoints(self):
        return self.surface["points"][self.getRimPointsIndices(), :]

    def getRimPointsIndices(self):
        """
        Return the indices of the points on the rim of the glenoid surface.
        To find the rim points, this function analyses the faces of the glenoid.surface
        structure. These faces are triangles, each triangle is defined by three points
        of the glenoid's surface. The points on the rim are defined to be the points
        that appear in the faces 4 times or less (usually 3 or 4 times). Consequently,
        the other points that appear more than 4 times in the faces (usually 5 or 6
        times) are expected to be the inner points of the surface.

        WARNING: This is an empirical method to define which points are the rim points.
        It is not based on the glenoid's surface tesselation algorithm and might
        return false positive and false negative points.
        For example, P200 shoulderAuto rim points found with the current function
        feature several 4-occurence points that are inner points of the surface. There
        are also 5-occurence rim points that are not detected.
        Fix hint to try: Run a new triangulation algorithm on the glenoid points (
        like the matlab's delaunay() funtion).
        """
        trianglePointsIndices = self.surface["faces"]
        pointsOccurence = np.zeros_like(np.unique(trianglePointsIndices))
        trianglePointsIndices1D = list(trianglePointsIndices.ravel())
        for ind in list(np.unique(trianglePointsIndices)):
            pointsOccurence[ind] = trianglePointsIndices1D.count(ind)
        return pointsOccurence <= 4

    def measureAnteroSuperiorAngle(self):
        """
        Glenoid inclinaison angle between yAxis of scapula and axis
        of height of glenoid

        Projection of height axis on IS and PA axis
        """
        NewPCAAxis = self.getNewPCAAxis()

        XYPlane = Plane()

        XYPlane.fit(np.concatenate([np.array([0, 0, 0]).reshape(1, -1),
                        self.scapula.coordSys._xAxis.reshape(1, -1),
                        self.scapula.coordSys._yAxis.reshape(1, -1)]))
        projectedPCAAxis = XYPlane.projectOnPlane(NewPCAAxis[:,0].reshape(1, -1))
        #anteroSuperiorRotation = rotation_angle.rotation_matrix_from_vectors(projectedPCAAxis,
        #                                                      self.scapula.coordSys._yAxis)
        #self.anteroSuperiorAngle = rotation_angle.angle_of_rotation(anteroSuperiorRotation)
        self.anteroSuperiorAngle = rotation_angle.angle_of_rotation_from_vectors(projectedPCAAxis,
                                                                                 self.scapula.coordSys._yAxis)

    def measureBetaAngle(self):
        """
        Glenoid BETA angle is the angle between the floor of the supraspinatus fossa
        marked by a sclerotic line and the glenoid fossa line. The floor of the
        supraspinatus fossa is the line fitted to the scapular groove points.
        The glenoid fossa line is the glenoid supero-inferior line.
        """
        supraspinatusFossaFloor = Vector(fitLine(self.scapula.groove)[0])
        supraspinatusFossaFloor = supraspinatusFossaFloor.orientToward(self.scapula.coordSys.ML)
        betaRotation = rotation_angle.rotation_matrix_from_vectors(supraspinatusFossaFloor.vector(),
                                                                   -self.inferoSuperiorLine.vector())

        # Unlike glenoid retroversion angle, glenoid beta angle method doesn't set
        # the sign of this angle which should be compared to the rotation axis alignment
        # with scapular PA axis. However, this alignement depends on the shoulder side.
        # It's assumed that glenoid supero-inferior line will always be aligned with
        # scapular SI axis, and supraspinatusFossaFloor line will always be aligned
        # with scapular ML axis. Therefore glenoid beta angle will always be a
        # positive number.
        #self.beta = rotation_angle.angle_of_rotation(betaRotation)
        self.beta = rotation_angle.angle_of_rotation_from_vectors(supraspinatusFossaFloor.vector(),
                                                                  -self.inferoSuperiorLine.vector())

    def measureCenterOld(self):
        """
        Glenoid center is the point of the segmented glenoid's surface which is the
        closest to the mean of all the points of the segmented glenoid's surface
        """
        surfacePointsFromMeanPoint = self.surface["points"] - self.surface["meanPoint"]
        centerIndex = findShortest3DVector(surfacePointsFromMeanPoint)
        self.center = self.surface["points"][centerIndex,:]

    def measureCenter(self):
        """
        Glenoid center is the point of the segmented glenoid's surface which is the
        closest to the mean of all the points of the segmented glenoid's surface

        Change by Osman Satir: It is calculated by taking the average of 3 points for each face
        and weighting this average by it's surface area. The final point is calculated by summing
        each face average*its surface area and dividing this sum by the total area. This takes into
        account the fact that the glenoid surface mesh may not be homogeneous.
        """

        face_means = []
        surface_areas = []
        for face in self.surface["faces"]:
            point1 = self.surface["points"][face[0]]
            point2 = self.surface["points"][face[1]]
            point3 = self.surface["points"][face[2]]
            face_mean = np.mean([point1, point2, point3], axis=0)

            x0 = point1[0]
            y0 = point1[1]
            z0 = point1[2]

            x1 = point2[0]
            y1 = point2[1]
            z1 = point2[2]

            x2 = point3[0]
            y2 = point3[1]
            z2 = point3[2]
            surface_area = np.sqrt(((np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2 + (z1 - z0) ** 2) + np.sqrt(
                (x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2) + np.sqrt(
                (x2 - x0) ** 2 + (y2 - y0) ** 2 + (z2 - z0) ** 2)) / 2) * (((np.sqrt(
                (x1 - x0) ** 2 + (y1 - y0) ** 2 + (z1 - z0) ** 2) + np.sqrt(
                (x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2) + np.sqrt(
                (x2 - x0) ** 2 + (y2 - y0) ** 2 + (z2 - z0) ** 2)) / 2) - np.sqrt(
                (x1 - x0) ** 2 + (y1 - y0) ** 2 + (z1 - z0) ** 2)) * (((np.sqrt(
                (x1 - x0) ** 2 + (y1 - y0) ** 2 + (z1 - z0) ** 2) + np.sqrt(
                (x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2) + np.sqrt(
                (x2 - x0) ** 2 + (y2 - y0) ** 2 + (z2 - z0) ** 2)) / 2) - np.sqrt(
                (x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)) * (((np.sqrt(
                (x1 - x0) ** 2 + (y1 - y0) ** 2 + (z1 - z0) ** 2) + np.sqrt(
                (x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2) + np.sqrt(
                (x2 - x0) ** 2 + (y2 - y0) ** 2 + (z2 - z0) ** 2)) / 2) - np.sqrt(
                (x2 - x0) ** 2 + (y2 - y0) ** 2 + (z2 - z0) ** 2)))

            surface_areas.append(surface_area)
            face_means.append(face_mean)

        weighted_sum = np.sum(np.multiply(np.array(face_means), np.array(surface_areas).reshape(len(surface_areas), 1)),
                              axis=0)

        self.center = weighted_sum / np.sum(surface_areas)

    def measureCenterLine(self):
        self.centerLine = self.fittedSphere.center.ravel() - self.center
        self.centerLine = orientVectorToward(self.centerLine, self.scapula.coordSys._zAxis)

    def measureCenterLocal(self):
        centerLocal = self.scapula.coordSys.express(self.center.reshape(1, -1))
        self.centerLocal["x"] = centerLocal[0, 0]
        self.centerLocal["y"] = centerLocal[0, 1]
        self.centerLocal["z"] = centerLocal[0, 2]
        
    def measureDepth(self):
        surfacePointsFromGlenoidCenter = self.surface["points"] - self.center
        surfacePointsProjectedOnCenterLine= projectVectorArrayOnVector(surfacePointsFromGlenoidCenter,
                                                                       self.centerLine)
        deepestPointIndex = findLongest3DVector(surfacePointsProjectedOnCenterLine)
        self.depth = np.linalg.norm(surfacePointsProjectedOnCenterLine[deepestPointIndex,:])

    def measureInferoSuperiorLine(self):
        rimPoints = self.getAnatomicalExtremeRimPoints()
        self.inferoSuperiorLine = Vector(rimPoints["inferior"], rimPoints["superior"])

    def measurePosteroAnteriorLine(self):
        rimPoints = self.getAnatomicalExtremeRimPoints()
        self.posteroAnteriorLine = Vector(rimPoints["posterior"], rimPoints["anterior"])

    def measureRadius(self):
        self.radius = self.fittedSphere.radius

    def measureRetroversion(self):
        """
        Glenoid retroversion is the angle between the glenoid postero-anterior line
        and the line perpendicular to the Friedman's line that goes through the most
        anterior glenoid rim point.
        obj.measurePosteroAnteriorLine() and obj.scapula.measureFriedmansLine() must
        have been run before measuring the glenoid retroversion.
        """
        FL = self.scapula.friedmansLine
        anteriorRimPoint = self.getAnatomicalExtremeRimPoints()["anterior"]
        FLToAnteriorRimPoint = FL.orthogonalComplementTo(anteriorRimPoint)

        # Lines projection on scapular axial plane.
        IS = Vector(np.array([0, 0, 0]), self.scapula.coordSys.IS)
        FLToAnteriorRimPointXZ = IS.orthogonalComplementTo(FLToAnteriorRimPoint.vector())
        glenoidPosteroAnteriorLineXZ = IS.orthogonalComplementTo(self.posteroAnteriorLine.vector())
        retroversionRotation = rotation_angle.rotation_matrix_from_vectors(FLToAnteriorRimPointXZ.vector(),
                                                                           glenoidPosteroAnteriorLineXZ.vector())

        #self.retroversion = -np.sign(np.dot(IS.vector(),rotation_angle.axis_of_rotation(retroversionRotation)))\
        #                             *rotation_angle.angle_of_rotation(retroversionRotation)
        retroversionRotationAxis = rotation_angle.axis_of_rotation(retroversionRotation)
        retroversionRotationAngle = rotation_angle.angle_of_rotation_from_vectors(FLToAnteriorRimPointXZ.vector(),
                                                                                  glenoidPosteroAnteriorLineXZ.vector())
        self.retroversion = -np.sign(np.dot(IS.vector(), retroversionRotationAxis))*retroversionRotationAngle

    def measureRimInclination(self):
        """
        Glenoid rim inclination is the angle between the glenoid infero-superior line
        and the line that goes through Trigonum Spinae and Spino-Glenoid notch.

        obj.measureInferoSuperiorLine() must have been run before measuring the
        glenoid retroversion.
        """
        TStoSG = Vector(self.scapula.trigonumSpinae.ravel(), self.scapula.spinoGlenoidNotch.ravel())
        rimInclinationRotation = rotation_angle.rotation_matrix_from_vectors(self.inferoSuperiorLine.vector(),
                                                                             TStoSG.vector())
        # Unlike glenoid retroversion angle, glenoid rim inclination method doesn't set
        # the sign of this angle which should be compared to the rotation axis alignment
        # with scapular PA axis. However, this alignement depends on the shoulder side.
        # It's assumed that glenoid infero-superior line will always be aligned with
        # scapular IS axis, and TStoSG line will always be aligned with scapular ML
        # axis. Therefore glenoid rim inclination angle will always be a positive number.
        #self.rimInclination = rotation_angle.angle_of_rotation(rimInclinationRotation)
        self.rimInclination = rotation_angle.angle_of_rotation_from_vectors(self.inferoSuperiorLine.vector(),
                                                                            TStoSG.vector())

    def measureVersionAndInclination(self):
        """
        Glenoid version 3D, defined by the amplitude and orientation of the angle
        between the glenoid centerline and the scapular transverse axis (z axis).
        The orientation is the angle between the -x axis and the glenoid
        centerline projected in the xy plane. Zero orientation correspond to -x
        axis (posterior side), 90 to superior, 180 to anterior orientaion, and
        -90 to inferior. Glenoid version is also reported as version2D (>0 for
        retroversion) and inclination (>0 for superior tilt).
        """
        absoluteCenterLine = self.centerLine + self.scapula.coordSys.origin
        localCenterLine = self.scapula.coordSys.express(absoluteCenterLine.reshape(1, -1))
        X = np.array([1, 0, 0])
        Y = np.array([0, 1, 0])
        Z = np.array([0, 0, 1])

        versionAmplitudeRotation = vrrotvec(Z,localCenterLine.ravel()) # angle between centerLine and scapular medio-lateral axis
        self.versionAmplitude = versionAmplitudeRotation.rotation_angle

        XYProjectedCenterLine = self.scapula.coordSys.projectOnXYPlane(absoluteCenterLine)
        localXYProjectedCenterLine = self.scapula.coordSys.express(XYProjectedCenterLine.reshape(1, -1))
        versionOrientationRotation = vrrotvec(-X,localXYProjectedCenterLine.ravel())
        self.versionOrientation = np.sign(localXYProjectedCenterLine.reshape(1, -1)@Y.reshape(-1, 1))\
                                  * versionOrientationRotation.rotation_angle
        self.versionOrientation = self.versionOrientation[0, 0]

        ZXProjectedCenterLine = self.scapula.coordSys.projectOnZXPlane(absoluteCenterLine)
        localZXProjectedCenterLine = self.scapula.coordSys.express(ZXProjectedCenterLine.reshape(1, -1))
        versionRotation = vrrotvec(Z,localZXProjectedCenterLine.ravel())
        self.version = np.sign(localZXProjectedCenterLine.reshape(1, -1)@X.reshape(-1, 1)) \
                       * versionRotation.rotation_angle
        self.version = self.version[0,0]

        YZProjectedCenterLine = self.scapula.coordSys.projectOnYZPlane(absoluteCenterLine)
        localYZProjectedCenterLine = self.scapula.coordSys.express(YZProjectedCenterLine.reshape(1, -1))
        inclinationRotation = vrrotvec(Z,localYZProjectedCenterLine.ravel())
        self.inclination = np.sign(localYZProjectedCenterLine.reshape(1, -1)@Y.reshape(-1, 1)) \
                           * inclinationRotation.rotation_angle
        self.inclination = self.inclination[0, 0]


    def measureWidthAndHeight(self):
        NewPCAAxis = self.getNewPCAAxis()
        PCAGlenSurf = (np.linalg.inv(NewPCAAxis) @ self.surface["points"].T).T

        self.height = np.max(PCAGlenSurf[:,0])-np.min(PCAGlenSurf[:,0])
        self.width = np.max(PCAGlenSurf[:,1])-np.min(PCAGlenSurf[:,1])

    def readWalchData(self):
        filename = os.path.join(getConfig()["dataDir"], "Excel", "ShoulderDataBase.xlsx")
        caseMetadata = pd.read_excel(filename, sheet_name="SCase")
        walchClass = caseMetadata[(caseMetadata.SCase_ID == self.scapula.shoulder.SCase.id) | \
                     (caseMetadata.shoulder_side == self.scapula.shoulder.side)].glenoid_walchClass.values[0]
        if isinstance(walchClass, str):
            self.walch = walchClass
    
    def measureGlenoPolarAngle(self):
        # Get local coordinate system from parent scapula
        origin = self.scapula.coordSys.origin
        # Glenoid surface in scapular coordinate system
        glenSurf = self.scapula.coordSys.express(self.scapula.glenoid.surface["points"])
        # Glenoid center in scapular coordinate system
        glenCenter = self.scapula.coordSys.express(self.center[None, :])
        ScapPlaneNormal = np.array([1, 0, 0])  # Normal of the scapular plane in the scapular coordinate system
        PlaneMean = np.array([0, 0, 0])  # Origin of scapular system in scapular coordinate system
        # Project glenoid surface in scapular plane
        glenSurf = project2Plane(glenSurf, ScapPlaneNormal, PlaneMean)
        glenPrinAxis = np.concatenate([glenSurf[np.where(glenSurf[:, 1] == np.min(glenSurf[:, 1])), :],
                                       glenSurf[np.where(glenSurf[:, 1] == np.max(glenSurf[:, 1])), :]],
                                      axis=0).squeeze()
        # Most inferior point of the glenoid surface
        IG = glenPrinAxis[0, :]
        # Most superior point of the glenoid surface
        SG = glenPrinAxis[1, :]
        IGSG = SG - IG

        localAI = self.scapula.coordSys.express(self.scapula.angulusInferior)
        localAI[0,0] = 0
        AItoGlenoidSuperiorPoint = Vector(localAI, SG).vector()
        GlenoPolarAngle = rotation_angle.angle_of_rotation_from_vectors(IGSG, AItoGlenoidSuperiorPoint.flatten())
        self.GPA = GlenoPolarAngle
        
    def measureDensityMain(self, *args):
        """
        Calculate the density of the glenoid bone in 6 volumes of interest
        """
        # This function was developped by Paul Cimadomo during a semester  project (Fall 2018-19)
        # and implemented by Alexandre Terrier (2019-02-03)
        # % sCase --> SCase
        # obj.cylinder removed
        # obj.volume removed
        # TRY-CATCH added for dicomreadVolume

        # Define a cylinder of 40mm height enclosing the glenoid
        Timer.start()
        Logger.log("Glenoid density: ")

        SCase = self.scapula.shoulder.SCase
        glenSurf = self.surface["points"]

        # Get local coordinate system from parent scapula object
        origin = self.scapula.coordSys.origin
        xAxis = self.scapula.coordSys.PA
        yAxis = self.scapula.coordSys.IS
        zAxis = self.scapula.coordSys.ML

        # Glenoid center is the closet point between the mean of the extracted
        # surface and all the points of the extracted surface.
        # This might be improved by find the "real" center, instead of
        # the closest one from the surface nodes.
        glenCenter = np.mean(glenSurf, axis=0)
        vect = np.zeros(glenSurf.shape)
        dist = np.zeros((glenSurf.shape[0],1))
        for i in range(glenSurf.shape[0]):
            vect[i,:] = glenSurf[i,:] - glenCenter
            dist[i] = np.linalg.norm(vect[i, :])
        glenCenterNode = np.argmin(dist)
        glenCenter = glenSurf[glenCenterNode,]

        # Rotation matrix to align to scapula coordinate system
        R = np.concatenate([xAxis.reshape(-1, 1),
                            yAxis.reshape(-1, 1),
                            zAxis.reshape(-1, 1)], axis=1)


        # Glenoid surface in scapular coordinate system
        glenSurf = (self.scapula.glenoid.surface["points"] - origin) @ R

        # Calculate the distance between the center of the scapular
        # coordinate system and each point of the glenoid surface

        glenoidCenterLocal = np.array([self.centerLocal["x"], self.centerLocal["y"], self.centerLocal["z"]])
        glenSurf = glenSurf - glenoidCenterLocal

        # Create a unit vector in the direction of the cylinder axis
        unit_vector = zAxis

        # Find the spinoglenoid notch located on the cylinder axis aligned with
        # its medial aspect of the cylinder with the
        spinoglenoid_notch = glenCenter + unit_vector * (-glenoidCenterLocal[2])

        normVector = []

        for n in range(len(glenSurf)):
            normVector.append(np.linalg.norm(glenSurf[n, :2]))
        norm_vector = np.array(normVector)

        # Define the radius of the cylinder as the maximum distance
        # between the scapular center and the glenoid points

        cylinder_radius = np.max(norm_vector)

        sphereRadius = self.radius
        sphereCenterGlen = self.center + self.centerLine
        # Including cylinder in the images

        # Extract the images as a 4D matrix where spatial is the
        # positions of the pixels in the CT coordinate system
        dataCTPath = SCase.dataCTPath
        filename = dataCTPath + os.sep + 'dicom'
        try:
            pixel_vals = readDicomVolume(readSlices(filename))
        except:
            Logger().logn("failed after %s" % Timer().stop())
            return False

        # Read the information from the middle dicom image (all image
        # informations are the same for all dicom files) to avoid a problem
        # if other files are located at the beginning or end of the folder
        list_ = os.listdir(filename)
        size_list_ = len(list_)
        dicominfopath = filename + os.sep + list_[int(round(size_list_)/2)]

        try:
            dicom_information = dcmread(dicominfopath)
        except:
            Logger().logn("failed after %s" % Timer().stop())
            return False

        # Take into account that the data will need to be linearly transformed
        # from stored space to memory to come back to true HU units afterwards:
        Rescale_slope = int(dicom_information.RescaleSlope)
        Rescale_intercept = int(dicom_information.RescaleIntercept)

        pixel_vals_shape = pixel_vals.shape

        # Extract position of each first pixel of each image (upper left corner)
        PatientPositions = readPatientPositions(readSlices(filename))

        # Get the distance between each pixel in neighboring rows and columns
        PixelSpacings = readPixelSpacings(readSlices(filename))

        # Get the position of the upper left pixel of the first image
        # and calculate the position of each pixel in the x,y
        # and z direction starting from the "origin" pixel
        origin_image = PatientPositions[0,]
        x_max = origin_image[0]+PixelSpacings[0,0]*(pixel_vals_shape[0])
        y_max = origin_image[1]+PixelSpacings[0,1]*(pixel_vals_shape[1])
        z_max = PatientPositions[-1,2]

        # Calculate the coefficient of the linear transformation (CT-scan
        # coordinate system (x,y,z) to images coordinates system (j,i,k))
        coefficients_i = np.polyfit([origin_image[0] , x_max],
                                    [1, pixel_vals_shape[0]], 1)
        a_i = coefficients_i[0]
        b_i = coefficients_i[1]

        coefficients_j = np.polyfit([origin_image[1] , y_max],
                                    [1, pixel_vals_shape[1]], 1)
        a_j = coefficients_j[0]
        b_j = coefficients_j[1]

        coefficients_k = np.polyfit([origin_image[2] , z_max],
                                    [1, pixel_vals_shape[2]], 1)
        a_k = coefficients_k[0]
        b_k = coefficients_k[1]

        # Pixel size : 0.31 mm x 0.31-0.51 mm x 0.51 mm
        dr = 0.31 # radius increment of the cylinder
        da = 0.25/cylinder_radius # %angle increment of the cylinder
        dz = 0.31 # height increment of the cylinder


        # Create two arrays : one empty (only 0) to fill with rescaling slope
        # and one with rescaling intercept to rescale value of the voxel
        # stored in matlab to its true HU value.
        V_cyl = np.zeros(pixel_vals_shape)
        V_rescale = np.zeros(pixel_vals_shape)

        h = 40 # height of the cylinder in [mm]
        for z in np.arange(-5, h, dz): # Make a h + 5 mm cylinder to be able to
            for r in np.arange(0, cylinder_radius, dr):
                for a in np.arange(0, 2*np.pi, da):

                    x = r * np.cos(a)
                    y = r * np.sin(a)
                    z = z

                    # Rotate and translate the cylinder to its position in the CTs
                    cylinder_coord_scap = R @ np.array([[x],[y],[z]]) + spinoglenoid_notch.reshape(-1, 1)

                    # Perform linear transformation CT (x,y,z) to image
                    # (j,i,k) coordinate system
                    i = a_i*cylinder_coord_scap[0,0] + b_i
                    j = a_j*cylinder_coord_scap[1,0] + b_j
                    k = a_k*cylinder_coord_scap[2,0] + b_k

                    # Round up to get integers
                    i = round(i)-1
                    j = round(j)-1
                    k = round(k)-1

                    # Ensure that the image coordinates don't exceed the
                    # size of the original images (constrain it at the
                    # bottom by avoiding negative cases and at the top
                    # with the maximum number of pixels in one
                    # direction)
                    if (i >= 0) and (j >= 0) and (k >= 0) and \
                        (i < pixel_vals_shape[1]) and \
                            (j < pixel_vals_shape[0]) and (k < pixel_vals_shape[2]):
                                # Fill the empty array and mask with the position of
                                # the cylinder
                                V_cyl[j,i,k] = Rescale_slope
                                V_rescale[j,i,k] = Rescale_intercept
                    else:
                        continue
        # Create a plate to close one side of the cylinder
        V_plate = np.zeros(pixel_vals_shape)
        for z in np.arange(-5, -4.9+0.1, 0.1):
            for y in np.arange(-cylinder_radius, cylinder_radius, 0.1):
                for x in np.arange(-cylinder_radius, cylinder_radius, 0.1):

                    cylinder_coord_scap = R @ np.array([[x],[y],[z]]) + spinoglenoid_notch.reshape(-1, 1)

                    # Perform linear transformation CT to image coordinate system
                    i = a_i*cylinder_coord_scap[0,0] + b_i
                    j = a_j*cylinder_coord_scap[1,0] + b_j
                    k = a_k*cylinder_coord_scap[2,0] + b_k

                    # Round up to get integers
                    i = round(i)-1
                    j = round(j)-1
                    k = round(k)-1


                    # Ensure that the image coordinates don't exceed the
                    # size of the original images (constrain it at the
                    # bottom by avoiding negative cases and at the top
                    # with the maximum number of pixels in one
                    # direction)
                    if (i >= 0) and (j >= 0) and (k >= 0) and \
                        (i < pixel_vals_shape[1]) and \
                            (j < pixel_vals_shape[0]) and (k < pixel_vals_shape[2]):
                                # Fill the empty array of the plate with a value of
                                # 1000 to avoid erasing of the plate by thresholding
                                V_plate[j,i,k] = 1000
                    else:
                        continue
        V_plate_mask = np.ones(pixel_vals_shape)
        for z in np.linspace(-5, 0, 51):
            for y in np.arange(-cylinder_radius, cylinder_radius, 0.1):
                for x in np.arange(-cylinder_radius, cylinder_radius, 0.1):

                    cylinder_coord_scap = R @ np.array([[x],[y],[z]]) + spinoglenoid_notch.reshape(-1, 1)

                    # Perform linear transformation CT to image coordinate system
                    i = a_i*cylinder_coord_scap[0,0] + b_i
                    j = a_j*cylinder_coord_scap[1,0] + b_j
                    k = a_k*cylinder_coord_scap[2,0] + b_k

                    # Round up to get integers
                    i = round(i)-1
                    j = round(j)-1
                    k = round(k)-1


                    # Ensure that the image coordinates don't exceed the
                    # size of the original images (constrain it at the
                    # bottom by avoiding negative cases and at the top
                    # with the maximum number of pixels in one
                    # direction)
                    if (i >= 0) and (j >= 0) and (k >= 0) and \
                        (i < pixel_vals_shape[1]) and \
                            (j < pixel_vals_shape[0]) and (k < pixel_vals_shape[2]):
                                # Fill the empty array of the plate with a value of
                                # 1000 to avoid erasing of the plate by thresholding
                                V_plate_mask[j,i,k] = 0
                    else:
                        continue
        # Create a sphere to exclude humeral head
        V_sphere = np.ones(pixel_vals_shape)
        for r in np.arange(0, sphereRadius, dr):
            for a in np.arange(0, 2*np.pi, da/2):
                for phi in np.arange(0, np.pi, da):

                    x = r * np.sin(phi) * np.cos(a)
                    y = r * np.sin(phi) * np.sin(a)
                    z = r * np.cos(phi)

                    sphere_coord_scap = np.array([[x],[y],[z]]) + sphereCenterGlen.reshape(-1, 1)

                    # Perform linear transformation CT to image coordinate system
                    i = a_i*sphere_coord_scap[0,0] + b_i
                    j = a_j*sphere_coord_scap[1,0] + b_j
                    k = a_k*sphere_coord_scap[2,0] + b_k

                    # Round up to get integers
                    i = round(i) - 1
                    j = round(j) - 1
                    k = round(k) - 1


                    # Ensure that the image coordinates don't exceed the
                    # size of the original images (constrain it at the
                    # bottom by avoiding negative cases and at the top
                    # with the maximum number of pixels in one
                    # direction)
                    if (i >= 0) and (j >= 0) and (k >= 0) and \
                        (i < pixel_vals_shape[1]) and \
                            (j < pixel_vals_shape[0]) and (k < pixel_vals_shape[2]):
                                # Fill the empty array and mask with the position of
                                # the cylinder
                                V_sphere[j,i,k] = 0
                    else:
                        continue
        # Create masks of the 5 spherical shells
        V_sphere_SC = np.zeros(pixel_vals_shape)
        for r in np.arange(0, sphereRadius+3, dr):
            for a in np.arange(0, 2*np.pi, da/2):
                for phi in np.arange(0, np.pi, da):

                    x = r * np.sin(phi) * np.cos(a)
                    y = r * np.sin(phi) * np.sin(a)
                    z = r * np.cos(phi)


                    sphere_coord_scap = np.array([[x],[y],[z]]) + sphereCenterGlen.reshape(-1, 1)

                    # Perform linear transformation CT to image coordinate system
                    i = a_i*sphere_coord_scap[0,0] + b_i
                    j = a_j*sphere_coord_scap[1,0] + b_j
                    k = a_k*sphere_coord_scap[2,0] + b_k

                    # Round up to get integers
                    i = round(i) - 1
                    j = round(j) - 1
                    k = round(k) - 1


                    # Ensure that the image coordinates don't exceed the
                    # size of the original images (constrain it at the
                    # bottom by avoiding negative cases and at the top
                    # with the maximum number of pixels in one
                    # direction)
                    if (i >= 0) and (j >= 0) and (k >= 0) and \
                        (i < pixel_vals_shape[1]) and \
                            (j < pixel_vals_shape[0]) and (k < pixel_vals_shape[2]):
                                # Fill the empty array and mask with the position of
                                # the cylinder
                                V_sphere_SC[j,i,k] = 1
                    else:
                        continue
        V_sphere_ST = np.zeros(pixel_vals_shape)
        for r in np.arange(0, sphereRadius+6, dr):
            for a in np.arange(0, 2*np.pi, da/2):
                for phi in np.arange(0, np.pi, da):

                    x = r * np.sin(phi) * np.cos(a)
                    y = r * np.sin(phi) * np.sin(a)
                    z = r * np.cos(phi)


                    sphere_coord_scap = np.array([[x],[y],[z]]) + sphereCenterGlen.reshape(-1, 1)

                    # Perform linear transformation CT to image coordinate system
                    i = a_i*sphere_coord_scap[0,0] + b_i
                    j = a_j*sphere_coord_scap[1,0] + b_j
                    k = a_k*sphere_coord_scap[2,0] + b_k

                    # Round up to get integers
                    i = round(i) - 1
                    j = round(j) - 1
                    k = round(k) - 1


                    # Ensure that the image coordinates don't exceed the
                    # size of the original images (constrain it at the
                    # bottom by avoiding negative cases and at the top
                    # with the maximum number of pixels in one
                    # direction)
                    if (i >= 0) and (j >= 0) and (k >= 0) and \
                        (i < pixel_vals_shape[1]) and \
                            (j < pixel_vals_shape[0]) and (k < pixel_vals_shape[2]):
                                # Fill the empty array and mask with the position of
                                # the cylinder
                                V_sphere_ST[j,i,k] = 1
                    else:
                        continue
        V_sphere_T1 = np.zeros(pixel_vals_shape)
        for r in np.arange(0, sphereRadius+9, dr):
            for a in np.arange(0, 2*np.pi, da/2):
                for phi in np.arange(0, np.pi, da):

                    x = r * np.sin(phi) * np.cos(a)
                    y = r * np.sin(phi) * np.sin(a)
                    z = r * np.cos(phi)


                    sphere_coord_scap = np.array([[x],[y],[z]]) + sphereCenterGlen.reshape(-1, 1)

                    # Perform linear transformation CT to image coordinate system
                    i = a_i*sphere_coord_scap[0,0] + b_i
                    j = a_j*sphere_coord_scap[1,0] + b_j
                    k = a_k*sphere_coord_scap[2,0] + b_k

                    # Round up to get integers
                    i = round(i) - 1
                    j = round(j) - 1
                    k = round(k) - 1


                    # Ensure that the image coordinates don't exceed the
                    # size of the original images (constrain it at the
                    # bottom by avoiding negative cases and at the top
                    # with the maximum number of pixels in one
                    # direction)
                    if (i >= 0) and (j >= 0) and (k >= 0) and \
                        (i < pixel_vals_shape[1]) and \
                            (j < pixel_vals_shape[0]) and (k < pixel_vals_shape[2]):
                                # Fill the empty array and mask with the position of
                                # the cylinder
                                V_sphere_T1[j,i,k] = 1
                    else:
                        continue
        V_sphere_T2 = np.zeros(pixel_vals_shape)
        for r in np.arange(0, sphereRadius+12, dr):
            for a in np.arange(0, 2*np.pi, da/2):
                for phi in np.arange(0, np.pi, da):

                    x = r * np.sin(phi) * np.cos(a)
                    y = r * np.sin(phi) * np.sin(a)
                    z = r * np.cos(phi)


                    sphere_coord_scap = np.array([[x],[y],[z]]) + sphereCenterGlen.reshape(-1, 1)

                    # Perform linear transformation CT to image coordinate system
                    i = a_i*sphere_coord_scap[0,0] + b_i
                    j = a_j*sphere_coord_scap[1,0] + b_j
                    k = a_k*sphere_coord_scap[2,0] + b_k

                    # Round up to get integers
                    i = round(i) - 1
                    j = round(j) - 1
                    k = round(k) - 1


                    # Ensure that the image coordinates don't exceed the
                    # size of the original images (constrain it at the
                    # bottom by avoiding negative cases and at the top
                    # with the maximum number of pixels in one
                    # direction)
                    if (i >= 0) and (j >= 0) and (k >= 0) and \
                        (i < pixel_vals_shape[1]) and \
                            (j < pixel_vals_shape[0]) and (k < pixel_vals_shape[2]):
                                # Fill the empty array and mask with the position of
                                # the cylinder
                                V_sphere_T2[j,i,k] = 1
                    else:
                        continue
        V_sphere_T3 = np.zeros(pixel_vals_shape)
        for r in np.arange(0, sphereRadius+15, dr):
            for a in np.arange(0, 2*np.pi, da/2):
                for phi in np.arange(0, np.pi, da/3):

                    x = r * np.sin(phi) * np.cos(a)
                    y = r * np.sin(phi) * np.sin(a)
                    z = r * np.cos(phi)
                    sphere_coord_scap = np.array([[x],[y],[z]]) + sphereCenterGlen.reshape(-1, 1)

                    # Perform linear transformation CT to image coordinate system
                    i = a_i*sphere_coord_scap[0,0] + b_i
                    j = a_j*sphere_coord_scap[1,0] + b_j
                    k = a_k*sphere_coord_scap[2,0] + b_k

                    # Round up to get integers
                    i = round(i) - 1
                    j = round(j) - 1
                    k = round(k) - 1


                    # Ensure that the image coordinates don't exceed the
                    # size of the original images (constrain it at the
                    # bottom by avoiding negative cases and at the top
                    # with the maximum number of pixels in one
                    # direction)
                    if (i >= 0) and (j >= 0) and (k >= 0) and \
                        (i < pixel_vals_shape[1]) and \
                            (j < pixel_vals_shape[0]) and (k < pixel_vals_shape[2]):
                                # Fill the empty array and mask with the position of
                                # the cylinder
                                V_sphere_T3[j,i,k] = 1
                    else:
                        continue
        try:
            V_visual = pixel_vals+1000*V_cyl
        except:
            Logger().logn("failed after %s" % Timer().stop())
            return 0

        # Create an array filled with the images contained within the
        # cylinder. Rescale it to the true HU values
        V_paint = pixel_vals*V_cyl + V_rescale

        # Set to zero the domain covered by the sphere (so the humerus)
        V_paint = V_paint*V_sphere

        # Close the cylinder with the plate
        V_paint = V_paint + V_plate
        # Volume processing
        # Initialize the mean vectors to calculate the mean in each part
        # of interest
        mean_vector_CO = []
        mean_vector_SC = []
        mean_vector_ST = []
        mean_vector_T1 = []
        mean_vector_T2 = []
        mean_vector_T3 = []

        # For loop across all the images to calculate glenoid bone
        # quality in each VOI of each image
        for n in range(pixel_vals_shape[2]):

            # Focus on one image and corresponding mask
            I = V_paint[:,:,n]
            I_plate_mask = V_plate_mask[:,:,n]

            # Apply a gaussian filter to smooth the edges. Sigma indicates
            # the Gaussian smoothing kernels.
            sigma = 0.3
            I_blurred = gaussian_filter(I, sigma)

            # Segment bone with a threshold of 300 HU
            threshold = 300
            I_threshold = I_blurred > threshold

            # Fill the holes in the image with 8-connected pixel for filling
            I_filled = binary_fill_holes(I_threshold,
                                         structure=np.ones((3,3))).astype(int)

            # Clean the remaining parts that are outside the glenoid part we consider
            I_cleaned = remove_small_objects(I_filled,  min_size=50, connectivity=4)

            # Create the image where the glenoid bone is filled and substract
            # from it the binarized plate to not take it into account in the
            # calculation of the mean
            I_plate_bin = V_plate[:,:,n] > 0
            I_mask = I_cleaned - I_plate_bin

            # Create a structural element having a disk shape with a radius
            # of the wanted erosion. Then convert it to image coordinate system
            # by dividing by the spacing in mm between two pixels
            erosion_mag = 3 # [mm]
            erosion_image_coord = int(erosion_mag/PixelSpacings[0, 0])
            footprint = disk(erosion_image_coord)
            I_eroded = erosion(I_mask, footprint)

            # Separate the image between cortical bone and trabecular
            I_mask_fin = remove_small_objects((I_mask*I_plate_mask).astype(bool),
                                              min_size=50, connectivity=4)
            I_eroded_fin = I_eroded*I_plate_mask

            I_cortical = I_mask_fin-I_eroded_fin

            # Get SC, ST, T1, T2 and T3 by substracting each VOI by the previous one
            I_SC = binary_fill_holes(V_sphere_SC[:,:,n],
                                         structure=np.ones((3,3))).astype(int)
            I_ST = binary_fill_holes(V_sphere_ST[:,:,n],
                                         structure=np.ones((3,3))).astype(int) \
                - binary_fill_holes(V_sphere_SC[:,:,n],
                                         structure=np.ones((3,3))).astype(int)
            I_T1 = binary_fill_holes(V_sphere_T1[:,:,n],
                                         structure=np.ones((3,3))).astype(int) \
                        - binary_fill_holes(V_sphere_ST[:,:,n],
                                         structure=np.ones((3,3))).astype(int)
            I_T2 = binary_fill_holes(V_sphere_T2[:,:,n],
                                         structure=np.ones((3,3))).astype(int) \
                        - binary_fill_holes(V_sphere_T1[:,:,n],
                                         structure=np.ones((3,3))).astype(int)
            I_T3 = binary_fill_holes(V_sphere_T3[:,:,n],
                                         structure=np.ones((3,3))).astype(int) \
                        - binary_fill_holes(V_sphere_T2[:,:,n],
                                         structure=np.ones((3,3))).astype(int)

            # Binarize the image to only have 0 and 1 in the final VOI masks
            I_SC_fin = I_SC*I_mask_fin > 0
            I_ST_fin = I_ST*I_eroded_fin > 0
            I_T1_fin = I_T1*I_eroded_fin > 0
            I_T2_fin = I_T2*I_eroded_fin > 0
            I_T3_fin = I_T3*I_eroded_fin > 0
            I_CO_fin = I_cortical - I_SC_fin > 0

            # Flatten the image to ease the calculation of the mean
            I_flat = I.reshape(I.shape[0]*I.shape[1],1)

            # Measure mean in the CO
            I_CO_flat = I_CO_fin.reshape(I_mask.shape[0]*I_mask.shape[1], 1)

            # For value in the flat CO = 1 (where there is glenoid), set
            # the mean_vector to the value of the corresponding pixel value
            # of the glenoid image
            for i in range(I_mask.shape[0]*I_mask.shape[1]):

                if I_CO_flat[i] == 1:
                    mean_vector_CO.append(I_flat[i])
                else:
                    continue


            # Measure mean in the SC
            I_SC_flat = I_SC_fin.reshape(I_mask.shape[0]*I_mask.shape[1], 1)

            # For value in the flat SC = 1 (where there is glenoid), set
            # the mean_vector to the value of the corresponding pixel value
            # of the glenoid image
            for i in range(I_mask.shape[0]*I_mask.shape[1]):

                if I_SC_flat[i] == 1:
                    mean_vector_SC.append(I_flat[i])
                else:
                    continue

            # Measure mean in the ST
            I_ST_flat = I_ST_fin.reshape(I_mask.shape[0]*I_mask.shape[1], 1)

            # For value in the flat ST = 1 (where there is glenoid), set
            # the mean_vector to the value of the corresponding pixel value
            # of the glenoid image
            for i in range(I_mask.shape[0]*I_mask.shape[1]):

                if I_ST_flat[i] == 1:
                    mean_vector_ST.append(I_flat[i])
                else:
                    continue

            # Measure mean in the T1
            I_T1_flat = I_T1_fin.reshape(I_mask.shape[0]*I_mask.shape[1], 1)

            # For value in the flat T1 = 1 (where there is glenoid), set
            # the mean_vector to the value of the corresponding pixel value
            # of the glenoid image
            for i in range(I_mask.shape[0]*I_mask.shape[1]):

                if I_T1_flat[i] == 1:
                    mean_vector_T1.append(I_flat[i])
                else:
                    continue

            # Measure mean in the T2
            I_T2_flat = I_T2_fin.reshape(I_mask.shape[0]*I_mask.shape[1], 1)

            # For value in the flat T2 = 1 (where there is glenoid), set
            # the mean_vector to the value of the corresponding pixel value
            # of the glenoid image
            for i in range(I_mask.shape[0]*I_mask.shape[1]):

                if I_T2_flat[i] == 1:
                    mean_vector_T2.append(I_flat[i])
                else:
                    continue

            # Measure mean in the T3
            I_T3_flat = I_T3_fin.reshape(I_mask.shape[0]*I_mask.shape[1], 1)

            # For value in the flat T3 = 1 (where there is glenoid), set
            # the mean_vector to the value of the corresponding pixel value
            # of the glenoid image
            for i in range(I_mask.shape[0]*I_mask.shape[1]):

                if I_T3_flat[i] == 1:
                    mean_vector_T3.append(I_flat[i])
                else:
                    continue

        mean_CO = np.mean(np.array(mean_vector_CO))
        mean_SC = np.mean(np.array(mean_vector_SC))
        mean_ST = np.mean(np.array(mean_vector_ST))
        mean_T1 = np.mean(np.array(mean_vector_T1))
        mean_T2 = np.mean(np.array(mean_vector_T2))
        mean_T3 = np.mean(np.array(mean_vector_T3))

        self.density = np.array([mean_CO, mean_SC, mean_ST,
                                mean_T1, mean_T2, mean_T3])
        Logger().logn("OK %s" % Timer().stop())
        return 1
    
    #@tf.function
    def measureDensity(self, *args):
        """
        Calculate the density of the glenoid bone in 6 volumes of interest
        """
        # This function was developped by Paul Cimadomo during a semester  project (Fall 2018-19)
        # and implemented by Alexandre Terrier (2019-02-03)
        # % sCase --> SCase
        # obj.cylinder removed
        # obj.volume removed
        # TRY-CATCH added for dicomreadVolume

        # Define a cylinder of 40mm height enclosing the glenoid
        Timer.start()
        Logger.log("Glenoid density: ")

        SCase = self.scapula.shoulder.SCase
        glenSurf = self.surface["points"]
        glenSurf = tf.constant(glenSurf, dtype=tf.float32)

        # Get local coordinate system from parent scapula object
        origin = tf.constant(self.scapula.coordSys.origin, dtype=tf.float32)
        xAxis = tf.constant(self.scapula.coordSys.PA, dtype=tf.float32)
        yAxis = tf.constant(self.scapula.coordSys.IS, dtype=tf.float32)
        zAxis = tf.constant(self.scapula.coordSys.ML, dtype=tf.float32)

        # Glenoid center is the closet point between the mean of the extracted
        # surface and all the points of the extracted surface.
        # This might be improved by find the "real" center, instead of
        # the closest one from the surface nodes.
        glenCenter = tf.reduce_mean(glenSurf, axis=0)
        vect = tf.Variable(initial_value=tf.zeros(shape=glenSurf.shape), dtype=tf.float32)
        dist = tf.Variable(initial_value=tf.zeros(shape=(glenSurf.shape[0],1)), dtype=tf.float32)
        for i in tf.range(glenSurf.shape[0]):
            vect[i,:].assign(glenSurf[i,:] - glenCenter)
            norm_vec = [tf.Variable(tf.norm(vect[i, :]))]
            dist[i].assign(norm_vec)
        glenCenterNode = tf.math.argmin(dist)
        glenCenter = glenSurf[glenCenterNode[0],:]

        # Rotation matrix to align to scapula coordinate system
        R = tf.concat([tf.reshape(xAxis, shape=(-1, 1)),
                       tf.reshape(yAxis, shape=(-1, 1)),
                       tf.reshape(zAxis, shape=(-1, 1))], axis=1)


        # Glenoid surface in scapular coordinate system
        glenSurf = (tf.constant(self.scapula.glenoid.surface["points"]) - origin) @ R

        # Calculate the distance between the center of the scapular
        # coordinate system and each point of the glenoid surface

        glenoidCenterLocal = tf.broadcast_to(tf.constant([self.centerLocal["x"],
                                                          self.centerLocal["y"],
                                                          self.centerLocal["z"]],
                                                          dtype=tf.float32),
                                             [glenSurf.shape[0], 3])
        glenSurf = glenSurf - glenoidCenterLocal

        # Create a unit vector in the direction of the cylinder axis
        unit_vector = zAxis

        # Find the spinoglenoid notch located on the cylinder axis aligned with
        # its medial aspect of the cylinder with the
        spinoglenoid_notch = glenCenter + unit_vector * (-glenoidCenterLocal[2])

        normVector = tf.Variable(initial_value=tf.zeros(shape=(len(glenSurf), 1)))

        for n in tf.range(len(glenSurf)):
            normVector[n].assign([tf.norm(glenSurf[n, :2])])

        # Define the radius of the cylinder as the maximum distance
        # between the scapular center and the glenoid points

        cylinder_radius = tf.reduce_max(normVector)

        sphereRadius = self.radius
        sphereCenterGlen = tf.constant(self.center, dtype=tf.float32) + tf.constant(self.centerLine, dtype=tf.float32)
        # Including cylinder in the images

        # Extract the images as a 4D matrix where spatial is the
        # positions of the pixels in the CT coordinate system
        dataCTPath = SCase.dataCTPath
        filename = dataCTPath + os.sep + 'dicom'
        try:
            pixel_vals = readDicomVolume(readSlices(filename))
            pixel_vals = tf.constant(pixel_vals, dtype=tf.float32)
        except:
            Logger().logn("failed after %s" % Timer().stop())
            return False

        # Read the information from the middle dicom image (all image
        # informations are the same for all dicom files) to avoid a problem
        # if other files are located at the beginning or end of the folder
        list_ = os.listdir(filename)
        size_list_ = len(list_)
        dicominfopath = filename + os.sep + list_[int(round(size_list_)/2)]

        try:
            dicom_information = dcmread(dicominfopath)
        except:
            Logger().logn("failed after %s" % Timer().stop())
            return False

        # Take into account that the data will need to be linearly transformed
        # from stored space to memory to come back to true HU units afterwards:
        Rescale_slope = int(dicom_information.RescaleSlope)
        Rescale_intercept = int(dicom_information.RescaleIntercept)

        pixel_vals_shape = pixel_vals.shape

        # Extract position of each first pixel of each image (upper left corner)
        PatientPositions = readPatientPositions(readSlices(filename))
        PatientPositions = tf.constant(PatientPositions, dtype=tf.float32)

        # Get the distance between each pixel in neighboring rows and columns
        PixelSpacings = readPixelSpacings(readSlices(filename))
        PixelSpacings = tf.constant(PixelSpacings, dtype=tf.float32)

        # Get the position of the upper left pixel of the first image
        # and calculate the position of each pixel in the x,y
        # and z direction starting from the "origin" pixel
        origin_image = PatientPositions[0,]
        x_max = origin_image[0]+PixelSpacings[0,0]*(pixel_vals_shape[0])
        y_max = origin_image[1]+PixelSpacings[0,1]*(pixel_vals_shape[1])
        z_max = PatientPositions[-1,2]

        # Calculate the coefficient of the linear transformation (CT-scan
        # coordinate system (x,y,z) to images coordinates system (j,i,k))
        coefficients_i = np.polyfit([origin_image[0] , x_max],
                                    [1, pixel_vals_shape[0]], 1)
        a_i = coefficients_i[0]
        b_i = coefficients_i[1]

        coefficients_j = np.polyfit([origin_image[1] , y_max],
                                    [1, pixel_vals_shape[1]], 1)
        a_j = coefficients_j[0]
        b_j = coefficients_j[1]

        coefficients_k = np.polyfit([origin_image[2] , z_max],
                                    [1, pixel_vals_shape[2]], 1)
        a_k = coefficients_k[0]
        b_k = coefficients_k[1]

        # Pixel size : 0.31 mm x 0.31-0.51 mm x 0.51 mm
        dr = 0.31 # radius increment of the cylinder
        da = 0.25/cylinder_radius # %angle increment of the cylinder
        dz = 0.31 # height increment of the cylinder


        # Create two arrays : one empty (only 0) to fill with rescaling slope
        # and one with rescaling intercept to rescale value of the voxel
        # stored in matlab to its true HU value.
        #V_cyl = tf.Variable(initial_value=tf.zeros(shape=pixel_vals_shape, dtype=tf.int16), dtype=tf.int16)
        #V_rescale = tf.Variable(initial_value=tf.zeros(shape=pixel_vals_shape, dtype=tf.int16), dtype=tf.int16)

        h = 40 # height of the cylinder in [mm]
        V_cyl = volumeGenerator(tf.range(-5, h, dz),
                                tf.range(0, cylinder_radius, dr),
                                tf.range(0, 2 * np.pi, da),
                                spinoglenoid_notch, R,
                                a_i, a_j, a_k, b_i, b_j, b_k,
                                Rescale_slope, pixel_vals_shape,
                                vol="cylinder", sparse=True)
        V_rescale = volumeGenerator(tf.range(-5, h, dz),
                                    tf.range(0, cylinder_radius, dr),
                                    tf.range(0, 2 * np.pi, da),
                                    spinoglenoid_notch, R,
                                    a_i, a_j, a_k, b_i, b_j, b_k,
                                    Rescale_intercept, pixel_vals_shape,
                                    vol="cylinder", sparse=True)
        # Create a plate to close one side of the cylinder
        V_plate = volumeGenerator(tf.range(-5, -4.9+0.1, 0.1),
                                  tf.range(-cylinder_radius, cylinder_radius, 0.1),
                                  tf.range(-cylinder_radius, cylinder_radius, 0.1),
                                  spinoglenoid_notch, R,
                                  a_i, a_j, a_k, b_i, b_j, b_k,
                                  1000.0, pixel_vals_shape,
                                  vol="plate", sparse=True)

        # V_plate_mask filled with one, assign 0 to ijk
        V_plate_mask = volumeGenerator(tf.range(-5, 0.1, 0.1),
                                       tf.range(-cylinder_radius, cylinder_radius, 0.1),
                                       tf.range(-cylinder_radius, cylinder_radius, 0.1),
                                       spinoglenoid_notch, R,
                                       a_i, a_j, a_k, b_i, b_j, b_k,
                                       0., pixel_vals_shape,
                                       vol="plate", sparse=False)

        # Create a sphere to exclude humeral head
        # V_sphere filled with one, assign 0 to ijk
        V_sphere = volumeGenerator(tf.range(0, sphereRadius, dr),
                                   tf.range(0, 2*np.pi, da/2),
                                   tf.range(0, np.pi, da),
                                   sphereCenterGlen, "",
                                   a_i, a_j, a_k, b_i, b_j, b_k,
                                   0., pixel_vals_shape,
                                   vol="sphere", sparse=False)

        # Create masks of the 5 spherical shells
        V_sphere_SC = volumeGenerator(tf.range(0, sphereRadius+3, dr),
                                       tf.range(0, 2*np.pi, da/2),
                                       tf.range(0, np.pi, da),
                                       sphereCenterGlen, "",
                                       a_i, a_j, a_k, b_i, b_j, b_k,
                                       1, pixel_vals_shape,
                                       vol="sphere", sparse=True)

        V_sphere_ST = volumeGenerator(tf.range(0, sphereRadius+6, dr),
                                      tf.range(0, 2*np.pi, da/2),
                                      tf.range(0, np.pi, da),
                                      sphereCenterGlen, "",
                                      a_i, a_j, a_k, b_i, b_j, b_k,
                                      1, pixel_vals_shape,
                                      vol="sphere", sparse=True)

        V_sphere_T1 = volumeGenerator(tf.range(0, sphereRadius+9, dr),
                                      tf.range(0, 2*np.pi, da/2),
                                      tf.range(0, np.pi, da),
                                      sphereCenterGlen, "",
                                      a_i, a_j, a_k, b_i, b_j, b_k,
                                      1, pixel_vals_shape,
                                      vol="sphere", sparse=True)

        V_sphere_T2 = volumeGenerator(tf.range(0, sphereRadius + 12, dr),
                                      tf.range(0, 2 * np.pi, da / 2),
                                      tf.range(0, np.pi, da),
                                      sphereCenterGlen, "",
                                      a_i, a_j, a_k, b_i, b_j, b_k,
                                      1, pixel_vals_shape,
                                      vol="sphere", sparse=True)

        V_sphere_T3 = volumeGenerator(tf.range(0, sphereRadius + 15, dr),
                                      tf.range(0, 2 * np.pi, da / 2),
                                      tf.range(0, np.pi, da/3),
                                      sphereCenterGlen, "",
                                      a_i, a_j, a_k, b_i, b_j, b_k,
                                      1, pixel_vals_shape,
                                      vol="sphere", sparse=True)

        try:
            V_visual = tf.sparse.add(pixel_vals, tf.cast(V_cyl*1000, tf.float32))
        except:
            Logger().logn("failed after %s" % Timer().stop())
            return 0

        # Create an array filled with the images contained within the
        # cylinder. Rescale it to the true HU values
        V_paint = tf.sparse.add(pixel_vals*tf.cast(V_cyl, tf.float32), tf.cast(V_rescale, tf.float32))


        # Set to zero the domain covered by the sphere (so the humerus)
        V_paint = V_paint*tf.cast(V_sphere, tf.float32)

        # Close the cylinder with the plate
        V_paint = tf.sparse.to_dense(tf.sparse.reorder(V_paint)) + tf.sparse.to_dense(tf.sparse.reorder(tf.cast(V_plate, tf.float32)))
        # Volume processing
        # Initialize the mean vectors to calculate the mean in each part
        # of interest
        mean_vector_CO = np.array([])
        mean_vector_SC = np.array([])
        mean_vector_ST = np.array([])
        mean_vector_T1 = np.array([])
        mean_vector_T2 = np.array([])
        mean_vector_T3 = np.array([])

        #instead of for on images???
        # For loop across all the images to calculate glenoid bone
        # quality in each VOI of each image
        for n in tf.range(pixel_vals_shape[2]):

            # Focus on one image and corresponding mask
            I = V_paint.numpy()[:, :, n]
            I_plate_mask = V_plate_mask.numpy()[:, :, n]

            # Apply a gaussian filter to smooth the edges. Sigma indicates
            # the Gaussian smoothing kernels.
            sigma = 0.3
            I_blurred = gaussian_filter(I, sigma=sigma)

            # Segment bone with a threshold of 300 HU
            threshold = 300
            I_threshold = I_blurred > threshold

            # Fill the holes in the image with 8-connected pixel for filling
            I_filled = binary_fill_holes(I_threshold,
                                         structure=np.ones((3,3))).astype(int)
            I_filled = tf.constant(I_filled)

            # Clean the remaining parts that are outside the glenoid part we consider
            I_cleaned = remove_small_objects(I_filled.numpy(),  min_size=50, connectivity=4)
            I_cleaned = tf.constant(I_cleaned)

            # Create the image where the glenoid bone is filled and substract
            # from it the binarized plate to not take it into account in the
            # calculation of the mean
            I_plate_bin = tf.sparse.to_dense(tf.sparse.reorder(tf.cast(V_plate, tf.float32))).numpy()[:, :, n] > 0
            I_mask = I_cleaned - I_plate_bin

            # Create a structural element having a disk shape with a radius
            # of the wanted erosion. Then convert it to image coordinate system
            # by dividing by the spacing in mm between two pixels
            erosion_mag = 3 # [mm]
            erosion_image_coord = int(erosion_mag/PixelSpacings[0, 0])
            footprint = disk(erosion_image_coord)
            I_eroded = erosion(I_mask, footprint)
            I_eroded = tf.constant(I_eroded)

            # Separate the image between cortical bone and trabecular
            I_mask_fin = remove_small_objects((I_mask*I_plate_mask).numpy().astype(bool),
                                              min_size=50, connectivity=4)
            I_mask_fin = tf.constant(I_mask_fin)
            I_eroded_fin = I_eroded*I_plate_mask

            I_cortical = I_mask_fin.numpy()-I_eroded_fin

            # Get SC, ST, T1, T2 and T3 by substracting each VOI by the previous one
            V_sphere_SC_n = tf.sparse.to_dense(tf.sparse.reorder(tf.cast(V_sphere_SC, tf.float32))).numpy()[:, :, n]
            I_SC = binary_fill_holes(V_sphere_SC_n, structure=np.ones((3,3))).astype(int)
            I_SC = tf.constant(I_SC)

            V_sphere_ST_n = tf.sparse.to_dense(tf.sparse.reorder(tf.cast(V_sphere_ST, tf.float32))).numpy()[:, :, n]
            I_ST = binary_fill_holes(V_sphere_ST_n, structure=np.ones((3,3))).astype(int) \
                   - binary_fill_holes(V_sphere_SC_n, structure=np.ones((3,3))).astype(int)
            I_ST = tf.constant(I_ST)

            V_sphere_T1_n = tf.sparse.to_dense(tf.sparse.reorder(tf.cast(V_sphere_T1, tf.float32))).numpy()[:, :, n]
            I_T1 = binary_fill_holes(V_sphere_T1_n, structure=np.ones((3,3))).astype(int) \
                   - binary_fill_holes(V_sphere_ST_n, structure=np.ones((3,3))).astype(int)
            I_T1 = tf.constant(I_T1)

            V_sphere_T2_n = tf.sparse.to_dense(tf.sparse.reorder(tf.cast(V_sphere_T2, tf.float32))).numpy()[:, :, n]
            I_T2 = binary_fill_holes(V_sphere_T2_n, structure=np.ones((3,3))).astype(int) \
                        - binary_fill_holes(V_sphere_T1_n, structure=np.ones((3,3))).astype(int)
            I_T2 = tf.constant(I_T2)

            V_sphere_T3_n = tf.sparse.to_dense(tf.sparse.reorder(tf.cast(V_sphere_T3, tf.float32))).numpy()[:, :, n]
            I_T3 = binary_fill_holes(V_sphere_T3_n, structure=np.ones((3,3))).astype(int) \
                        - binary_fill_holes(V_sphere_T2_n, structure=np.ones((3,3))).astype(int)
            I_T3 = tf.constant(I_T3)

            # Binarize the image to only have 0 and 1 in the final VOI masks
            I_SC_fin = I_SC.numpy()*I_mask_fin.numpy().astype(int) > 0
            I_ST_fin = I_ST.numpy()*I_eroded_fin.numpy().astype(int) > 0
            I_T1_fin = I_T1.numpy()*I_eroded_fin.numpy().astype(int) > 0
            I_T2_fin = I_T2.numpy()*I_eroded_fin.numpy().astype(int) > 0
            I_T3_fin = I_T3.numpy()*I_eroded_fin.numpy().astype(int) > 0
            I_CO_fin = I_cortical.numpy().astype(int) - I_SC_fin.astype(int) > 0

            # Flatten the image to ease the calculation of the mean
            I_flat = tf.reshape(I, shape=(I.shape[0]*I.shape[1],1))

            # Measure mean in the CO
            I_CO_flat = tf.reshape(I_CO_fin, shape=(I_mask.shape[0]*I_mask.shape[1], 1))

            # For value in the flat CO = 1 (where there is glenoid), set
            # the mean_vector to the value of the corresponding pixel value
            # of the glenoid image
            """
            for i in tf.range(I_mask.shape[0]*I_mask.shape[1]):

                if I_CO_flat[i]:
                    mean_vector_CO = np.append(mean_vector_CO, I_flat[i])
                else:
                    continue
            """
            mean_vector_CO = np.append(mean_vector_CO, tf.boolean_mask(I_flat, I_CO_flat))


            # Measure mean in the SC
            I_SC_flat = tf.reshape(I_SC_fin, shape=(I_mask.shape[0] * I_mask.shape[1], 1))

            # For value in the flat SC = 1 (where there is glenoid), set
            # the mean_vector to the value of the corresponding pixel value
            # of the glenoid image
            """
            for i in tf.range(I_mask.shape[0]*I_mask.shape[1]):

                if I_SC_flat[i]:
                    mean_vector_SC = np.append(mean_vector_SC, I_flat[i])
                else:
                    continue
            """
            mean_vector_SC = np.append(mean_vector_SC, tf.boolean_mask(I_flat, I_SC_flat))

            # Measure mean in the ST
            I_ST_flat = tf.reshape(I_ST_fin, shape=(I_mask.shape[0] * I_mask.shape[1], 1))

            # For value in the flat ST = 1 (where there is glenoid), set
            # the mean_vector to the value of the corresponding pixel value
            # of the glenoid image
            mean_vector_ST = np.append(mean_vector_ST, tf.boolean_mask(I_flat, I_ST_flat))

            # Measure mean in the T1
            I_T1_flat = tf.reshape(I_T1_fin, shape=(I_mask.shape[0] * I_mask.shape[1], 1))

            # For value in the flat T1 = 1 (where there is glenoid), set
            # the mean_vector to the value of the corresponding pixel value
            # of the glenoid image
            mean_vector_T1 = np.append(mean_vector_T1, tf.boolean_mask(I_flat, I_T1_flat))

            # Measure mean in the T2
            I_T2_flat = tf.reshape(I_T2_fin, shape=(I_mask.shape[0] * I_mask.shape[1], 1))

            # For value in the flat T2 = 1 (where there is glenoid), set
            # the mean_vector to the value of the corresponding pixel value
            # of the glenoid image
            mean_vector_T2 = np.append(mean_vector_T2, tf.boolean_mask(I_flat, I_T2_flat))

            # Measure mean in the T3
            I_T3_flat = tf.reshape(I_T3_fin, shape=(I_mask.shape[0] * I_mask.shape[1], 1))

            # For value in the flat T3 = 1 (where there is glenoid), set
            # the mean_vector to the value of the corresponding pixel value
            # of the glenoid image
            mean_vector_T3 = np.append(mean_vector_T3, tf.boolean_mask(I_flat, I_T3_flat))

        mean_CO = tf.reduce_mean(tf.constant(mean_vector_CO))
        mean_SC = tf.reduce_mean(tf.constant(mean_vector_SC))
        mean_ST = tf.reduce_mean(tf.constant(mean_vector_ST))
        mean_T1 = tf.reduce_mean(tf.constant(mean_vector_T1))
        mean_T2 = tf.reduce_mean(tf.constant(mean_vector_T2))
        mean_T3 = tf.reduce_mean(tf.constant(mean_vector_T3))

        self.density = np.array([mean_CO, mean_SC, mean_ST,
                                mean_T1, mean_T2, mean_T3])
        Logger().logn("OK %s" % Timer().stop())
        return 1

    def plot(self, color):
        # plot glenoid surface
        points = self.surface["points"]
        faces = self.surface["faces"]
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]

        f1_data = ff.create_trisurf(x=x, y=y, z=z, simplices=faces, show_colorbar=False, colormap=color)
        f1 = go.Figure(data=f1_data)

        # plot glenoid centerline
        pt1 = self.center
        pt2 = pt1 + self.centerLine
        x = [pt1[0], pt2[0]]
        y = [pt1[1], pt2[1]]
        z = [pt1[2], pt2[2]]
        f2 = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z,
                                            mode='markers+lines',
                                            name="Glenoid Center Line")])

        fig = go.Figure(data=f1.data + f2.data)
        #fig.show()
        return fig

def selectPointsFromDotProductWithAxis(points,coord,selectFunction):
    dotProductResults = points@coord.reshape(-1, 1)
    if selectFunction == "min":
        selectedPoints = np.argmin(dotProductResults, axis=0)
    elif selectFunction == "max":
        selectedPoints = np.argmax(dotProductResults, axis=0)
    return points[selectedPoints[0],]

def volumeGenerator(a_range, b_range, c_range, axis_, Rot_matrix,
                    a_i, a_j, a_k, b_i, b_j, b_k,
                    matrix_value, matrix_shape,
                    vol="cylinder", sparse=True):

    a, b, c = tf.meshgrid(tf.cast(a_range, tf.float32), b_range, c_range)

    if vol == "cylinder":
        x = b * tf.math.cos(c)
        y = b * tf.math.sin(c)
        z = a
    elif vol == "plate":
        x, y, z = c, b, a
    elif vol == "sphere":
        x = a * tf.math.sin(c) * tf.math.cos(b)
        y = a * tf.math.sin(c) * tf.math.sin(b)
        z = a * tf.math.cos(c)


    xlong = tf.reshape(x, shape=(1, x.shape[0] * x.shape[1] * x.shape[2]))
    ylong = tf.reshape(y, shape=(1, y.shape[0] * y.shape[1] * y.shape[2]))
    zlong = tf.reshape(z, shape=(1, z.shape[0] * z.shape[1] * z.shape[2]))
    xyz = tf.concat([xlong, ylong, zlong], axis=0)

    if Rot_matrix != "":

        axis_ = tf.broadcast_to(tf.reshape(axis_, shape=(-1, 1)), shape=(3, xyz.shape[1]))
        axis_ = tf.constant(axis_.numpy().round(4))
        Rot_matrix = Rot_matrix
        coord_scap = Rot_matrix @ xyz + axis_
        coord_scap = coord_scap

    elif Rot_matrix == "":
        axis_ = tf.broadcast_to(tf.reshape(axis_, shape=(-1, 1)),
                                shape=(3, xyz.shape[1]))
        coord_scap = xyz + axis_


    a_ijk = tf.constant([[a_i], [a_j], [a_k]])
    a_ijk = tf.broadcast_to(a_ijk, shape=(3, xyz.shape[1]))

    b_ijk = tf.constant([[b_i], [b_j], [b_k]])
    b_ijk = tf.broadcast_to(b_ijk, shape=(3, xyz.shape[1]))

    ijk = tf.cast(a_ijk, tf.float32) * coord_scap + tf.cast(b_ijk, tf.float32)
    
    ijk = tf.cast(tf.math.round(ijk)-1, tf.int64)

    i_condition = tf.logical_and(ijk[0, :] < matrix_shape[1], ijk[0, :] >= 0)
    i_condition = tf.reshape(i_condition, shape=(1, -1))
    j_condition = tf.logical_and(ijk[1, :] < matrix_shape[0], ijk[0, :] >= 0)
    j_condition = tf.reshape(j_condition, shape=(1, -1))
    k_condition = tf.logical_and(ijk[2, :] < matrix_shape[2], ijk[0, :] >= 0)
    k_condition = tf.reshape(k_condition, shape=(1, -1))

    ijk_condition = tf.concat([i_condition, j_condition, k_condition], axis=0)
    ijk_condition = tf.math.reduce_all(ijk_condition, axis=0)

    ijk = tf.boolean_mask(ijk, ijk_condition, axis=1)
    ijk = tf.transpose(ijk)
    ijk = tf.constant(np.unique(ijk, axis=0))
    jik = tf.constant(ijk.numpy()[:, [1,0,2]])

    if sparse == True:

        return tf.sparse.SparseTensor(indices=jik,
                                      values=tf.ones(shape=(jik.shape[0],), dtype=tf.int64) * tf.cast(matrix_value, tf.int64),
                                      dense_shape=matrix_shape)

    output_ = tf.ones(shape=matrix_shape, dtype=tf.int64)
    output_ = tf.tensor_scatter_nd_update(output_,
                                          indices=jik,
                                          updates=tf.zeros(dtype=tf.int64, shape=(jik.shape[0],)))
    return output_

def vrrotvec(vec1, vec2):
    rotation_axis = np.cross(vec1, vec2)/np.linalg.norm(np.cross(vec1, vec2))
    rotation_angle = np.arccos(np.dot(vec1, vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2)))*180/np.pi
    Result = namedtuple("Result", ["rotation_axis", "rotation_angle"])
    return Result(rotation_axis, rotation_angle)

