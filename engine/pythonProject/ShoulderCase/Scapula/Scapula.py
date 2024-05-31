import numpy as np
from utils.Plane.Plane import Plane
from utils.CoordinateSystemAnatomical.CoordinateSystemAnatomical import CoordinateSystemAnatomical
from ShoulderCase.Acromion.Acromion import Acromion
from utils.Logger.Logger import Logger
from ShoulderCase.findLongest3DVector import findLongest3DVector
from ShoulderCase.orientVectorToward import orientVectorToward
from ShoulderCase.fitLine import fitLine
import warnings
from utils.Vector.Vector import Vector
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
import plotly.graph_objects as go
import plotly.io as pio
import os


class Scapula:
    """
    This class defines the scapula. Landmarks are used to define its
    coordinate system. It includes the glenoid object.

    """
    def __init__(self, shoulder):
        self.angulusInferior = np.array([])  # Landmark Angulus Inferior read from amira
        self.trigonumSpinae = np.array([])  # Landmark Trigonum Spinae Scapulae (the midpoint of the triangular surface on the medial border of the scapula in line with the scaoular spine read from amira
        self.processusCoracoideus = np.array([])  # Landmark Processus Coracoideus (most lateral point) read from amira
        self.acromioClavicular = np.array([])  # Landmark Acromio-clavicular joint (most lateral part on acromion)read from amira
        self.angulusAcromialis = np.array([])  # Landmark Angulus Acromialis (most laterodorsal point) read from amira
        self.spinoGlenoidNotch = np.array([])  # Landmark Spino-glenoid notch read from amira
        self.pillar = np.array([])  # 5 landmarks on the pillar
        self.groove = np.array([])  # 5 landmarks on the scapula (supraspinatus) groove

        self.friedmansLine = np.array([])  # Line that goes through trigonumSpinae end the glenoid center
        self.coordSys = CoordinateSystemAnatomical()
        self.plane = Plane()
        self.segmentation = "N"  # either none 'N', manual 'M' or automatic 'A'

        self.surface = {}  # surface points and triangles of the scapula
        self.glenoid = []
        self.acromion = Acromion(self)
        self.comment = ""
        self.shoulder = shoulder

    def isempty(self):
        """
        The following landmarks are used to define the scapula coordinate
        system which is the basis for all ShoulderCase measurements.
        """
        return self.angulusInferior.size == 0 \
            or self.trigonumSpinae.size == 0 \
            or self.processusCoracoideus.size == 0 \
            or self.groove.size == 0 \
            or self.spinoGlenoidNotch.size == 0 \
            or self.angulusAcromialis.size == 0

    def resetLandmarks(self):
        self.angulusInferior = np.array([])
        self.trigonumSpinae = np.array([])
        self.processusCoracoideus = np.array([])
        self.acromioClavicular = np.array([])
        self.angulusAcromialis = np.array([])
        self.spinoGlenoidNotch = np.array([])
        self.pillar = np.array([])
        self.groove = np.array([])

    def loadData(self):
        """
        Call methods that can be run after the ShoulderCase object has
        been instanciated.
        """
        success = Logger().timeLogExecution("Scapula landmarks: ",
                        lambda self: self.loadLandmarks(), self)
        success = success and Logger().timeLogExecution("Scapula groove points: ",
                        lambda self: self.loadGroovePoints(), self)
        success = success and Logger().timeLogExecution("Scapula pillar  points: ",
                        lambda self: self.loadPillarPoints(), self)

        # The following operations used to be called by morphology() but we need
        # the coordinate system to be checked in order to assess the shoulder
        # side consistency. Manualy segmented landmarks shoulder side is not
        # specified in the filename so a manual left shoulder might load right
        # side landmarks.
        success = Logger().timeLogExecution("Scapula plane: ",
                        lambda self: self.measurePlane(), self)
        success = success and Logger().timeLogExecution("Scapula coordinate system: ",
                        lambda self: self.measureCoordinateSystem(), self)
        if success and self.isInconsistentWithShoulderSide():
            Logger().logn("")
            Logger().logn("Loaded landmarks are inconsistent with shoulder side. ")
            Logger().logn("Shouder will be reset after data loading step.")
            Logger().logn("")
            # reset scapula data
            self.resetLandmarks()
            self.coordSys = CoordinateSystemAnatomical()
            self.plane = Plane()
            return
        #print(success)
        success = success and Logger().timeLogExecution("Scapula surface: ",
                        lambda self: self.loadSurface(), self)
        return success

    def morphology(self):
        """
        Call methods that can be run after loadData() methods has been run by
        all ShoulderCase objects.
        """
        return True

    def measureFirst(self):
        """
        Call methods that can be run after morphology() methods has been run by
        all ShoulderCase objects.
        """

        success = Logger().timeLogExecution("Scapula Friedmans line: ",
                        lambda self: self.measureFriedmansLine(), self)
        return success

    def getSortedGrooveLateralToMedial(self):
        TS = self.trigonumSpinae
        if np.all(self.groove):
            # sort groove from most lateral to most medial
            rawGroove = self.groove
            groove = np.zeros_like(rawGroove)
            for i in range(rawGroove.shape[0]):
                nextGrooveIndex = findLongest3DVector(TS-rawGroove)

                groove[i, :] = rawGroove[nextGrooveIndex, :]

                rawGroove = np.delete(rawGroove, nextGrooveIndex, axis=0)
        return groove

    def isInconsistentWithShoulderSide(self):
        return self.coordSys.isRightHanded() and not self.shoulder.side == "R" \
            or self.coordSys.isLeftHanded() and not self.shoulder.side == "L"

    def measureCoordinateSystem(self):
        try:
            self.setCoordinateSystemWithLandmarks()
        except Exception as e:
            warnings.warning(str(e))
            return False
        return True

    def measureFriedmansLine(self):
        """
        The Friedman's line goes through the glenoid center and the medial border of
        the scapula.
        """
        self.friedmansLine = Vector(self.glenoid.center, self.trigonumSpinae)

    def measurePlane(self):
        """
        Scapular plane is fitted on 3 points (angulusInferior,
        trigonumSpinae, most laretal scapular groove landmark).
        """
        inferior = self.angulusInferior.reshape(1, -1)
        medial = self.trigonumSpinae.reshape(1, -1)
        mostLateralGrooveIndex = findLongest3DVector(medial-self.groove)

        mostLateralGroovePoint = self.groove[mostLateralGrooveIndex, :].reshape(1, -1)
        self.plane.fit(np.concatenate([inferior,
                                       medial,
                                       mostLateralGroovePoint]))

        anterior = self.processusCoracoideus
        posterior = self.angulusAcromialis
        self.plane.normal = orientVectorToward(self.plane.normal, anterior.ravel()-posterior.ravel())

    def plotLandmarks(self, color):
        # get landmarks
        AI = self.angulusInferior
        TS = self.trigonumSpinae
        PC = self.processusCoracoideus
        AC = self.acromioClavicular
        AA = self.angulusAcromialis
        SG = self.spinoGlenoidNotch
        if self.groove.shape[0] != 0:
            # sort groove from most lateral to most medial
            rawGroove = self.groove
            groove = np.zeros((rawGroove.shape[0], rawGroove.shape[1]))
            for i in range(rawGroove.shape[0]):
                nextGrooveIndex = findLongest3DVector(TS-rawGroove)
                groove[i,:] = rawGroove[nextGrooveIndex, :]
                rawGroove = np.delete(rawGroove, nextGrooveIndex, axis=0)

        # plot wireframe
        acromion = np.concatenate([SG, AA, AC], axis=0)
        clavicle = np.concatenate([SG, PC], axis=0)
        scapulaPlane = np.concatenate([groove[0, :].reshape(1, -1),
                                       TS,
                                       AI,
                                       groove[0, :].reshape(1, -1),
                                       SG])

        acromionFig = go.Figure(data=[go.Scatter3d(x=acromion[:, 0], y=acromion[:, 1], z=acromion[:, 2],
                                                    mode='markers',
                                                    marker=dict(color=color),
                                                    showlegend=False)])

        clavicleFig = go.Figure(data=[go.Scatter3d(x=clavicle[:, 0], y=clavicle[:, 1], z=clavicle[:, 2],
                                                   mode='markers',
                                                   marker=dict(color=color),
                                                   showlegend=False)])

        scapulaPlaneFig = go.Figure(data=[go.Scatter3d(x=scapulaPlane[:, 0], y=scapulaPlane[:, 1], z=scapulaPlane[:, 2],
                                                   mode='markers',
                                                   marker=dict(color=color),
                                                   showlegend=False)])

        grooveFig = go.Figure(data=[go.Scatter3d(x=groove[:, 0], y=groove[:, 1], z=groove[:, 2],
                                                   mode='markers',
                                                   marker=dict(color=color),
                                                   showlegend=False)])

        fig = go.Figure(data=acromionFig.data + clavicleFig.data + scapulaPlaneFig.data + grooveFig.data)
        #fig.show()
        return fig

    def plotSurface(obj, color, lightingFeatures, lightPosition):
        if obj.surface:
            points = obj.surface["points"]
            faces  = obj.surface["faces"]
            x = points[:,0]
            y = points[:,1]
            z = points[:,2]

            simplices = faces

            data = ff.create_trisurf(x=x, y=y, z=z,
                                     simplices=simplices,
                                     show_colorbar=False,
                                     colormap=color,
                                     plot_edges=False)

            fig = go.Figure(data=data)
            fig.update_traces(lighting=lightingFeatures,
                              lightposition=lightPosition)
            return fig

    def setCoordinateSystemWithLandmarks(self):
        """
        Calculate the (EPFL) scapular coordinate system
        The EPFL scapular coordinate system is defined for a right scapula see
        paper (10.1302/0301-620X.96B4.32641). The X axis is
        antero-posterior, the Y axis is infero-superior, the Z axis
        is meddio-lateral. This system is right-handed.
        This orientation corresponds approximatively to the ISB
        recommendadtion (doi:10.1016/j.jbiomech.2004.05.042).
        For left scapula we keep the x axis as postero-anterior and
        get a left-handed coordinate system.
        """
        if len(self.groove) < 2:
            raise Exception("Can't set scapular coordinate system with actual self.groove")
            return
        # The four following methods have to be called in this specific order
        self.setPAAxis()
        self.setMLAxis()
        self.setISAxis()
        self.setOrigin()

    def setPAAxis(self):
        """
        Set the postero-anterior axis to be the scapula plane normal
        """
        self.coordSys.set_PA(self.plane.normal)

    def setMLAxis(self):
        """
        Set the media-lateral axis to be the line fitted to the projection of
        the groove points on the scapula plane
        """
        lateral = self.spinoGlenoidNotch
        medial = self.trigonumSpinae
        groovePointsProjection = self.plane.projectOnPlane(self.groove)
        grooveAxis = fitLine(groovePointsProjection)[0]
        grooveAxis = grooveAxis/np.linalg.norm(grooveAxis)

        self.coordSys.set_ML(orientVectorToward(grooveAxis, lateral.ravel()-medial.ravel()))


    def setISAxis(self):
        """
        Set the infero-superior axis to be orthogonal with the two other axes
        """
        superior = self.spinoGlenoidNotch
        inferior = self.angulusInferior
        self.coordSys.set_IS(np.cross(self.coordSys.PA, self.coordSys.ML))
        self.coordSys.set_IS(orientVectorToward(self.coordSys.IS, superior.ravel()-inferior.ravel()))

    def setOrigin(self):
        """
        Set the origin of the coordinate system to the spino-glenoid notch
        projected on the scapular axis
        """
        spinoGlenoid = self.spinoGlenoidNotch
        grooveMeanProjection = np.mean(self.plane.projectOnPlane(self.groove), axis=0)

        self.coordSys.origin = grooveMeanProjection + \
            np.dot((spinoGlenoid - grooveMeanProjection).ravel(), self.coordSys.ML) * self.coordSys.ML

