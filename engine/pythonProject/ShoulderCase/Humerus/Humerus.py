import numpy as np
from utils.Logger.Logger import Logger
from utils.Vector.Vector import Vector
import plotly.graph_objects as go

class Humerus:
    """
    HUMERUS Properties and methods associated to the humerus
    Detailed explanation goes here

    Author: Alexandre Terrier, EPFL-LBO
    Creation date: 2018-07-01
    Revision date: 2019-06-29
  
    TO DO:
    Local coordinate system
    """
    def __init__(self, shoulder):
        self.landmarks = {} # 5 3D points
        self.insertionsRing = {}
        self.center = [] #
        self.radius = [] # Radius of the humeral head (sphere fit on 5 points
        self.jointRadius = [] # Radius of cuvature of the articular surface (todo)
        self.SHSAngle = [] # Scapulo-humeral subluxation angle
        self.SHSPA = [] # Scapulo-humeral subluxation angle in the postero-anterior direction (posterior is negative, as for glenoid version)
        self.SHSIS = [] # Scapulo-humeral subluxation angle in the infero-superior direction
        self.SHSAmpl = [] # Scapulo-humeral subluxation (center ofset / radius)
        self.SHSOrient = [] # Scapulo-humral subluxation orientation (0 degree is posterior)
        self.GHSAmpl = [] # Gleno-humeral subluxation (center ofset / radius)
        self.GHSOrient = [] # Gleno-humral subluxation orientation (0 degree is posterior)
        self.subluxationIndex3D = []
        self.shoulder = shoulder
    
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
    
    def measureFirst(self):
        """
        Call methods that can be run after morphology() methods has been run 
        by all ShoulderCase objects.
        """
        success = Logger.timeLogExecution("Humerus scapulo-humeral subluxation: ",
                      lambda self : self.measureScapuloHumeralSubluxation(), self)
        return success
    
    def measureSecond(self):
        """
        Call methods that can be run after measureFirst() methods has been run 
        by all ShoulderCase objects.
        """
        success = Logger.timeLogExecution("Humerus gleno-humeral subluxation: ",
                      lambda self : self.measureGlenoHumeralSubluxation(), self)

        success = success and Logger.timeLogExecution("Humerus 3D subluxation index: ",
                      lambda self : self.measureSubluxationIndex3D(), self)
        return success

    def measureScapuloHumeralSubluxation(self):
        # Scapulo-humeral subluxation (SHS)

        scapula = self.shoulder.scapula

        xAxis = Vector(scapula.coordSys.origin,
                       scapula.coordSys.origin + scapula.coordSys.PA)
        yAxis = Vector(scapula.coordSys.origin,
                       scapula.coordSys.origin + scapula.coordSys.IS)
        zAxis = Vector(scapula.coordSys.origin,
                       scapula.coordSys.origin + scapula.coordSys.ML)

        glenoidToHH = Vector(scapula.glenoid.center, self.center)

        # SHS amplitude (ratio rather than amplitude)
        # glenoidToHHxy is the vector from humeral head center to scapular (z) axis,
        # perpendicular to z-axis --> similar to projection in xy plane
        glenoidToHHxy = zAxis.orthogonalComplementTo(glenoidToHH)
        if glenoidToHHxy.norm() > 5*self.radius:
            return
        self.SHSAmpl = glenoidToHHxy.norm() / self.radius

        # SHS orientation
        # Convention that superior is positive
        self.SHSOrient = (-xAxis).angle(glenoidToHHxy)
        self.SHSOrient = np.sign(yAxis.dot(glenoidToHHxy)) * self.SHSOrient

        # SHSAngle: angle between glenHumHead and zAxis
        self.SHSAngle = (glenoidToHH).angle(zAxis)

        # SHSPA is angle between zxProjection and zAxis
        # Convention that posterior is negative
        glenoidToHHzx = yAxis.orthogonalComplementTo(glenoidToHH)
        self.SHSPA = (glenoidToHHzx).angle(zAxis)
        self.SHSPA = np.sign(xAxis.dot(glenoidToHHzx)) * self.SHSPA

        # SHSIS is angle between yzProjection and zAxis
        # Convention that superior is positive
        glenoidToHHyz = xAxis.orthogonalComplementTo(glenoidToHH)
        self.SHSIS = (glenoidToHHyz).angle(zAxis)
        self.SHSIS = np.sign(yAxis.dot(glenoidToHHyz)) * self.SHSIS

    """"  
    def exportInsertionsPlane(self):
        towardSubscapularis = 2 * Vector.Vector(self.insertionsRing.center,
            self.landmarks["insertions"]["subscapularis_inferior"].T)
        towardSupraspinatus = 2 * Vector.Vector(self.insertionsRing.center,
            self.landmarks["insertions"]["supraspinatus_anterior"].T)
        towardTeresMinor = 2 * Vector.Vector(self.insertionsRing.center,
            self.landmarks["insertions"]["teres_minor_inferior"].T)
        
        insertionsPlane = SlicerMarkupsPlane();
        insertionsPlane.addControlPoint(SlicerControlPoint("near subscapularis",
                                                           towardSubscapularis.target))
        insertionsPlane.addControlPoint(SlicerControlPoint("near supraspinatus",
                                                           towardSupraspinatus.target))
        insertionsPlane.addControlPoint(SlicerControlPoint("near teresMinor",
                                                           towardTeresMinor.target))
        
        exporter = SlicerMarkupsExporter()
        exporter.addMarkups(insertionsPlane)
        exportFilename = os.path.join(self.shoulder.SCase.dataSlicerPath,
            "rotator_cuff_insertions_plane_" + self.shoulder.side + ".mrk.json")
        exporter.export(exportFilename)
    """
    
    def getInsertionsExtremitiesByMuscle(self):
        subscapularisExtremities = np.concatenate([self.landmarks["insertions"]["subscapularis_inferior"],
                                                   self.landmarks["insertions"]["subscapularis_superior"]],
                                                  axis=1)
        supraspinatusExtremities = np.concatenate([self.landmarks["insertions"]["supraspinatus_anterior"],
                                                   self.landmarks["insertions"]["supraspinatus_infraspinatus"]],
                                                  axis=1)
        infraspinatusExtremities = np.concatenate([self.landmarks["insertions"]["supraspinatus_infraspinatus"],
                                                   self.landmarks["insertions"]["infraspinatus_teres_minor"]],
                                                  axis=1)
        teresMinorExtremities = np.concatenate([self.landmarks["insertions"]["infraspinatus_teres_minor"],
                                                self.landmarks["insertions"]["teres_minor_inferior"]],
                                                  axis=1)
        insertionsExtremities = {}
        insertionsExtremities["subscapularis"] = subscapularisExtremities
        insertionsExtremities["supraspinatus"] = supraspinatusExtremities
        insertionsExtremities["infraspinatus"] = infraspinatusExtremities
        insertionsExtremities["teres_minor"] = teresMinorExtremities
        return insertionsExtremities

    def measureGlenoHumeralSubluxation(self):
        # Gleno-humeral subluxation (GHS)
        scapula = self.shoulder.scapula
        
        xAxis = Vector(scapula.coordSys.origin,
                       scapula.coordSys.origin + scapula.coordSys.PA)
        yAxis = Vector(scapula.coordSys.origin,
                       scapula.coordSys.origin + scapula.coordSys.IS)
        zAxis = Vector(scapula.coordSys.origin,
                       scapula.coordSys.origin + scapula.coordSys.ML)
    
        glenoidCenterLine = Vector(scapula.glenoid.center,
                                   scapula.glenoid.center + scapula.glenoid.centerLine)
        
        # Vector from glenoid sphere centre to glenoid centerline (perpendicular)
        GHS = glenoidCenterLine.orthogonalComplementTo(self.center)
        
        # GHS amplitude (ratio rather than amplitude)
        self.GHSAmpl = GHS.norm() / self.radius
        
        # GHS orientation
        # Superior is positive
        GHSxy = zAxis.orthogonalComplementTo(GHS)
        self.GHSOrient = (-xAxis).angle(GHSxy)
        self.GHSOrient = np.sign(yAxis.dot(GHSxy)) * self.GHSOrient
        
    def measureSubluxationIndex3D(self):
        # -	GL: (Glenoid Line) Line that joins the most anterior glenoid rim point and 
        #         the most posterior glenoid rim point.
        # -	HHv: (Humeral Head vector) Vector from the glenoid center to the humeral 
        #         head’s fitted sphere center.
        # -	HHp: (Humeral Head projection) Projection of HHv on GL.
        # -	HHr: (Humeral Head radius) Humeral head’s fitted sphere radius.
        #
        # The 3D humeral subluxation index is the ratio of ||HHp|| + ||HHr|| over 2*||HHr||.
        GL = self.shoulder.scapula.glenoid.posteroAnteriorLine
        HHv = Vector(self.shoulder.scapula.glenoid.center, self.center)
        HHp = GL.project(HHv)
        HHr = self.radius
        self.subluxationIndex3D = (HHp.norm() + HHr) / (2*HHr)

    def plot(self, color):
        def makeSphere(x, y, z, radius, resolution=20):
            """Return the coordinates for plotting a sphere centered at (x,y,z)"""
            u, v = np.mgrid[0:2 * np.pi:resolution * 2j, 0:np.pi:resolution * 1j]
            X = radius * np.cos(u) * np.sin(v) + x
            Y = radius * np.sin(u) * np.sin(v) + y
            Z = radius * np.cos(v) + z
            return (X, Y, Z)
        sphereX, sphereY, sphereZ = makeSphere(self.center[0], self.center[1], self.center[2], self.radius )
        colors = np.zeros(shape=sphereZ.shape)
        data = go.Surface(x=sphereX, y=sphereY, z=sphereZ, opacity=0.7,showscale=False,
                          colorscale=color, surfacecolor=colors, cmax=0, cmin=0)
        fig = go.Figure(data=data)
        #fig.show()
        return fig


def landmarksBelongToCorrectShoulder(landmarks, shoulder):
    """
    humerus landmarks should be located lateraly to the scapula coordinate system
    """

    medioLateralAxis = Vector(shoulder.scapula.coordSys.ML)
    scapulaOrigin = shoulder.scapula.coordSys.origin
    originToLandmarks = []
    for i in range(0, landmarks.shape[0]-1):
        originToLandmarks.append(Vector(scapulaOrigin, landmarks[i,:]))

    landmarksLaterality = np.zeros((1, len(originToLandmarks)))
    for i in range(len(originToLandmarks)):
        landmarksLaterality[0,i] = originToLandmarks[i].dot(medioLateralAxis)
    return np.all(landmarksLaterality > 0)
    
def raise_(ex):
    raise ex        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
