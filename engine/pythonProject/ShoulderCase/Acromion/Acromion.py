from utils.Logger.Logger import Logger
import numpy as np
from ShoulderCase.project2Plane import project2Plane
from utils.Rotations import rotation_angle
from utils.Vector.Vector import Vector


class Acromion:
    """"
    ACROMION Properties and methods associted to the acromion.
     Detailed explanation goes here
    """
    def __init__(self, scapula):
        self.AI = [] # Acromion Index (doi:10.2106/JBJS.D.03042)
        self.CSA = [] # Critital Shoulder Angle (doi:10.1302/0301-620X.95B7.31028)
        self.PSA = [] # Posterior Slope Angle (10.1016/S1058-2746(05)80036-9)
        self.PSL = [] # Length of segment beteeen AA and AC
        self.AAA = [] # Angulus Angle Angle (Angle between AA-origin and PA axis)
        self.AAL = [] # Length of segment between origine and AA
        self.APA = [] # acromion posterior angle
        self.AAx = [] # PA (x) position of AA
        self.AAy = [] # IS (y) position of AA
        self.AAz = [] # ML (z) position of AA
        self.ACx = [] # PA (x) position of AC
        self.ACy = [] # IS (y) position of AC
        self.ACz = [] # ML (z) position of AC
        self.APA = [] # Acromion posterior angle
        self.AA = [] # Acromion angle
        self.comment = ""
        self.scapula = scapula

    def measureFirst(self):
        # Can be run after morphology() methods has been run by all ShoulderCase objects.
        success = Logger.timeLogExecution("Acromion measurements: ",
                                          lambda self : self.measureEverything(), self)
    def measureEverything(self):
        """"
        Copy-paste of the former morphology() method.
        To refactor with methods extraction, obviously. Not enough time right now.

        It caluculates acromion index (AI), critical shoulder angle
        (CSA), and posterior slope (PS). The glenoid and scapula
        surface are re-oriented (rotated and translated) in the
        scapular coordinate system. For AI nd CSA, glenoid and
        scapula points are projected in the scapular plane, which is
        [1 0 0] after the re-orientation.

        TODO:
        Might be removed anatomy for concistency
        """
        SCase = self.scapula.shoulder.SCase

        # Get local coordinate system from parent scapula
        origin = self.scapula.coordSys.origin
        xAxis = self.scapula.coordSys.PA
        yAxis = self.scapula.coordSys.IS
        zAxis = self.scapula.coordSys.ML

        # Rotation matrix to align to scapula coordinate system
        R = np.c_[xAxis.reshape(-1, 1), yAxis.reshape(-1, 1), zAxis.reshape(-1, 1)]

        ## Scapula, glenoid, and scapula landmarks in scapula coordinate system

        # Scapula surface alignment is done in next section, for the
        # caulation of AI, because of mix between auto, manual and no
        # segmentation. This should be corrected when there will be two
        # scapula object per shoulder (manual and auto).

        # Glenoid surface in scapular coordinate system
        glenSurf = (self.scapula.glenoid.surface["points"] - origin) @ R

        # Glenoid center in scapular coordinate system
        glenCenter = (self.scapula.glenoid.center - origin) @ R

        # Scapula landmarks in scapular coordinate system
        AC = (self.scapula.acromioClavicular - origin) @ R  # Acromio - clavicula landmark
        AA = (self.scapula.angulusAcromialis - origin) @ R # Angulus acromialis landmark
        AInf = (self.scapula.angulusInferior - origin) @ R # Angulus Inferior landmark

        self.ACx = AC[0,0]
        self.ACy = AC[0,1]
        self.ACz = AC[0,2]
        self.AAx = AA[0,0]
        self.AAy = AA[0,1]
        self.AAz = AA[0,2]

        ## Acromion Index (AI)
        # Adapted by AT (2018-07-18) from Valerie Mafroy Camine (2017)

        # AI = GA/GH, where:
        # GA: GlenoAcromial distance in scapula plane = distance from
        # AL to glenoid principal axis
        # GH: GlenoHumeral distance = 2*HHRadius, humeral head diameter (radius*2)
        # AL: most lateral point of the acromion

        # Get all required data, aligned in scapular plane, and
        # project in scapular plane.

        ScapPlaneNormal = np.array([1, 0, 0]) # Normal of the scapular plane in the scapular coordinate system
        PlaneMean = np.array([0, 0, 0]) # Origin of scapular system in scapular coordinate system

        # Project glenoid surface in scapular plane
        glenSurf = project2Plane(glenSurf, ScapPlaneNormal, PlaneMean)

        # Project glenoid center in scapular plane
        glenCenter = project2Plane(glenCenter, ScapPlaneNormal, PlaneMean)

        # If scapula is segmented, get AL from most lateral part of the
        # scapula, otherwise use acromio-clavicular landmark
        # Get segmentation propertiy from parent scapula
        segmentedScapula = self.scapula.segmentation
        segmentedScapula = segmentedScapula == "A" or segmentedScapula == "M"
        if segmentedScapula:
            # Rotate scapula surface to align the scapular plane with
            # the YZ plane, and then take the max Z (lateral) point
            # Transform scapula surface to scapula coordinate system
            scapSurf = self.scapula.surface["points"]
            scapSurf = (scapSurf - origin) @ R
            scapSurf = project2Plane(scapSurf, ScapPlaneNormal, PlaneMean)

            # Take the most lateral point of the scapula, assuming that
            # this point is the most lateral point of the acromion, however to make sure we are selecting points
            # from Coracoid, we make sure the distance between the most lateral point of the acromion and
            # AC is below 2 cm.
            while True:
                ALpositionInScapula = np.argmax(scapSurf[:, 2])
                AL = scapSurf[ALpositionInScapula,:]
                if np.linalg.norm(AL - AC) / 10 <= 2:
                    break

                scapSurf = np.delete(scapSurf, ALpositionInScapula, axis=0)

        else:
            # No scapula points, we approximate AL with the acromioClavicular
            # landmark, in the scapula coordinate system
            AL = AC # Acromio-clavicalar scapula landmark
            AL = project2Plane(AL, ScapPlaneNormal, PlaneMean)

        # Find glenoid principal axis with most superior and most inferior projected glenoid points
        # AT: This method is not ideal. PCA would be better. It is also used below by the CSA
        glenPrinAxis = np.concatenate([glenSurf[np.where(glenSurf[:, 1] == np.min(glenSurf[:, 1])),:],
                                       glenSurf[np.where(glenSurf[:, 1] == np.max(glenSurf[:, 1])),:]], axis=0).squeeze()

        # Compute GA (distance from AL to glenoid principal axis)
        # Most inferior point of the glenoid surface
        IG = glenPrinAxis[0, :]
        # Most superior point of the glenoid surface
        SG = glenPrinAxis[1, :]
        GA = np.linalg.norm(np.cross(SG - IG, AL - IG))/np.linalg.norm(SG - IG)

        # GH (Humeral head diameter)
        # get humeral head radius from associated humerus
        GH = 2 * self.scapula.shoulder.humerus.radius

        # Acromion index
        if GH:
            self.AI = GA/GH

        # Critical Shoulder Angle (CSA)
        # Adapted by AT (2018-07-18) from Bharath Narayanan (2018)
        # Comment by Pezhman, 05 September 2023
        # We use the ( glenoid inclination + acromion angle ) for CSA
        # The following way of calculating CSA might be removed

        # Vectors connecting IG to SG, and IG to AL
        IGSG = SG - IG
        IGAL = AL - IG
        CSA = rotation_angle.angle_of_rotation_from_vectors(IGAL, IGSG)
        self.CSA = CSA

        ## Posterior Slope Angle and length (PSA & PSL)
        # By AT (2018-08-10)

        # Project AA & AC in PA-IS plane
        AAxy = np.array([AA[0, 0], AA[0, 1], 0]) # Juste take x and y component
        ACxy = np.array([AC[0, 0], AC[0, 1], 0]) # Juste take x and y component
        PSv = ACxy - AAxy # Posterior slope vector
        ISv = np.array([0, 1, 0]) # IS axis
        PSA = rotation_angle.angle_of_rotation_from_vectors(PSv, ISv)
        PSL = np.linalg.norm(PSv)

        self.PSA = PSA
        self.PSL = PSL

        ## Acromial Angle Angle and length (AAA & AAL)
        # By AT (2018-09-13)

        # Vector between scapular origin and AA in the plane PA-IS
        AAv = np.array([AA[0, 0], AA[0, 1], 0])

        # Angle between AAvect and PA axis
        PAv = np.array([1, 0, 0])
        AAA = rotation_angle.angle_of_rotation_from_vectors(AAv, PAv)
        AAL = np.linalg.norm(AAv)

        self.AAA = 180 - AAA
        self.AAL = AAL

        # We might save an image of AI, CSA, PS

        # APA, written by Pezhman , 13th June 2023
        AIxy = np.array([AInf[0, 0], AInf[0, 1], 0])
        AIAAvec = AAxy - AIxy
        APA = rotation_angle.angle_of_rotation_from_vectors(AIAAvec, [0, 1, 0])
        self.APA = APA

        # Acromion angle (AA) by Pezhman, 04 September 2023
        acromionAngle = rotation_angle.angle_of_rotation_from_vectors(IGAL, np.array([0, 1, IG[2]])-IG)
        self.AA = acromionAngle

        return True







