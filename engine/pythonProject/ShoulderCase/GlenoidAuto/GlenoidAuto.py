import os
import numpy as np
from ShoulderCase.loadPly import loadPly
from ShoulderCase.Glenoid.Glenoid import Glenoid

class GlenoidAuto(Glenoid):
    """
    To be used with ShoulderAuto data.
    The load() method requires a specific implementation.
    """
    def loadSurface(self):
        """
        LOADAUTO Load genoid surface from auto segmentation
        Load the points of the glenoid surface from matlab dir
        """
        SCase = self.scapula.shoulder.SCase
        SCaseId4C = SCase.id4c

        pythonDir = SCase.dataPythonPath()
        side = self.scapula.shoulder.side
        # Import glenoid surface points
        fileName = "glenoidSurfaceAuto" + side + ".ply"
        fileName = os.path.join(pythonDir, fileName)
        if os.path.isfile(fileName):
            points,faces=loadPly(fileName)
            self.surface["points"] = points
            self.surface["meanPoint"] = np.mean(points, axis=0)
            self.surface["faces"] = faces
            return 1
        else:
            return 0
