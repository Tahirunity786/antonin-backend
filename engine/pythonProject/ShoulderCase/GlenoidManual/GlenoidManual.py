import os
import numpy as np
from ShoulderCase.loadStl import loadStl
from ShoulderCase.Glenoid.Glenoid import Glenoid

class GlenoidManual(Glenoid):
    """
    To be used with ShoulderAuto data.
    The load() method requires a specific implementation.
    """
    def loadSurface(self):
        """
        Load the points of the glenoid surface from amira dir
        """
        SCase = self.scapula.shoulder.SCase
        SCaseId4C = SCase.id4c

        amiraDir = SCase.dataAmiraPath()
        side = self.scapula.shoulder.side

        # Import glenoid surface points
        fileName = "ExtractedSurface" + SCaseId4C + ".stl"
        fileName = os.path.join(amiraDir, fileName)
        if os.path.isfile(fileName):
            points,faces,*_=loadStl(fileName,1)
            self.surface["points"] = points
            self.surface["meanPoint"] = np.mean(points, axis=0)
            self.surface["faces"] = faces
            return 1
        else:
            return 0
