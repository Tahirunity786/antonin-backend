import numpy as np
from utils.BinaryVolume.BinarySphere import BinarySphere
from ShoulderCase.fitSphere import fitSphere

class Sphere:
    """
    Simple sphere object.
    Used to be fitted to glenoid points.
    This class could be used elsewhere and the constructor
    access might be widen.
    """
    def __init__(self, *args):
        self.center = []
        self.radius = []
        self.residuals = []
        self.R2 = []
        self.RMSE = []
        if len(args) == 2:
            self.center = args[0]
            self.radius = args[1]

    def fitTo(self, points):
        self.center, self.radius, self.residuals, self.R2 = fitSphere(points)
        self.center = self.center.reshape(-1, 1)
        self.RMSE = np.linalg.norm(self.residuals)/np.sqrt(len(self.residuals))

    def isempty_(self):
        return bool(len(['' for x in self.__dict__.values() if not x]))

    def exportPly(self, filename):
        sphereToExport = BinarySphere.BinarySphere(np.round(self.radius),
                                                   np.round(self.center))
        sphereToExport.exportPly(sphereToExport, filename)
