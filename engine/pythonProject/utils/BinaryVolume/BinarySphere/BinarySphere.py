import numpy as np
from utils.BinaryVolume.BinaryVolume.BinaryVolume import BinaryVolume

class BinarySphere(BinaryVolume):
    def __init__(self, radius, center=np.array([0,0,0])):
        BinaryVolume.__init__(self, np.ones((1, 3))*2*radius)
        self.translate(center)
        squaredDistanceToCenter = self.getSquaredDistanceToPointXYZ(center)
        self.volume = squaredDistanceToCenter <= radius**2
        self.resize(self.size + 2*self.resolution)
