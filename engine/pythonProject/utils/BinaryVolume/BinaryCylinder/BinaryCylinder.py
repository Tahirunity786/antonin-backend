from utils.BinaryVolume.BinaryVolume.BinaryVolume import BinaryVolume
import numpy as np

class BinaryCylinder(BinaryVolume):
    def __init__(self, radius, length, center=np.array([0,0,0])):
        BinaryVolume.__init__(np.array(2*radius, 2*radius, length))
        self.translate(center)

        squaredDistanceToZAxis = self.getSquaredDistanceToPointXY(center)
        cylinder = squaredDistanceToZAxis <= radius**2

        cylinderStartPoint = self.getIndicesOfPointInVolume(center - np.array([0 0 length/2]))
        cylinderEndPoint = self.getIndicesOfPointInVolume(center + np.array([0 0 length/2]))
        cylinder[:, :, 0:cylinderStartPoint[2]] = 0
        cylinder[:, :, cylinderEndPoint[2]:-1] = 0
        self.volume = cylinder

        self.resize(self.size + 2*self.resolution)
