from utils.BinaryVolume.BinaryVolume.BinaryVolume import BinaryVolume

class BinaryRectangle(BinaryVolume):
    def __init__(self, rectangleSize, center=np.array([0,0,0]):
        BinaryVolume.__init__(self, rectangleSize)
        self.translate(center)
        self.fillVolume()
        self.resize(self.size + 2*self.resolution)
