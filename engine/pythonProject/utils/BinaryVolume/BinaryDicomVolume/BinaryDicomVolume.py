from utils.BinaryVolume.BinaryDicomVolume.readDicomVolume import \
readDicomVolume, readSlices, readPatientPositions, readPixelSpacings
import numpy as np
from utils.BinaryVolume.BinaryVolume.BinaryVolume import BinaryVolume

class BinaryDicomVolume(BinaryVolume):
    def __init__(self, dicomFolderPath):
        volume = readDicomVolume(readSlices(dicomFolderPath))
        pixelSpacings = readPixelSpacings(readSlices(dicomFolderPath))
        patientPositions = readPatientPositions(readSlices(dicomFolderPath))
        volume = np.squeeze(volume)
        resolution = getSmallestResolutionDividedBy2(pixelSpacings,
                                                     patientPositions)
        volume = normaliseDicomVolume(volume,resolution)

        BinaryVolume.__init__(volume.shape)
        self.dicomVolume = volume
        self.setResolution(resolution)

    def filterHUBelow(self, threshold):
        self.setVolume(self.dicomVolume < threshold)
        return self

    def filterHUBetween(self, threshold):
        self.setVolume(np.logical_and((self.dicomVolume > threshold[0]),
                                      (self.dicomVolume < threshold[1])))

    def filterHUOver(self, threshold):
        self.setVolume(self.dicomVolume, threshold)
        return self

def getSmallestResolutionDividedBy2(pixelSpacings, patientPositions):
    resolution = [0, 0, 0]
    resolution[0] = np.min(pixelSpacings[:, 0])/2
    resolution[1] = np.min(pixelSpacings[:, 1])/2
    resolution[2] = patientPositions(-1, 2) - patientPositions(0, 2)/(2*patientPositions.shape[0])
    return resolution

def normaliseDicomVolume(volume,newResolution):
    newSize = np.abs(np.array(volume.shape)/newResolution)
    volume = np.array(PIL.Image.fromarray(volume).resize(size=newSize))
    return volume
