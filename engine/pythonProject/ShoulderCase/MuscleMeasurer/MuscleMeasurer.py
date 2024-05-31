import numpy as np
import cv2
from skimage import measure

class MuscleMeasurer:
    """
    Used to perfom PCSA and degeneration measurements of a Muscle object.
    Slices and mask images are expected to be found at some specific places and have specific names.
    This class is the results of extracting measurements methods from the project of Nathan Donini.
    """
    def __init__(self, segmentationMask, sliceForMeasurement, slicePixelSpacings):
        self.segmentationMask = segmentationMask
        self.sliceForMeasurement = sliceForMeasurement
        self.segmentationDensity = self.getNormalizedImage(sliceForMeasurement)*segmentationMask
        self.slicePixelSpacings = slicePixelSpacings

        self.rangeHU = [-1000, 1000]
        self.muscleThreshold = 0
        self.fatThreshold = 30
        self.osteochondromaThreshold = 166

    def getNormalizedImage(self, image):
        image[image < self.rangeHU[0]] = self.rangeHU[0]
        image[image > self.rangeHU[1]] = self.rangeHU[1]
        return (image-self.rangeHU[0])/(self.rangeHU[1]-self.rangeHU[0])

    def normalizeThresholds(self):
        self.muscleThreshold =  ((self.muscleThreshold - 1)-self.rangeHU[0])/(self.rangeHU[1]-self.rangeHU[0])
        self.fatThreshold = ((self.fatThreshold - 1)-self.rangeHU[1])/(self.rangeHU[2]-self.rangeHU[1])
        self.osteochondromaThreshold = ((self.osteochondromaThreshold - 1)-self.rangeHU[0])/(self.rangeHU[1]-self.rangeHU[0])

    def getPCSA(self):
        pixelSurface = (self.slicePixelSpacings[0] * self.slicePixelSpacings[1]) / 100 # cm ^ 2
        return pixelSurface*np.sum(self.segmentationMask)

    def getRatioAtrophy(self):
        return self.getRatioAreas(self.getAreaAtrophy(), self.segmentationMask)

    def getRatioFat(self):
        return self.getRatioAreas(self.getAreaFar(), self.segmentationMask)

    def getRatioOsteochondroma(self):
        return self.getRatioAreas(self.getAreaOsteochondroma(), self.segmentationMask)

    def getRatioDegeneration(self):
        return self.getRatioAtrophy() + self.getRatioFat() + self.getRatioOsteochondroma()

    def getRatioAreas(self, partialArea, totalArea):
        return np.sum(partialArea)/np.sum(totalArea)

    def getAreaMuscle(self):
        # The area "Muscle" is the area of the segmented image that is not atrophied.
        # Looking for a better name.
        areaMuscle = self.segmentationDensity > self.muscleThreshold
        cv2.floodFill(areaMuscle, None, (0, 0), 1)
        # Keep biggest island only
        islands = measure.label(areaMuscle)
        areaMuscle = np.zeros(shape=areaMuscle.shape)
        if np.sum(islands) != 0:
            islandsSizes = [np.count_nonzero(islands == i) for i in range(1, np.max(islands)+1)]
            biggestIslandIndex = np.argmax(islandsSizes)+1
            areaMuscle[areaMuscle == biggestIslandIndex] = True
        return areaMuscle

    def getAreaAtrophy(self):
        return np.logical_and(self.segmentationMask, np.logical_not(self.getAreaMuscle()))

    def getAreaFat(self):
        muscleImage = self.segmentationDensity*self.getAreaMuscle
        areaFat = np.logical_and(self.getAreaMuscle(), np.logical_not(muscleImage > self.fatThreshold))
        return areaFat

    def getAreaOsteochondroma(self):
        muscleImage = self.segmentationDensity*self.getAreaMuscle
        areaOsteochondroma = muscleImage > self.osteochondromaThreshold
        areaOsteochondroma = cv2.floodFill(areaOsteochondroma, None, (0, 0), 1)
        return areaOsteochondroma













