import numpy as np
from utils.Bwmorph.bwmorph import bwmorph_remove

class MaskSubdivider:

    def __init__(self, mask):
        self.mask =mask
        # default center is the middle of the mask
        self.center = np.round(self.mask.shape / 2)
        # 10° slices
        self.numberOfAngularDivisions = 36;
        self.numberOfRadialDivisions = 10;
        self.center = []
        self.mask = []

        self.sinusGrid = []
        self.cosineGrid = []
        self.angleGrid = []
        self.radiusGrid = []

    def setNumberOfAngularDivisions(self, value):
        self.numberOfAngularDivisions = value

    def setNumberOfRadialDivisions(self, value):
        self.numberOfAngularDivisions = value

    def setCenter(self, centerIndices):
        self.center = centerIndices

    def set_center(self, value):
        self.center = value
        self.updateGrids()

    def updateGrids(self):
        xGrid, yGrid = np.meshgrid(np.arang(self.mask.shape[0], self.mask.shape[1]))
        xGrid = xGrid - self.center[1]
        yGrid = yGrid - self.center[0]
        self.radiusGrid = np.sqrt(xGrid**2 + yGrid**2)
        self.sinusGrid = yGrid / self.radiusGrid
        self.cosineGrid = xGrid / self.radiusGrid
        partialAngleGrid = np.arccos(self.cosineGrid * np.pi / 180) * 180 / np.pi
        self.angleGrid = (360 - partialAngleGrid)*(self.sinusGrid < 0) + partialAngleGrid*(self.sinusGrid >= 0)

    def getAngularDivision(self, lowerAngle, upperAngle):
        lowerAngle = lowerAngle % 360
        upperAngle = upperAngle % 360

        if lowerAngle > upperAngle:
            # zero angle included in the subdivision
            output = (self.angleGrid > lowerAngle) | (self.angleGrid < upperAngle)
        else:
            output = (self.angleGrid > lowerAngle) & (self.angleGrid < upperAngle)
        return output

    def getRadialDivision(self, minRadius, maxRadius):
        return (self.radiusGrid > minRadius) & (self.radiusGrid < maxRadius)

    def getMaskSubdivisions(self):
        maskSubdivisions = []

        minRadius, maxRadius = self.getMinMaxGridValueInMask(self.radiusGrid)

        radialIncrement = (maxRadius - minRadius) / self.numberOfRadialDivisions

        for radius in np.range(minRadius, maxRadius, radialIncrement):
            radialDivision = self.getRadialDivision(radius, radius + radialIncrement)

            lowerAngle, upperAngle = self.getMaskBoundingAngles()
            angularIncrement = (upperAngle - lowerAngle) / self.numberOfAngularDivisions

            for angle in np.arange(lowerAngle, upperAngle, angularIncrement):
                angularDivision = self.getAngularDivision(angle, angle + angularIncrement)
                maskSubdivisions.append(np.logical_and(np.logical_and(self.mask, radialDivision), angularDivision))

        output = np.array(maskSubdivisions)
        return self.removeEmptySubdivisions(maskSubdivisions)

    def getMinMaxGridValueInMask(self, inputGrid):
        gridInMask = self.mask * inputGrid
        gridInMask[np.logical_not(self.mask)] = np.nan
        minOutput = np.min(gridInMask)
        maxOutput = np.max(gridInMask)
        return minOutput, maxOutput

    def getMaskBoundingAngles(self):
        """
        If the zero angle is contained in the mask the simpler getMinMaxGridValueInMask() function fails at
        returning the angular boundaries of the mask (i.e. it will return approximately [0 360] which are the true min
        and max angular values found in mask).
        The present algorithm is a bit more computationally intensive but will return the actual angle boundaries.
        The idea is that the angle values of the elements of the mask are splitted into two groups around the 0° (360°) angle.
        We need to compute upperAngle and lowerAngle such that:
        lowerAngle <= highAnglesValueInMask <= 360 & 0 <= lowAnglesValueInMask <= upperAngle
        """
        angleInMask = self.mask * self.angleGrid
        angleInMask[np.logical_not(self.mask)] = np.nan
        allAnglesValueInMask = np.unique(angleInMask)

        allAnglesDifferenceInMask = allAnglesValueInMask[1:] - allAnglesValueInMask[0: -2]
        allAnglesDifferenceInMaskSTD = np.std(allAnglesDifferenceInMask)
        bigAnglesDifference = np.abs(allAnglesDifferenceInMask - np.mean(allAnglesDifferenceInMask)) >  3*allAnglesDifferenceInMaskSTD
        # for non anular masks including the zero angle, the angle values in the mask are splitted into two groups.
        # The boundaries of these groups are the bounding angles looked for.
        # If there's one clear outlier in the angles difference, this is probably the where the group split.
        if np.sum(bigAnglesDifference) == 1:
            splittingIndex = np.where(bigAnglesDifference)
            # upperAngle is incremented by modulo 360° so that lowerAngle < upperAngle
            upperAngle = allAnglesValueInMask[splittingIndex] + 360
            lowerAngle = allAnglesValueInMask[splittingIndex + 1]
        else:
            lowerAngle = np.min(allAnglesValueInMask, axis=0)
            upperAngle = np.max(allAnglesValueInMask, axis=0)

        # +- 1° added to the boundaries to avoid subdivisions with only 1 pixel
        lowerAngle = lowerAngle - 1
        upperAngle = upperAngle + 1

        return lowerAngle, upperAngle

    def removeEmptySubdivisions(self, subdivisionsArray):
        subdivisionsToRemove = []
        for i in range(subdivisionsArray.shape[2]):
            if np.logical_not(np.any(subdivisionsArray[:,:,i])):
                subdivisionsToRemove.append(i)

        subdivisionsArray = np.array(subdivisionsArray)
        subdivisionsArray = np.delete(subdivisionsArray, subdivisionsToRemove, axis=2)

        return subdivisionsArray

    def getMaskSubdivisionsStacked(self):
        maskSubdivisions = self.getMaskSubdivisions()
        maskSubdivisionsStacked = np.zeros(self.mask.shape)
        for i in np.arange(maskSubdivisions.shape[2]):
            maskSubdivisionsStacked = (maskSubdivisionsStacked) | \
                                      (maskSubdivisionsStacked[:, :, i] - bwmorph_remove(maskSubdivisions[:,:,i]))

        return maskSubdivisionsStacked






























