import numpy as np

class MaskSubdividerWithSquareGrid:

    def __init__(self, mask):
        self.mask = mask
        self.xGrid, self.yGrid = np.meshgrid(np.arange(mask.shape[0]), np.arange(mask.shape[1]))
        maskPixelY, maskPixelX = np.where(mask)

        self.maskLimit.top = np.min(maskPixelY)
        self.maskLimit.bottom = np.max(maskPixelY)
        self.maskLimit.left = np.min(maskPixelX)
        self.maskLimit.right = np.max(maskPixelX)

    def setXResolutionInPixels(self, resolutionInPixels):
        resolutionInPixels = np.round(resolutionInPixels)
        self.xResolution = np.min((self.maskLimit.right - self.maskLimit.left) + 1, np.max(1, resolutionInPixels))

    def setXResolutionInMm(self, resolutionInMm, pixelSizeX):
        self.setXResolutionInPixels(resolutionInMm / pixelSizeX)

    def setYResolutionInPixels(self, resolutionInPixels):
        resolutionInPixels = np.round(resolutionInPixels)
        self.yResolution = np.min((self.maskLimit.bottom - self.maskLimit.top) + 1, np.max(1, resolutionInPixels))

    def setYResolutionInMm(self, resolutionInMm, pixelSizeY):
        self.setYResolutionInPixels(resolutionInMm / pixelSizeY)

    def getDivition(self, top, bottom, left, right):
        division = (self.xGrid >= left) & (self.xGrid <= right) & (self.yGrid >= top) & (self.yGrid <= bottom)
        return division

    def getMaskSubdivisions(self):
        maskSubdivisions = []
        for x in np.arange(self.maskLimit.left, self.xResolution, self.maskLimit.right):
            for y in np.arange(self.maskLimit.top, self.yResolution, self.maskLimit.bottom):
                maskSubdivisions.append(self.mask & self.getDivition(y, (y + self.yResolution-1), x, (x + self.xResolution-1)))

        maskSubdivisions = np.array(maskSubdivisions)
        return self.removeEmptySubdivisions(maskSubdivisions)

    def removeEmptySubdivisions(selfself, subdivisions):
        subdivisionsToRemove = []
        for i in range(subdivisions.shape[2]):
            if np.logical_not(np.any(subdivisions[:, :, i])):
                subdivisionsToRemove.append(i)

        subdivisions = np.delete(subdivisions, subdivisionsToRemove, axis=2)
        return subdivisions



