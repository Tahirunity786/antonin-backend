from ShoulderCase.DicomVolumeNormaliser.DicomVolumeNormaliser import DicomVolumeNormaliser
import numpy as np
from utils.Rotations.rotation_angle import angle_of_rotation_from_vectors, axis_of_rotation_from_vectors
from scipy.ndimage import rotate, map_coordinates
from skimage.transform import resize
from utils.Plane.Plane import Plane
from skimage.measure import regionprops
from PIL import Image, ImageOps


class DicomVolumeSlicer(DicomVolumeNormaliser):
    """
    Use a normalised dicom volume to create slices.
    The rescale methods have been extracted form the project of Nathan Donini to produce formatted slices data that
    can be used with the rotator cuff auto segmentation system and with the MuscleMeasurer class.
    """
    def __init__(self, *args):
        self.sliced = []
        self.slicedPixelSpacings = []
        self.slicedPixelCoordinates = []
        self.slicedPlane = []
        self.slicedX = []
        self.slicedY = []
        self.slicedZ = []
        self.pointsInVolume = []
        if len(args) == 1:
            if self.sliced == []:
                self.loadDataFromFolder(args[0])

    def addEmptyBackgroundToSlice(self, height, width, emptyIsZero=True):
        # The background is evenly added around the slice. height and width must be given in pixels.
        sliceHeight = self.sliced.shape[0]
        sliceWidth = self.sliced.shape[1]
        height = np.max((height, sliceHeight))
        width = np.max((width, sliceWidth))
        top = round(height / 2 - sliceHeight / 2)
        left = round(width / 2 - sliceWidth / 2)
        bottom = top + sliceHeight
        right = left + sliceWidth

        newSlice = np.ones((height, width))*np.min(self.sliced)
        newSlice[top:bottom, left:right] = self.sliced
        self.sliced = newSlice

        newSliceX = -np.ones((height, width))
        newSliceX[top:bottom, left:right] = self.slicedX
        self.slicedX = newSliceX

        newSliceY = -np.ones((height, width))
        newSliceY[top:bottom, left:right] = self.slicedY
        self.slicedY = newSliceY

        newSliceZ = -np.ones((height, width))
        newSliceZ[top:bottom, left:right] = self.slicedZ
        self.slicedZ = newSliceZ

        newPointsInVolume = np.zeros((height, width))
        newPointsInVolume[top:bottom, left:right] = self.pointsInVolume
        self.pointsInVolume = newPointsInVolume

    def crop(self, center, height, width):
        # center: point in volume
        # width, height: length in mm
        assert(width>0, "Width must be positive")
        assert(height>0, "Height must be positive")
        centerIndex = self.getPointIndexInSlice(center)

        top = np.max((1, centerIndex[0] - round((height / 2) / self.slicedPixelSpacings[0])))
        bottom = np.min((self.sliced.shape[0], centerIndex[0] + round((height / 2) / self.slicedPixelSpacings[0])))
        left = np.max((1, centerIndex[1] - round((width / 2) / self.slicedPixelSpacings[1])))
        right = np.min((self.sliced.shape[1], centerIndex[1] + round((width / 2) / self.slicedPixelSpacings[1])))

        self.sliced = self.sliced[top:bottom+1, left: right+1]
        self.slicedX = self.slicedX[top:bottom+1, left: right+1]
        self.slicedY = self.slicedY[top:bottom+1, left: right+1]
        self.slicedZ = self.slicedZ[top:bottom+1, left: right+1]
        self.pointsInVolume = self.pointsInVolume[top:bottom+1, left:right+1]

        return top, bottom, left, right

    def getPointIndexInSlice(self, point):
        """
        The current implementation actually returns the point in the slice that is the closest to the given point,
        even if the latter is clearly not in the slice (!).

        The former implementation has been kept but commented out.
        It was close to the solution I (Matthieu Boubat) proposed
        here https://ch.mathworks.com/matlabcentral/answers/524181-how-can-i-retrieve-x-y-coordinates-of-a-sliced-plane-through-a-3d-volume-using-the-obliqueslice-func.
        This solution has been discarded because it failed sometime (did not find point indices).
        This method might be refactored at some point.
        """
        pointIndices = self.getPointIndexInVolume(point)

        coordinatesError = (np.abs(self.slicedX-pointIndices[0]) + np.abs(self.slicedY-pointIndices[1]) + np.abs(self.slicedZ-pointIndices[2]))
        i, j = np.where(coordinatesError == np.min(coordinatesError))[0], np.where(coordinatesError == np.min(coordinatesError))[1]
        assert ((len(i) != 0) or (len(j) != 0), "Point has not been found in slice")
        return np.array([i[0], j[0]])

    def getSlicedPixelCoordinates(self):
        slicePixelIndex = {}
        slicePixelCoordinates = {}

        slicePixelIndex["x"] = np.minimum(self.volume.shape[0]-1, np.maximum(1, np.round(self.slicedX).astype(np.int32)))
        slicePixelCoordinates["x"] = self.getPointsCoordinatesX(slicePixelIndex["x"])
        slicePixelCoordinates["x"][np.logical_not(self.pointsInVolume)] = np.nan

        slicePixelIndex["y"] = np.minimum(self.volume.shape[1]-1, np.maximum(1, np.round(self.slicedY).astype(np.int32)))
        slicePixelCoordinates["y"] = self.getPointsCoordinatesY(slicePixelIndex["y"])
        slicePixelCoordinates["y"][np.logical_not(self.pointsInVolume)] = np.nan

        slicePixelIndex["z"] = np.minimum(self.volume.shape[2]-1, np.maximum(1, np.round(self.slicedZ).astype(np.int32)))
        slicePixelCoordinates["z"] = self.getPointsCoordinatesZ(slicePixelIndex["z"])
        slicePixelCoordinates["z"][np.logical_not(self.pointsInVolume)] = np.nan

        return slicePixelCoordinates

    def measureSlicedPixelSpacings(self):
        """
        Find top, bottom, leftest, and rightest points in the slice for which
        coordinates in the volume can be found. Then measure the top to bottom
        and the leftest to rightest vectors' norms.

        Find the row and the column in the slice that contains the most points
        for which coordinates in the volume can be found.
        """
        longestRowIndex = np.where(np.sum(self.pointsInVolume, 1) == np.max(np.sum(self.pointsInVolume, 1)))
        longestColumnIndex = np.where(np.sum(self.pointsInVolume, 0) == np.max(np.sum(self.pointsInVolume, 0)))

        longestRowElements = np.where(self.pointsInVolume[longestRowIndex[0][0], :])[-1]
        longestColumnElements = np.where(self.pointsInVolume[:, longestColumnIndex[0][0]])[0]

        # Find the leftest point
        leftSliceIndex = np.array([longestRowIndex[0][0], longestRowElements[0]])
        leftVolumeIndex = np.array([round(self.slicedX[leftSliceIndex[0], leftSliceIndex[1]]),
                                    round(self.slicedY[leftSliceIndex[0], leftSliceIndex[1]]),
                                    round(self.slicedZ[leftSliceIndex[0], leftSliceIndex[1]])])
        # Volume boundaries can be exceeded at this point but must be asserted.
        leftVolumeIndex = np.maximum(leftVolumeIndex, [0, 0, 0])
        leftVolumeIndex = np.minimum(leftVolumeIndex, np.array(self.volume.shape)-1)
        leftPoint = self.getPointCoordinates(leftVolumeIndex)

        # Find the rightest point
        rightSliceIndex = [longestRowIndex[0][0], longestRowElements[-1]]
        rightVolumeIndex = np.array([round(self.slicedX[rightSliceIndex[0], rightSliceIndex[1]]),
                                     round(self.slicedY[rightSliceIndex[0], rightSliceIndex[1]]),
                                     round(self.slicedZ[rightSliceIndex[0], rightSliceIndex[1]])])
        # Volume boundaries can be exceeded at this point but must be asserted.
        rightVolumeIndex = np.maximum(rightVolumeIndex, [0, 0, 0])
        rightVolumeIndex = np.minimum(rightVolumeIndex, np.array(self.volume.shape)-1)
        rightPoint = self.getPointCoordinates(rightVolumeIndex)

        # Find the top point
        topSliceIndex = [longestColumnElements[0], longestColumnIndex[-1][0]]
        topVolumeIndex = np.array([round(self.slicedX[topSliceIndex[0], topSliceIndex[1]]),
                                   round(self.slicedY[topSliceIndex[0], topSliceIndex[1]]),
                                   round(self.slicedZ[topSliceIndex[0], topSliceIndex[1]])])
        # Volume boundaries can be exceeded at this point but must be asserted.
        topVolumeIndex = np.maximum(topVolumeIndex, [0, 0, 0])
        topVolumeIndex = np.minimum(topVolumeIndex, np.array(self.volume.shape)-1)
        topPoint = self.getPointCoordinates(topVolumeIndex)

        # Find the bottom point
        bottomSliceIndex = [longestColumnElements[-1], longestColumnIndex[0][0]]
        bottomVolumeIndex = np.array([round(self.slicedX[bottomSliceIndex[0], bottomSliceIndex[1]]),
                                      round(self.slicedY[bottomSliceIndex[0], bottomSliceIndex[1]]),
                                      round(self.slicedZ[bottomSliceIndex[0], bottomSliceIndex[1]])])
        # Volume boundaries can be exceeded at this point but must be asserted.
        bottomVolumeIndex = np.maximum(bottomVolumeIndex, [0, 0, 0])
        bottomVolumeIndex = np.minimum(bottomVolumeIndex, np.array(self.volume.shape)-1)
        bottomPoint = self.getPointCoordinates(bottomVolumeIndex)

        self.slicedPixelSpacings = [np.linalg.norm(rightPoint - leftPoint)/(len(longestRowElements)-1),
                                    np.linalg.norm(bottomPoint - topPoint)/(len(longestColumnElements)-1)]

    def orientSliceUpwardVector(self, point, vector):
        """
        Give a vector in the volume and rotate the slice such that the projection of this vector
        onto the slice is pointing upward.
        """
        pointIndices = self.getPointIndexInSlice(point)
        vectorIndices = self.getPointIndexInSlice(self.slicedPlane.projectOnPlane(point + 10 * vector / np.linalg.norm(vector)).flatten()) - pointIndices
        rotation_angle = angle_of_rotation_from_vectors(np.append(vectorIndices, 0), [-1, 0, 0])
        rotation_angle = axis_of_rotation_from_vectors(np.append(vectorIndices, 0), [-1, 0, 0])[-1]*rotation_angle
        self.sliced = np.array(Image.fromarray(self.sliced).rotate(rotation_angle, expand=True))
        self.slicedX = np.array(Image.fromarray(self.slicedX).rotate(rotation_angle, expand=True))
        self.slicedY = np.array(Image.fromarray(self.slicedY).rotate(rotation_angle, expand=True))
        self.slicedZ = np.array(Image.fromarray(self.slicedZ).rotate(rotation_angle, expand=True))
        self.pointsInVolume = np.array(Image.fromarray(self.pointsInVolume).rotate(rotation_angle, expand=True))
        self.measureSlicedPixelSpacings()

    def rescaleSliceToInt16(self, outOfVolumeValue="sliceMin"):
        # Format used to run muscle measurements
        if outOfVolumeValue == "sliceMin":
            outOfVolumeValue = self.sliced[~np.isnan(self.sliced)].min()
        elif outOfVolumeValue == "0":
            outOfVolumeValue = 0
        self.sliced = np.where(self.pointsInVolume<0.1, outOfVolumeValue, self.sliced)
        self.sliced = convertToInt16(self.sliced)

    def rescaleSliceToUint8(self):
        # Format used by the rotator cuff segmentation system
        HU_interval_png = [-100, 160]
        lin_coef = np.linalg.inv(
            np.array(
            [[HU_interval_png[0], 1],
            [HU_interval_png[1], 1]])
        )@np.array([[0], [255]])
        self.sliced = lin_coef[0,0]*self.sliced + lin_coef[1,0]
        self.sliced = convertToUint8(self.sliced)

    def resize(self, rowsAndColumns):
        """
        rowAndColumns should be a 1x2 array specifying the number of pixels wanted
        in rows an columns. For further information check the documentation of imresize().
        """
        self.sliced = resize(self.sliced, rowsAndColumns)
        self.slicedX = resize(self.slicedX, rowsAndColumns)
        self.slicedY = resize(self.slicedY, rowsAndColumns)
        self.slicedZ = resize(self.slicedZ, rowsAndColumns)
        self.pointsInVolume = resize(np.where(self.pointsInVolume>0.1, True, False), rowsAndColumns, preserve_range=True)
        self.measureSlicedPixelSpacings()

    def slice(self, point, normalVector):

        # The Plane object is used to project points on the slice
        self.slicedPlane = Plane()
        self.slicedPlane.point = point
        self.slicedPlane.normal = normalVector

        pointIndices = self.getPointIndexInVolume(point)

        vectorIndices = self.getPointIndexInVolume(point + 50 * normalVector / np.linalg.norm(normalVector)) - pointIndices

        self.sliced, self.slicedX, self.slicedY, self.slicedZ = self.obliqueSlice(pointIndices, vectorIndices)


        # The following properties are used to manipulate the slice and must be modified the same was the slice is modified.
        self.pointsInVolume = np.where(np.isnan(self.sliced), 0, 1)
        self.sliced = (self.sliced * self.dicomInfo.RescaleSlope) + self.dicomInfo.RescaleIntercept

        self.measureSlicedPixelSpacings()


    def obliqueSlice(self, point, normalVector):

        numRows, numCols, numChannels = self.volume.shape[0], self.volume.shape[1], self.volume.shape[2]

        intialNormVector = [0, 0, 1]

        unitNormalVector = normalVector / np.linalg.norm(normalVector)

        if np.all(intialNormVector ==  unitNormalVector):
            W = unitNormalVector
        else:
            W = np.cross(intialNormVector, unitNormalVector)
            W = W / np.linalg.norm(W)
            W[np.where(np.isnan(W))] = 1e-6

        # Compute angle of rotation in radians
        angle = np.arccos(np.dot(intialNormVector, unitNormalVector))

        # Quaternion rotation matrix
        tQuat = quatMatrix(W, -angle)

        planeSize = 3 * np.max(self.volume.shape)
        numRows1 = planeSize
        numCols1 = planeSize

        [xp, yp, zp] = np.meshgrid(np.arange(round(-numCols1/2), round(numCols1/2)+1),
                                   np.arange(round(-numRows1/2), round(numRows1/2)+1),
                                   0)
        xp, yp, zp = np.squeeze(xp), np.squeeze(yp), np.squeeze(zp)

        xr, yr, zr = rotate_plane_using_transformation_matrix_T(tQuat, xp, yp, zp)

        # Shift input point relative to input volume having origin as center
        shifted_point = np.array([point[0] - round(numCols/2),
                                  point[1] - round(numRows/2),
                                  point[2] - round(numChannels/2)])

        # Find the shortest distance between the plane that passes through input point and origin
        D = -(unitNormalVector[0]*shifted_point[0] +
              unitNormalVector[1]*shifted_point[1] +
              unitNormalVector[2]*shifted_point[2])

        # Translate a plane that passes from origin to input point
        xq = xr - D * unitNormalVector[0] + round(numCols / 2)
        yq = yr - D * unitNormalVector[1] + round(numRows / 2)
        zq = zr - D * unitNormalVector[2] + round(numChannels / 2)

        # Performing interpolation to find the values of the volume on the plane
        mappedCoordinates = map_coordinates(
            np.moveaxis(self.volume,
                        (0, 1, 2),
                        (1, 0, 2)
                        ),
            np.vstack(
                (xq.ravel()[None, :],
                 yq.ravel()[None, :],
                 zq.ravel()[None, :])
            ),
            order=1,
            mode="constant",
            cval=-1e5
        )
        oblique_slice = np.where(mappedCoordinates < -10000, np.nan, mappedCoordinates)
        oblique_slice = oblique_slice.reshape(xq.shape[0], xq.shape[1])

        sliceMaskLimitX = np.logical_and(xq >= 1, xq <= self.volume.shape[1])
        sliceMaskLimitY = np.logical_and(yq >= 1, yq <= self.volume.shape[0])
        sliceMaskLimitZ = np.logical_and(zq >= 1, zq <= self.volume.shape[2])

        sliceMaskLimit = np.logical_and(np.logical_and(sliceMaskLimitX, sliceMaskLimitY), sliceMaskLimitZ)

        B1 = regionprops(sliceMaskLimit.astype(np.int8))
        sliceSize = (round(B1[0].bbox[1]),
                     round(B1[0].bbox[0]),
                     round(B1[0].bbox[3])-round(B1[0].bbox[1]),
                     round(B1[0].bbox[2])-round(B1[0].bbox[0]))
        sliceRows = (sliceSize[1], sliceSize[1]+sliceSize[3]-1)
        sliceCols = (sliceSize[0], sliceSize[0]+sliceSize[2]-1)
        sliceCenter = [(sliceRows[0] + sliceRows[1]) / 2, (sliceCols[0] + sliceCols[1]) / 2]

        croppedSlice = oblique_slice[sliceRows[0]:sliceRows[1] + 1, sliceCols[0]:sliceCols[1] + 1]

        xData = xq[sliceRows[0]:sliceRows[1]+1, sliceCols[0]:sliceCols[1]+1]
        yData = yq[sliceRows[0]:sliceRows[1]+1, sliceCols[0]:sliceCols[1]+1]
        zData = zq[sliceRows[0]:sliceRows[1]+1, sliceCols[0]:sliceCols[1]+1]

        return croppedSlice, xData, yData, zData

def quatMatrix(W, angle):
    aX, aY, aZ = W[0], W[1], W[2]

    angleCos = np.cos(angle)
    angleSin = np.sin(angle)

    t1 = angleCos + aX**2 * (1 - angleCos)
    t2 = aX * aY * (1 - angleCos) - aZ*angleSin
    t3 = aX * aZ * (1 - angleCos) + aY*angleSin
    t4 = aY * aX * (1 - angleCos) + aZ*angleSin
    t5 = angleCos + aY**2 * (1 - angleCos)
    t6 = aY * aZ * (1 - angleCos) - aX * angleSin
    t7 = aZ * aX * (1 - angleCos) - aY * angleSin
    t8 = aZ * aY * (1 - angleCos) + aX * angleSin
    t9 = angleCos + aZ**2 * (1 - angleCos)

    return np.array([[t1, t2, t3, 0],
                     [t4, t5, t6, 0],
                     [t7, t8, t9, 0],
                     [0,  0,  0,  1]])

def rotate_plane_using_transformation_matrix_T(T, x, y, z):
    TT = T.T
    xx = TT[0, 0] * x + TT[1, 0] * y + TT[2, 0] * z + TT[3, 0]
    yy = TT[0, 1] * x + TT[1, 1] * y + TT[2, 1] * z + TT[3, 1]
    zz = TT[0, 2] * x + TT[1, 2] * y + TT[2, 2] * z + TT[3, 2]
    return xx, yy, -zz

def convertToUint8(img):
    img1 = np.where(img > 255, 255, img)
    img2 = np.where(img1 < 0, 0, img1)
    return img2.astype(np.uint8)

def convertToInt16(img):
    img1 = np.where(img > 32767, 32767, img)
    img2 = np.where(img1 < -32768, -32768, img1)
    return img2.astype(np.int16)


