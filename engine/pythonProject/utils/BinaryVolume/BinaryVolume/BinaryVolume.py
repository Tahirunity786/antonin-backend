import numpy as np
from PIL import Image
from utils.BinaryVolume.plyWrite import plyWrite
from scipy.spatial import Delaunay
#from scipy.ndimage import rotate
from scipy.spatial.transform import Rotation as R
import visvis as vv


class BinaryVolume:
    def __init__(self, *args):
        self.origin = np.array([-50, -50, -50])
        self.size = np.array([100, 100, 100])
        self.resolution = np.array([1, 1, 1])
        self.volume = np.zeros((100, 100, 100))
        if len(args) == 1:
            self.volume = np.zeros(args[0])
            self.origin = -self.size/2

    def setVolume(self, value):
        self.volume = value
        self.size = np.array(self.volume.shape)*self.resolution

    def setResolution(self, value):
        assert np.all(value) > 0
        self.resolution = value
        self.size = np.array(self.volume.shape)*self.resolution

    def addBoundingBox(self):
        self.volume[:, 0, 0] = True
        self.volume[:, -1, 0] = True
        self.volume[:, 0, -1] = True
        self.volume[:, -1, -1] = True

        self.volume[0, :, 0] = True
        self.volume[-1, :, 0] = True
        self.volume[0, :, -1] = True
        self.volume[-1, :, -1] = True

        self.volume[0, 0, :] = True
        self.volume[-1, 0, :] = True
        self.volume[0, -1, :] = True
        self.volume[-1, -1, :] = True
        return self

    def changeResolution(self, newResolution):
        assert np.all(newResolution > 0)
        if np.all(newResolution == self.resolution):
            return self
        self.volume = Image.fromarray(self.volume).resize(size=(np.ceil(self.size/newResolution)))
        self.resolution = newResolution
        return self

    def clearVolume(self):
        self.volume = np.zeros(self.volume.shape)
        return self

    def containsVolume(self, binaryVolume):
        boundaryMin = self.origin
        boundaryMax = self.origin + self.size
        testedVolumeMin = binaryVolume.origin
        testedVolumeMax = binaryVolume.origin + binaryVolume.size

        if not np.all(boundaryMin <= testedVolumeMin):
            return False

        if not np.all(boundaryMax >= testedVolumeMax):
            return False

        return True

    def copy(self):
        copiedVolume = BinaryVolume()
        copiedVolume.setResolution(self.resolution)
        copiedVolume.setVolume(self.Volume)
        copiedVolume.translate(self.origin-copiedVolume.origin)

        return copiedVolume

    def difference(self, volumeToSubstract):
        output = self.copy()
        output.volume = np.logical_and(output.volume,
                        np.logical_not(volumeToSubstract.volume))
        return output

    def exportPly(self, filename):
        volumeToExport = bwmorph(self.volume)
        X, Y, Z = np.unravel_index(shape=volumeToExport.shape,
                                   indices=volumeToExport != 0, order='C') # order="F"?
        X = X*self.resolution[0] + self.origin[0]
        Y = Y*self.resolution[1] + self.origin[1]
        Z = Z*self.resolution[2] + self.origin[2]
        points = np.concatenate([X, Y, Z], axis=1)
        plyWrite(filename, points)

    def fillVolume(self):
        self.volume = np.ones(self.volume.shape)
        return self

    def getIndicesOfPointInVolume(self, point):
        return np.round((point - self.origin)/self.resolution) + np.array([1,1,1])

    def getSmallestCommonBoundingVolume(self, comparedVolume):
        commonOrigin = np.minimum(self.origin, comparedVolume.origin)
        commonDiagonal = np.maximum((self.origin + self.size),
                                    (comparedVolume.origin + comparedVolume.size))
        boundingVolume = BinaryVolume(np.ceil(commonDiagonal-commonOrigin))
        boundingVolume.setOrigin(commonOrigin)
        return boundingVolume

    def getSquaredDistanceToPointXY(self, origin):
        X, Z, Y = np.meshgrid(np.arange(self.volume[0], dtype=np.int32),
                              np.arange(self.volume[1], dtype=np.int32),
                              np.arange(self.volume[2], dtype=np.int32),
                              indexing='xy')
        point = self.getIndicesOfPointInVolume(point)

        X = (X - point[0]) * self.resolution[0];
        Y = (Y - point[1]) * self.resolution[1];

        return X**2 + Y**2

    def getSquaredDistanceToPointXYZ(self, point):
        X, Z, Y = np.meshgrid(np.arange(self.volume[0], dtype=np.int32),
                              np.arange(self.volume[1], dtype=np.int32),
                              np.arange(self.volume[2], dtype=np.int32),
                              indexing='xy')
        point = self.getIndicesOfPointInVolume(point)

        X = (X - point[0]) * self.resolution[0]
        Y = (Y - point[1]) * self.resolution[1]
        Z = (Z - point[2]) * self.resolution[2]

        return X**2 + Y**2 + Z**2

    def getTriangulation(self):
        volume = bwmorph(self.volume)
        X, Y, Z = np.unravel_index(shape=volume.shape,
                                   indices=volume != 0, order='C')
        return Delaunay(np.concatenate([X, Y, Z], axis=1)) # Delaunay.simplices

    def insertVolume(self, insertedVolume):
        copiedInsertedVolume = insertedVolume.copy()
        startingResolution = self.resolution
        lowestResolution  = np.minimum(self.resolution, copiedInsertedVolume.resolution)
        self.changeResolution(lowestResolution)
        copiedInsertedVolume.changeResolution(lowestResolution)

        insertionPoint = self.getIndicesOfPointInVolume(copiedInsertedVolume.origin)
        insertionPointOpposite = (insertionPoint + \
                                  copiedInsertedVolume.volume.shape - np.array([1,1,1]))
        insertionPoint = np.minimum(self.colume.shape,
                                    np.maximum(np.array([1,1,1]), insertionPoint))
        insertionPointOpposite = np.minimum(self.volume.shape,
                                 np.maximum(np.array([1,1,1], insertionPointOpposite)))
        if np.any(insertionPoint == insertionPointOpposite):
            return
        extractionPoint = copiedInsertedVolume.getIndicesOfPointInVolume(\
                self.origin + self.resolution*(insertionPoint-np.array([1,1,1])))
        extractionPointOpposite = extractionPoint + (insertionPointOpposite - insertionPoint)
        extractionPoint = np.minimum(copiedInsertedVolume.volume.shape,
                                    np.maximum(np.array([1,1,1], extractionPoint)))
        extractionPointOpposite = np.minimum(copiedInsertedVolume.volume.shape,
                                  np.maximum(np.array([1,1,1]), extractionPointOpposite))
        insertionPointOpposite = insertionPoint + (extractionPointOpposite - extractionPoint)
        subvolumeToInsert = copiedInsertedVolume.volume(\
                        np.arange(extractionPoint[0],extractionPointOpposite[0]),
                        np.arange(extractionPoint[1],extractionPointOpposite[1]),
                        np.arange(extractionPoint[2],extractionPointOpposite[2]))
        self.volume[np.arange(insertionPoint[0], insertionPointOpposite[0]),
                    np.arange(insertionPoint[1], insertionPointOpposite[1]),
                    np.arange(insertionPoint[2], insertionPointOpposite[2])] = \
                    subvolumeToInsert

        self.changeResolution(startingResolution)
        return self

    def intersection(self, volumeToIntersect):
        output = self.copy()
        output.volume = np.logical_and(output.volume, volumeToIntersect.volume)
        return output

    def isempty(self):
        return (self.volume == 0).size == 0

    def minus(self, volumeSubstracted):
        currentVolumeForDifference = BinaryVolume.insertVolume(\
                        self.getSmallestCommonBoundingVolume(volumeSubstracted),
                        self)
        substractedVolumeForDifference = BinaryVolume.insertVolume(\
                        self.getSmallestCommonBoundingVolume(volumeSubstracted),
                        volumeSubstracted)
        return BinaryVolume.difference(currentVolumeForDifference,
                                       substractedVolumeForDifference)

    def mtimes(self, volumeCompared):
        currentVolumeForIntersection = BinaryVolume.insertVolume(\
                    self.getSmallestCommonBoundingVolume(volumeCompared), self)
        comparedVolumeForIntersection = BinaryVolume.insertVolume(\
                    self.getSmallestCommonBoundingVolume(volumeCompared), volumeCompared)
        return BinaryVolume.intersection(currentVolumeForIntersection,
                                         comparedVolumeForIntersection)

    def plus(self, volumeAdded):
        currentVolumeForUnion = BinaryVolume.insertVolume(\
                        self.getSmallestCommonBoundingVolume(volumeAdded), self)
        addedVolumeForUnion = BinaryVolume.insertVolume(\
                        self.getSmallestCommonBoundingVolume(volumeAdded), volumeAdded)
        return BinaryVolume.union(currentVolumeForUnion, addedVolumeForUnion)

    def removeEmptySpace(self):
        X, Y, Z = np.unravel_index(shape=self.volume.shape,
                                   indices=self.volume != 0, order='C')

        if np.concatenate([X, Y, Z], axis=1).size == 0:
            lowestPoint = np.array([0, 0, 0])
            opposite = np.array([1, 1, 1])
        else:
            lowestPoint = np.min(np.concatenate([X, Y, Z], axis=1), axis=1)
            opposite = np.max(np.concatenate([X, Y, Z], axis=1), axis=1)

        resizedVolume = BinaryVolume(opposite-lowestPoint)
        resizedVolume.setOrigin(self.origin + lowestPoint*self.resolution)
        resizedVolume.insertedVolume(self)

        self.volume = resizedVolume.volume
        self.origin = resizedVolume.origin

        return self

    def resize(self, newSize):
        currentCenter = self.origin + self.size/2
        resizedEmptyVolume = BinaryVolume(newSize)
        resizedEmptyVolume.translate(currentCenter)
        resizedVolume = resizedEmptyVolume.insertVolume(self)

        self.setVolume(resizedVolume.volume)
        self.setOrigin(resizedVolume.origin)
        self.setResolution(resizedVolume.resolution)

        return self

    def rotate(self, rotationVector):
        rotationVector[3] = rotationVector[3]*180/np.pi

        startingResolution = self.resolution
        self.setResolution([1, 1, 1])
        #self.volume = rotate(self.volume, rottationVector[3],
        #                    rottaionVector[:3], mode="nearest")
        rotationVector = rotationVector / np.linalg.norm(rotationVector)  # normalize the rotation vector first
        rot = R.from_rotvec(rotationVector[3] * rotationVector, degrees=True)
        self.volume = rot.apply(self.volume)
        self.setResolution(startingResolution)
        return self

    def setOrigin(self, newOrigin):
        self.origin = newOrigin

    def setResolution(self, newResolution):
        self.resolution = newResolution

    def setVolume(self, volume):
        self.volume = volume

    def show(self):
        app = vv.use()
        vv.clf()
        # create volume
        vol = vv.aVolume(size=64)
        # set labels
        vv.xlabel('x axis')
        vv.ylabel('y axis')
        vv.zlabel('z axis')
        # show
        t = vv.volshow(self.volume, renderStyle='mip')
        # Get axes and set camera to orthographic mode (with a field of view of 70)
        a = vv.gca()
        a.camera.fov = 45
        # Create colormap editor wibject.
        vv.ColormapEditor(a)
        # Start app
        app.Run()

    def translate(self, translation):
        self.origin = self.origin + translation
        return self

    def uminus(self):
        output = self.copy()
        output.volume = np.logical_not(output.volume)
        output.resize(output.size + 2*output.resolution)
        return output

    def union(self, volumeToUnite):
        output = self.copy()
        output.volume = np.logical_or(output.volume, volumeToUnite.volume)
        return output








def bwmorph(input_matrix):
    output_matrix = input_matrix.copy()
    # Change. Ensure single channel
    if len(output_matrix.shape) == 3:
        output_matrix = output_matrix[:, :, 0]
    nRows,nCols = output_matrix.shape # Change
    orig = output_matrix.copy() # Need another one for checking
    for indexRow in range(0,nRows):
        for indexCol in range(0,nCols):
            center_pixel = [indexRow,indexCol]
            neighbor_array = neighbors(orig, center_pixel) # Change to use unmodified image
            if np.all(neighbor_array): # Change
                output_matrix[indexRow,indexCol] = 0

    return output_matrix
