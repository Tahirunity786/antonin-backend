from ShoulderCase.DicomVolume.DicomVolume import DicomVolume
from ShoulderCase.DicomVolume.readDicomVolume import readDicomVolume
import numpy as np
from math import sqrt
from scipy.ndimage import gaussian_filter
import os
import pydicom

class DicomVolumeForFE(DicomVolume):

    def __init__(self, dicomPath):
        super().__init__()
        self.folder = dicomPath
        self.files = sorted(os.listdir(dicomPath))
        self.slices = [pydicom.dcmread(file) for file in [os.path.join(self.folder, filename) for filename in self.files]]
        self.patientPositionZ = np.array([slice.ImagePositionPatient[2] for slice in self.slices])
        self.patientPositionX = np.array([slice.ImagePositionPatient[0] for slice in self.slices])
        self.patientPositionY = np.array([slice.ImagePositionPatient[1] for slice in self.slices])
        self.dicomSet = readDicomVolume(dicomPath)
        self.setVolume(self.dicomSet["V"])
        self.dicomSet.pop("V")
        self.setSpatial(self.dicomSet)
        self.loadDicomInfoFromFolder(dicomPath)

    def getPixelIndicesOfNodesCoordinates2(self, nodesCoordinates):
        """
        Return an Nx3 array of pixel indices corresponding to the given Nx3 nodes
        spatial coordinates.
        """
        pixelIndices2 = []
        for nodeCoordinates in nodesCoordinates:
            pixelX = self.getPixelIndexOfCoordinateX2(nodeCoordinates[0])
            pixelY = self.getPixelIndexOfCoordinateY2(nodeCoordinates[1])
            pixelZ = self.getPixelIndexOfCoordinateZ2(nodeCoordinates[2])
            pixelIndices2.append([pixelX, pixelY, pixelZ])
        return np.array(pixelIndices2)

    def getPixelIndexOfCoordinateX2(self, xCoordinate):
        """
            Return the pixel x index corresponding to the given x node coordinate
        """
        xIndex = np.round( (xCoordinate) / self.slices[0].PixelSpacing[0])
        return xIndex.astype("int")

    def getPixelIndexOfCoordinateY2(self, yCoordinate):
        """
            Return the pixel y index corresponding to the given y node coordinate
        """
        yIndex = np.round((yCoordinate) / self.slices[0].PixelSpacing[1])
        return yIndex.astype("int")

    def getPixelIndexOfCoordinateZ2(self, zCoordinate):
        """
            Return the pixel z index corresponding to the given z node coordinate
        """
        zCoordinate = zCoordinate + self.slices[0].ImagePositionPatient[2]
        zPositionDiff = np.abs(self.patientPositionZ - zCoordinate)
        zIndex = np.where(zPositionDiff == zPositionDiff.min())[0][0]

        return zIndex.astype("int")

    def filteredHU(self, gaussianKernelStandardDeviation=2):
        """
        Return the 3D matrix of HU values (HU values for every pixel) filtered by
        a gaussian filter. The gaussian kernel standard deviation can be given in
        argument.
        """
        return gaussian_filter(np.moveaxis(self.volume, -1, 0), gaussianKernelStandardDeviation)

    def getFilteredHUAtCoordinates(self, coordinates):
        """
        Return the Nx1 array of filtered HU values for the given Nx3 spatial
        coordinates.
        """
        pixels = self.getPixelIndicesOfNodesCoordinates2(coordinates)
        filteredHU = self.filteredHU()

        # used for debug only, prints on the dicom the pixels of interest in red,
        display_red_nodes = 0
        if (display_red_nodes == 1):
            inputDir = self.folder + "\\"
            HUdebugScript.RenderPixelsRed(pixels, inputDir)

        filteredHUAtCoordinates = np.array([filteredHU[pixel[2]][pixel[0]][pixel[1]] for pixel in pixels])
        return filteredHUAtCoordinates

    def calculateTransitionalMatrix(self):
        return [float(self.slices[0].ImagePositionPatient[0]),
                float(self.slices[0].ImagePositionPatient[1]),
                float(self.slices[0].ImagePositionPatient[2])]
        # WASBEFORE float(self.slices[0].ImagePositionPatient[2]) TRIED TO REVERSE THE VECTOR

    def calculateTransformationMatrix(self):

        # Note: Transformation matrix is wrong according to z direction

        transitionVector = self.calculateTransitionalMatrix()
        # Max Coordinate of the CT scan
        Xmax = transitionVector[0] + (self.slices[0].Rows - 1) * self.slices[0].PixelSpacing[0]
        Ymax = transitionVector[1] + (self.slices[0].Columns - 1) * self.slices[0].PixelSpacing[1]
        Zmax = float(self.slices[-1].ImagePositionPatient[2])  # WASBEFORE .slices[-1]

        P2 = [transitionVector[0], Ymax, transitionVector[2]]
        P3 = [Xmax, transitionVector[1], transitionVector[2]]
        P4 = [transitionVector[0], transitionVector[1], Zmax]

        # Director Vectors for each axis (for Abaqus)
        P21 = [P2[0] - transitionVector[0], P2[1] - transitionVector[1], P2[2] - transitionVector[2]]
        P31 = [P3[0] - transitionVector[0], P3[1] - transitionVector[1], P3[2] - transitionVector[2]]
        P41 = [P4[0] - transitionVector[0], P4[1] - transitionVector[1], P4[2] - transitionVector[2]]

        # Normalize those vectors
        normP21 = sqrt(P21[0] ** 2 + P21[1] ** 2 + P21[2] ** 2)
        normP31 = sqrt(P31[0] ** 2 + P31[1] ** 2 + P31[2] ** 2)
        normP41 = sqrt(P41[0] ** 2 + P41[1] ** 2 + P41[2] ** 2)
        if normP21 != 0:
            unitP21 = [P21[0] / normP21, P21[1] / normP21, P21[2] / normP21]
        else:
            unitP21 = [0, 0, 0]

        if normP31 != 0:
            unitP31 = [P31[0] / normP31, P31[1] / normP31, P31[2] / normP31]
        else:
            unitP31 = [0, 0, 0]

        if normP41 != 0:

            unitP41 = [P41[0] / normP41, P41[1] / normP41, P41[2] / normP41]
        else:
            unitP41 = [0, 0, 0]

        # Transformation Matrix
        P = [unitP21, unitP31, unitP41]
        Pinv = np.linalg.inv(P)  # inverse Matrix

        return Pinv


