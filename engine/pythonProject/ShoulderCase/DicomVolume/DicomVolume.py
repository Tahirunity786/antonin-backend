import numpy as np
from scipy.ndimage import gaussian_filter
import os
import glob
from ShoulderCase.DicomVolume.readDicomVolume import readDicomVolume
import pydicom

class DicomVolume:
    """
    Used to manipulate the data given by dicomreadVolume() and dicominfo().
    Can be used to retrieve a point coordinates given its indices, and vice
    versa.
    """
    def __init__(self):
        self.volume = []
        self.spatial = []
        self.dicomInfo = []
        self.dicomFolderPath = []
    
    def setVolume(self, volume):
        assert len(volume.shape) >=3, "Volume must be a 3D array"
        self.volume = np.squeeze(volume)
        
    def setSpatial(self, spatial):
        requiredDictionaryKeys = np.unique(('PatientPositions',
                                            'PixelSpacings',
                                            'PatientOrientations'))
        assert np.all(np.in1d(list(spatial.keys()), requiredDictionaryKeys)), "Invalid spatial structure"
        self.spatial = spatial
        
    def applyGaussianFilter(self):
        # Apply a gaussian filter to the loaded volume.
        self.volume = gaussian_filter(self.volume, sigma=1, mode="nearest", truncate=2.0)
        
    def assertPathContainsDicomFiles(self, path):
        assert os.path.isdir(path), "Provided argument is not a valid folder path"

        dicomFiles = glob.glob(os.path.join(path, "*.dcm"))
        if len(dicomFiles) == 0:
            dicomFiles = glob.glob(os.path.join(path, ".*.dcm"))
        if len(dicomFiles) == 0:
            dicomFiles = glob.glob(os.path.join(path, "*"))
        assert dicomFiles, "No dicom file found there %s" % path
    
    def getPointCoordinates(self, pointsIndices):
        # the output are the three coordinates in space of the given volume indices.
        # this function is the inverse of obj.getPointIndexInVolume
        coordinatesX = self.getPointsCoordinatesX(pointsIndices[0])
        coordinatesY = self.getPointsCoordinatesY(pointsIndices[1])
        coordinatesZ = self.getPointsCoordinatesZ(pointsIndices[2])
        return np.array([coordinatesX, coordinatesY, coordinatesZ])

    def getPointsCoordinatesX(self, pointsIndicesX):
        coordinatesX = self.spatial["PatientPositions"][0, 0] + \
                       (self.spatial["PixelSpacings"][0, 0] * pointsIndicesX)
        return coordinatesX

    def getPointsCoordinatesY(self, pointsIndicesY):
        coordinatesY = self.spatial["PatientPositions"][0, 1] + \
                       (self.spatial["PixelSpacings"][0, 1] * pointsIndicesY)
        return coordinatesY

    def getPointsCoordinatesZ(self, pointsIndicesZ):
        if np.isscalar(pointsIndicesZ):
            coordinatesZ = self.spatial["PatientPositions"][pointsIndicesZ, 2]
            return coordinatesZ

        pointsLinearIndicesZ = pointsIndicesZ.T.flatten()
        coordinatesZ = self.spatial["PatientPositions"][pointsLinearIndicesZ, 2]
        return coordinatesZ.reshape(pointsIndicesZ.shape)
    
    def getPointIndexInVolume(self, pointsCoordinates):
        # the output are the three indices of the given points in the given volume.
        indicesX = self.getPointsIndicesX(pointsCoordinates[0])
        indicesY = self.getPointsIndicesY(pointsCoordinates[1])
        indicesZ = self.getPointsIndicesZ(pointsCoordinates[2])
        return np.array([indicesX, indicesY, indicesZ])
    
    def getPointsIndicesX(self, pointsCoordinatesX):
        indicesX = round((pointsCoordinatesX - self.spatial["PatientPositions"][0,0])/self.spatial["PixelSpacings"][0,0])
        return indicesX
    
    def getPointsIndicesY(self, pointsCoordinatesY):
        indicesY = round((pointsCoordinatesY - self.spatial["PatientPositions"][0,1])/self.spatial["PixelSpacings"][0,1])
        return indicesY
    
    def getPointsIndicesZ(self, pointsCoordinatesZ):
        pointsLinearCoordinatesZ = pointsCoordinatesZ
        patientPositionsZ = self.spatial["PatientPositions"][:,2]
        
        differenceFromPatientPositionsToPointsCoordinates = patientPositionsZ - pointsLinearCoordinatesZ
        
        pointsLinearIndicesZ = np.argmin(np.abs(differenceFromPatientPositionsToPointsCoordinates), axis=0)
        return pointsLinearIndicesZ
    
    def loadDataFromFolder(self, dicomFolderPath):
        self.assertPathContainsDicomFiles(dicomFolderPath)
        self.dicomFolderPath = dicomFolderPath
        dicomVolume = readDicomVolume(dicomFolderPath)

        self.volume = dicomVolume["V"]
        self.spatial = {"PatientPositions" : dicomVolume["PatientPositions"],
                        "PixelSpacings" : dicomVolume["PixelSpacings"],
                        "PatientOrientations" : dicomVolume["PatientOrientations"]}

        self.loadDicomInfoFromFolder(dicomFolderPath)
    
    def loadDicomInfoFromFolder(self, dicomFolderPath):
        self.assertPathContainsDicomFiles(dicomFolderPath)
        dicomFiles = glob.glob(os.path.join(dicomFolderPath, "*.dcm"))
        if len(dicomFiles) == 0:
            dicomFiles = glob.glob(os.path.join(dicomFolderPath, ".*.dcm"))
        if len(dicomFiles) == 0:
            dicomFiles = glob.glob(os.path.join(dicomFolderPath, "*"))
        self.dicomInfo = pydicom.dcmread(dicomFiles[0])
    
    def setVolumeToSubvolume(self, center, x, y, z):
        #
        # find boundaries
        volumeSize = self.volume.shape
        minXYZ = self.getPointIndexInVolume(center - np.array([x/2, y/2, z/2]))
        maxXYZ = self.getPointIndexInVolume(center + np.array([x/2, y/2, z/2]))
        left = np.max(1, minXYZ[0])
        right = np.min(volumeSize[0], maxXYZ[0])
        front = np.max(1, minXYZ[1])
        rear = np.min(volumeSize[1], maxXYZ[1])
        bottom = np.max(1, minXYZ[2])
        top = np.min(volumeSize[2], maxXYZ[2])
        
        #set subvolume
        self.volume = self.volume[left:right+1, front:rear+1, bottom:top+1]
        
        #update spatial
        self.spatial["PixelSpacings"] = self.spatial["PixelSpacings"][bottom:top+1, :]
        self.spatial["PatientPositions"] = self.spatial["PatientPositions"][bottom:top+1, :]
        self.spatial["PatientPositions"][:, 0] = self.spatial["PatientPositions"][:, 0] + \
                                            left*self.spatial["PixelSpacings"][:, 0]
        self.spatial["PatientPositions"][:, 1] = self.spatial["PatientPositions"][:, 1] + \
                                            front*self.spatial["PixelSpacings"][:, 1]
        self.spatial["PatientOrientations"] = self.spatial["PatientOrientations"][:,:,bottom:top+1]