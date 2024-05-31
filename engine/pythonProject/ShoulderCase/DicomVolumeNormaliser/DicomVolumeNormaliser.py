from ShoulderCase.DicomVolume.DicomVolume import DicomVolume
import numpy as np
import PIL
import skimage
from scipy.ndimage import zoom

class DicomVolumeNormaliser(DicomVolume):
    """
    Used to normalise dicom volumes spacings.
    Dicom volume are not necessarily isotrope, often slices spacing
    is not equal to the slices' pixel spacing.
    This class normalise the spacings in the three directions
    based on the smallest spacing found divided by a chosen
    factor.
    The higher the factor, the more time will take the computation
    of the normalisation.

    Once volumic data are loaded with DicomeVolume methods, the
    volume normalisation is achieved in two steps:

    divisionFactor = 2;   % 2 has been a good enough tradeoff
                            % while testing this class
    loadedDicomVolume.setMinimalResolutionDividedBy(divisionFactor);
    loadedDicomVolume.normaliseVolume();
    """
    def __init__(self, *args):
        super().__init__()
        self.resolution = []
        if len(args) == 1:
            if self.volume == []:
                self.loadDataFromFolder(args[0])

    def setMinimalResolutionDividedBy(self, divisionFactor):
        zSpacing  = self.spatial["PatientPositions"][1:, 2] - self.spatial["PatientPositions"][:-1, 2]
        maxZ = np.max(zSpacing[zSpacing != 0])
        minXY = np.min(self.spatial["PixelSpacings"][self.spatial["PixelSpacings"] != 0])
        self.resolution = min([minXY, maxZ])/divisionFactor

    def normaliseVolumeMain(self):
        normalisedSizeZ = round(np.abs((self.spatial["PatientPositions"][0, 2]) - self.spatial["PatientPositions"][-1, 2])/self.resolution)
        normalisedSizeX = round(np.max(self.spatial["PixelSpacings"][:, 0])*self.volume.shape[0]/self.resolution)
        normalisedSizeY = round(np.max(self.spatial["PixelSpacings"][:, 1])*self.volume.shape[1]/self.resolution)

        if not (self.volume.shape[0] == normalisedSizeX and self.volume.shape[1] == normalisedSizeY and self.volume.shape[2] == normalisedSizeZ):
            try:
                self.volume = zoom(self.volume,
                                  (normalisedSizeX/self.volume.shape[0],
                                   normalisedSizeY/self.volume.shape[1],
                                   normalisedSizeZ/self.volume.shape[2]))
            except:
                print("Dicom volume can not be resized.")
        newZPositions = np.arange(self.spatial["PatientPositions"][0,  2],
                                  self.spatial["PatientPositions"][-1, 2],
                                  self.resolution).reshape(-1, 1)
        self.spatial["PatientPositions"] = np.concatenate([np.array([self.spatial["PatientPositions"][0, :2] for i in range(len(newZPositions))]),
                                                           newZPositions],
                                                          axis=1)
        self.spatial["PixelSpacings"] = np.vstack([np.array([[self.resolution, self.resolution]]) for i in range(self.spatial["PatientPositions"].shape[0])])

    def normaliseVolume(self):
        normalisedSizeZ = round(np.abs((self.spatial["PatientPositions"][0, 2]) - self.spatial["PatientPositions"][-1, 2])/self.resolution)
        normalisedSizeX = round(np.max(self.spatial["PixelSpacings"][:, 0])*self.volume.shape[0]/self.resolution)
        normalisedSizeY = round(np.max(self.spatial["PixelSpacings"][:, 1])*self.volume.shape[1]/self.resolution)

        if not (self.volume.shape[0] == normalisedSizeX and self.volume.shape[1] == normalisedSizeY and self.volume.shape[2] == normalisedSizeZ):
            try:
                self.volume = zoom(self.volume,
                                  (normalisedSizeX/self.volume.shape[0],
                                   normalisedSizeY/self.volume.shape[1],
                                   normalisedSizeZ/self.volume.shape[2]))
            except:
                print("Dicom volume can not be resized.")
        newZPositions = np.arange(self.spatial["PatientPositions"][0,  2],
                                  self.spatial["PatientPositions"][-1, 2],
                                  self.resolution).reshape(-1, 1)
        self.spatial["PatientPositions"] = np.concatenate([np.array([self.spatial["PatientPositions"][0, :2] for i in range(len(newZPositions))]),
                                                           newZPositions],
                                                          axis=1)
        self.spatial["PixelSpacings"] = np.vstack([np.array([[self.resolution, self.resolution]]) for i in range(self.spatial["PatientPositions"].shape[0])])
