from ShoulderCase.DicomVolumeNormaliser.DicomVolumeNormaliser import DicomVolumeNormaliser
import numpy as np

def getDicomBoundingBox(dicomPath):
    # Return the three dimensional bounding borders
    # of a dicom set found at the provided folder path.
    dicomVolume = DicomVolumeNormaliser(dicomPath)
    
    xMin = dicomVolume.spatial["PatientPositions"][0, 0]
    yMin = dicomVolume.spatial["PatientPositions"][0, 1]
    zMin = np.min(dicomVolume.spatial["PatientPositions"][:, 2])
    
    xMax = xMin + dicomVolume.volume.shape[0] * dicomVolume.spatial.PixelSpacings[0, 0]
    yMax = yMin + dicomVolume.volume.shape[1] * dicomVolume.spatial.PixelSpacings[0, 1]
    zMax = np.max(dicomVolume.spatial.PatientPositions[:, 2])
    
    boundingBox = {"xlim":[xMin, xMax],
                   "ylim":[yMin, yMax],
                   "zlim":[zMin, zMax]}

    return boundingBox

    
    

    