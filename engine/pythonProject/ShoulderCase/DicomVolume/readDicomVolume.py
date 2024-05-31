import os
import numpy as np
from pydicom import dcmread

def readDicomVolume(path):
    try:
        slices = [dcmread(path + os.sep + s) for s in os.listdir(path)]
    except:
        slices = [dcmread(path + os.sep + s, force=True) for s in os.listdir(path)]

    try:
        slices.sort(key = lambda x: int(x.InstanceNumber))
    except:
        slices = slices[:-1]
        slices.sort(key=lambda x: int(x.InstanceNumber))


    # Volume
    pixel_vals = []
    for s in slices:
        pixel_vals.append(s.pixel_array)
    V = np.moveaxis(np.flip(np.array(pixel_vals), axis=0), 0, -1)

    # Patient Positions
    patientPositions = []
    for s in slices:
        patientPositions.append(s.ImagePositionPatient)

    patientPositions = np.flipud(np.array(patientPositions)) if patientPositions[0][2] > patientPositions[1][2] else np.array(patientPositions)

    # Pixel spacing
    pixelSpacings = []
    for s in slices:
        pixelSpacings.append(s.PixelSpacing)
    pixelSpacings = np.flipud(np.array(pixelSpacings))

    # Patient Orientation
    patientOrientation = np.zeros((2, 3, len(slices)))
    for i, s in enumerate(slices):
        patientOrientation[0, :, i] = np.array(s.ImageOrientationPatient)[:3]
        patientOrientation[1, :, i] = np.array(s.ImageOrientationPatient)[3:]

    return {"V":V,
            "PatientPositions":patientPositions,
            "PixelSpacings":pixelSpacings,
            "PatientOrientations":patientOrientation}