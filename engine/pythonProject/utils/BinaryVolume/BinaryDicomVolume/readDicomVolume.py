import os
import numpy as np
from pydicom import dcmread

def readSlices(path):
    slices = [dcmread(path + os.sep + s) for s in os.listdir(path)]
    slices.sort(key = lambda x: int(x.InstanceNumber))
    return slices
    
def readDicomVolume(slices):
    pixel_vals = []
    for s in slices:
        pixel_vals.append(s.pixel_array)
    return np.moveaxis(np.flip(np.array(pixel_vals), axis=0), 0, -1)

def readPatientPositions(slices):
    patientPositions = []
    for s in slices:
        patientPositions.append(s.ImagePositionPatient)
    return np.flipud(np.array(patientPositions))   

def readPixelSpacings(slices):
    pixelSpacings = []
    for s in slices:
        pixelSpacings.append(s.PixelSpacing)
    return np.flipud(np.array(pixelSpacings)) 
    
    
        
    