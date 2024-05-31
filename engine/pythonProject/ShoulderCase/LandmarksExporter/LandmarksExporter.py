import numpy as np
import os

class LandmarksExporter:
    def __init__(self):
        self.landmarks= np.array([])
        
    def addLandmarks(self, landmarks):
        self.landmarks = np.concatenate([self.landmarks, landmarks], axis=0)

    def exportAmiraFile(self, filepath, filename):
        fid = os.path.join(filepath, filename)
        with open(fid, 'wb') as f:
            f.write(self.getAmiraLandmarksFileWithCurrentLandmarks())

    def getAmiraLandmarksFileWithCurrentLandmarks(self):
        rawText = self.getAmiraLandmarksRawFile()
        fileText = rawText
        for i in range(self.landmarks.shape[0]):
            fileText += getTextFromPoint(self.landmarks[i,:]) 
        return fileText %(self.landmarks.shape[0])   
    
    def getAmiraLandmarksRawFile(self):
        return '# AmiraMesh 3D ASCII 2.0\n' + \
            '\n' + \
            '\n' + \
            'define Markers %d\n' + \
            '\n' + \
            'Parameters {\n' + \
            '    NumSets 1,\n' + \
            '    ContentType "LandmarkSet"\n' + \
            '}\n' + \
            '\n' + \
            'Markers { float[3] Coordinates } @1\n' + \
            '\n' + \
            '# Data section follows\n' + \
            '@1\n'
                
def getTextFromPoint(landmark):
    return "{:.15e} {:.15e} {:.15e}\n".format(landmark[0],
                                              landmark[1],
                                              landmark[2])  