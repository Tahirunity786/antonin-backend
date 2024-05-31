import json
import os

def setup():
    # to execute once after cloning the repo.
    createDefaultConfigFile()
    
def createDefaultConfigFile():
    defaultConfig = {}
    defaultConfig["OS"] = "windows"
    defaultConfig["maxSCaseIDDigits"] = 3
    defaultConfig["SCaseIDValidTypes"] = ["N", "P"]
    defaultConfig["muscleSegmentationModelDir"] = r"D:\Data\Pezhman\EPFL\Shoulder\segmentationElham\pythonCode"
    defaultConfig["dataDir"] = r"/Users/antonin/Desktop/Projet_ingénierie_simultanée/data"
    defaultConfig["landmarkAndSurfaceFilesFolder"] = "python"
    defaultConfig["casesToMeasure"] = ["P281"]
    
    defaultConfig["shouldersToMeasure"] = {}
    defaultConfig["shouldersToMeasure"]["rightAuto"] = True
    defaultConfig["shouldersToMeasure"]["rightManual"] = False
    defaultConfig["shouldersToMeasure"]["leftAuto"] = True
    defaultConfig["shouldersToMeasure"]["leftManual"] = False
    
    defaultConfig["runMeasurements"] = {}
    defaultConfig["runMeasurements"]["loadData"] = True
    defaultConfig["runMeasurements"]["sliceRotatorCuffMuscles"] = False
    defaultConfig["runMeasurements"]["segmentRotatorCuffMuscles"] = False
    defaultConfig["runMeasurements"]["morphology"] = True
    defaultConfig["runMeasurements"]["measureFirst"] = True
    defaultConfig["runMeasurements"]["measureSecond"] = True
    defaultConfig["runMeasurements"]["measureGlenoidDensity"] = False

    defaultConfig["rotatorCuffSliceName"] = "rotatorCuffMatthieu"
    defaultConfig["numberOfObliqueSlices"] = 0 # This can be 1, 3 or 10
    defaultConfig["croppedRotatorCuff"] = False
    defaultConfig["rotatorCuffSegmentationName"] = "autoMatthieu"
    defaultConfig["muscleSubdivisionsResolutionInMm"] = {}
    defaultConfig["muscleSubdivisionsResolutionInMm"]["x"] = 5
    defaultConfig["muscleSubdivisionsResolutionInMm"]["y"] = 5

    #AutoFE
    defaultConfig["runFE"] = False
    defaultConfig["AbaqusTempDir"] = r"C:\Temp\autoFE"
    defaultConfig["abaqusBat"] = r"C:\SIMULIA\Commands\abq2023.bat"
    defaultConfig["codeRootDir"] = os.getcwd()
    defaultConfig["implantFilesDir"] = r"D:\Data\Pezhman\EPFL\Shoulder\Data\PerformGlenoid\stl"
    defaultConfig["tempDockerDir"] = r"D:\Data\Pezhman\EPFL\Shoulder\Data\PerformGlenoid\stl"

    defaultConfig["overwriteMeasurements"] = True
    defaultConfig["saveMeasurements"] = True
    defaultConfig["saveAllMeasurementsInOneFile"] = False
    
    with open("config.json", "w") as json_file:
        json.dump(defaultConfig, json_file, indent=1)
