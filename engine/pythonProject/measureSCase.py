import time
from getConfig import getConfig
import createEmptySCase
import loadSCase
import pickle
import os
from utils.Logger.Logger import Logger


def measureSCase(casesToMeasure = [], CTDirs = None):

    #measureSCaseStart = time.time()
    #Logger.start("measureSCase")
    config = getConfig()

    # Initialise cases
    if casesToMeasure == []:
        casesToMeasure = config["casesToMeasure"]

    if config["overwriteMeasurements"]:
        SCases = createEmptySCase.createEmptySCase(casesToMeasure, CTDirs)

    else:
        SCases = loadSCase.loadSCase(casesToMeasure, CTDirs)

    # Run measurements
    for i in range(len(SCases)):
        #SCaseStart = time.time()

        if config["runMeasurements"]["loadData"]:
            SCases[i].patient.loadData()

        if config["shouldersToMeasure"]["rightAuto"]:
            runShoulderMeasurements(SCases[i].shoulders["right"]["auto"])

        if config["shouldersToMeasure"]["rightManual"]:
            runShoulderMeasurements(SCases[i].shoulders["right"]["manual"])

        if config["shouldersToMeasure"]["leftAuto"]:
            runShoulderMeasurements(SCases[i].shoulders["left"]["auto"])

        if config["shouldersToMeasure"]["leftManual"]:
            runShoulderMeasurements(SCases[i].shoulders["left"]["manual"])

    # Save results
    if config["saveMeasurements"]:
        saveMeasurements(SCases)

    if config["saveAllMeasurementsInOneFile"]:
        saveAllMeasurementsInOneFile(SCases, casesToMeasure)

def runShoulderMeasurements(shoulder):
    #shoulderStart = time.time()
    config = getConfig()

    # Measurements
    if config["runMeasurements"]["loadData"]:
        shoulder.loadData()

    if config["runMeasurements"]["sliceRotatorCuffMuscles"]:
        Logger.newDelimitedSection("Slice");
        Logger.logn("");
        Logger.timeLogExecution("Slice rotator cuff muscles: ",
            lambda shoulder:shoulder.rotatorCuff.slice(doSlicing=True), shoulder)
        Logger.closeSection()
    if config["runMeasurements"]["segmentRotatorCuffMuscles"]:
        Logger.newDelimitedSection("Segment");
        Logger.logn("");
        Logger.timeLogExecution("Segment rotator cuff muscles: ",
            lambda shoulder: shoulder.rotatorCuff.segment(doSegmentation=True), shoulder)
        Logger.closeSection()

    if config["runMeasurements"]["morphology"]:
        shoulder.morphology()

    if config["runMeasurements"]["measureFirst"]:
        shoulder.measureFirst()

    if config["runMeasurements"]["measureSecond"]:
        shoulder.measureSecond()

    #if config["runMeasurements"]["measureGlenoidDensity"]:
        #shoulder.measureDensity()

    if config["runMeasurements"]["measureGlenoidDensity"]:
        Logger.newDelimitedSection("Glenoid density")
        Logger.timeLogExecution("Glenoid density: ",
            lambda shoulder:shoulder.scapula.glenoid.measureDensity(), shoulder)
        Logger.closeSection()

    if config["runFE"]:
        shoulder.runFE()

def saveMeasurements(SCases):
    for SCase in SCases:
        SCase.savePython()

def saveAllMeasurementsInOneFile(SCases, casesToMeasure):
    Logger.log("Saving whole database in one file: ")
    try:
        if casesToMeasure != ["*"]:
            SCaseDB = loadSCase(["*"])
        else:
            SCaseDB = SCases
        path = os.path.join(getConfig()["dataDir"], getConfig()["landmarkAndSurfaceFilesFolder"], "SCaseDB.pkl")
        with open(path, "wb") as SCases_pkl:
            pickle.dump(SCaseDB, SCases_pkl)
    except Exception as e:
        Logger.logn(str(e))
