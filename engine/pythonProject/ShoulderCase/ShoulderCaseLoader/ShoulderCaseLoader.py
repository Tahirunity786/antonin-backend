import os
from getConfig import getConfig
from ShoulderCase.SCaseIDParser.SCaseIDParser import SCaseIDParser
from ShoulderCase.ShoulderCase.ShoulderCase import ShoulderCase
import glob
from utils.UpdatableText import UpdatableText
import pickle

class ShoulderCaseLoader:
    """
    Class to interact with the database.

    The main purpose of the ShoulderCaseLoader is to
    easily load the cases. By initialisation, a
    ShoulderCaseLoader instance will set its dataRootPath
    property to the dataDir variable found in the
    config.txt file.

    Because of cross-systems utilisation of the database
    the paths stored in the ShoulderCase instances may
    vary from one execution to the other.
    Thus, this class evolved to be the access point to
    the ShoulderCase constructor, to avoid misusing
    the paths. This was also the incentive to create the
    ShoulderCase.propagateDataPath() method.

    example:
    database = ShoulderCaseLoader()
    SCase = database.loadCase(['P500']);
    """
    def __init__(self):
        self.shoulderCasePath = {}
        self.dataRootPath = ""
        self.allCasesFound = False
        try:
            self.setDataRootPath(getConfig()["dataDir"])
        finally:
            pass

    def setDataRootPath(self, dataRootPath):
        assert os.path.isdir(dataRootPath), "%s is not a valid path."%dataRootPath

        # Do not reset obj.shoulderCasePath if dataRootPath doesn't change
        if self.dataRootPath == dataRootPath:
            return
        self.dataRootPath = dataRootPath
        self.shoulderCasePath = {}
        self.allCasesFound = False

    def containsCase(self, SCaseID):
        return self.findCase(SCaseID)

    def findCase(self, SCaseID):

        # Case already found
        SCaseID = SCaseID[0] if isinstance(SCaseID, list) else SCaseID
        if SCaseID in self.shoulderCasePath:
            return True

        # Format ID to construct the case's path
        givenID = SCaseIDParser(SCaseID)
        assert givenID.isValidID(), '%s is not a valid ID.' %SCaseID
        SCaseIDmaxDigits = givenID.getIDWithNumberOfDigits(getConfig()["maxSCaseIDDigits"])

        SCasePath = os.path.join(self.dataRootPath,
                                 SCaseIDmaxDigits[0], #SCaseID type
                                 SCaseIDmaxDigits[1], #SCaseID hundreds
                                 SCaseIDmaxDigits[2], #SCaseID tens
                                 SCaseID) + '*' #SCaseID folder whatever the IPP is

        SCaseFound = glob.glob(SCasePath)
        if not SCaseFound:
            return

        self.shoulderCasePath[SCaseID] = os.path.join(os.path.realpath(SCaseFound[0]))

        return True

    def createEmptyCase(self, SCaseID, CTDir):

        emptyCases = []
        # Recursion if a list of SCaseID is given
        if isinstance(SCaseID, list) and len(SCaseID) >= 1:
            for i in range(len(SCaseID)):
                try:
                    if CTDir is None:
                        emptyCases.append(self.createEmptyCase(SCaseID[i], None))
                    else:
                        emptyCases.append(self.createEmptyCase(SCaseID[i], CTDir[i]))
                except:
                    continue
            return emptyCases

        self.findCase(SCaseID)
        SCaseDataCTPath = {}

        smoothKernels = ["STANDARD", "DETAIL", "A", "B", "FC13", "B25s", "B26s", "B31s", "I31s"]
        if CTDir is None:
            for kernel in smoothKernels:
                if SCaseID[0] == "P":
                    SCaseCTFolders = glob.glob(os.path.join(self.shoulderCasePath[SCaseID], f"CT_*_shoulder_{kernel}_*_preop"))
                elif SCaseID[0] == "N":
                    SCaseCTFolders = glob.glob(os.path.join(self.shoulderCasePath[SCaseID], f"CT_*_shoulder_{kernel}_*"))
                if SCaseCTFolders:
                    SCaseDataCTPath.setdefault("pathWithPython", SCaseCTFolders[0])
                    return ShoulderCase(SCaseID, SCaseDataCTPath)
            if not SCaseCTFolders:
                SCaseCTFolders = glob.glob(os.path.join(self.shoulderCasePath[SCaseID], "CT_*_shoulder_*"))
                SCaseDataCTPath.setdefault("pathWithPython", SCaseCTFolders[0])
                assert SCaseCTFolders, f"No CT folder found for {SCaseID}"
        else:
            SCaseCTFolders = glob.glob(os.path.join(self.shoulderCasePath[SCaseID], CTDir))
            SCaseDataCTPath.setdefault("pathWithPython", SCaseCTFolders[0])
            assert SCaseCTFolders, f"There is no {CTDir} directory"
            return ShoulderCase(SCaseID, SCaseDataCTPath)

        for SCaseCTFolder in SCaseCTFolders:
            SCaseDataCTPath.setdefault(SCaseCTFolder.split(os.sep)[-1], os.path.realpath(SCaseCTFolder))

        # Priority is given to folder containing a python archive
        for SCaseCTFolder in SCaseCTFolders:
            if os.path.isfile(os.path.join(
                    SCaseCTFolder,
                    getConfig()["landmarkAndSurfaceFilesFolder"],
                    "SCase.pkl"
            )):
                SCaseDataCTPath.setdefault("pathWithPython", SCaseCTFolder)
                break

        # Then, priority is given to a preoperative CT folder
        if not SCaseDataCTPath:
            for SCaseCTFolder in SCaseCTFolders:
                if SCaseCTFolder.split(os.sep)[-1].split("_")[-1] == "preoperative":
                    SCaseDataCTPath.setdefault("pathWithPython", SCaseCTFolder)

        # Then, priority is given to CT folder with lowest ending number
        if not SCaseDataCTPath:
            SCaseDataCTPath.setdefault("pathWithPython", SCaseCTFolders[0])

        return ShoulderCase(SCaseID, SCaseDataCTPath)

    def findAllCases(self):
        if self.allCasesFound:
            return

        types = getConfig()["SCaseIDValidTypes"]
        maxSCaseNumber = 10**(getConfig()["maxSCaseIDDigits"])-1

        progression = UpdatableText.UpdatableText('',' All valid cases are being looked for.')
        for i in range(len(types)):
            for j in range(1, maxSCaseNumber+1):
                progression.printPercent(getProgressionFraction(i,j,len(types),maxSCaseNumber))
                progression.printProgressBar(getProgressionFraction(i,j,len(types),maxSCaseNumber))
                SCaseID = [types[i] + str(j)]
                try:
                    self.findCase(SCaseID)
                except:
                    pass

        self.allCasesFound = True

    def getAllCasesID(self):
        self.findAllCases()
        casesID = list(self.shoulderCasePath.keys())
        return casesID

    def getAllNormalCasesID(self):
        self.findAllCases()
        casesID = list(self.shoulderCasePath.keys())
        for case in casesID:
            if case[0] == "P":
                casesID.remove(case)
        return casesID

    def getAllPathologicalCasesID(self):
        self.findAllCases()
        casesID = list(self.shoulderCasePath.keys())
        for case in casesID:
            if case[0] == "N":
                casesID.remove(case)
        return casesID

    def getCasePath(self, SCaseID):
        assert self.findCase(SCaseID), '%s not found in the database.' % SCaseID
        return self.shoulderCasePath(SCaseID)

    def getDataRootPath(self):
        return self.dataRootPath

    def getNumberOfFoundCases(self):
        return len(self.shoulderCasePath.keys())

    def loadAllCases(self):
        allCases = []

        self.findAllCases()
        allCasesID = list(self.shoulderCasePath.keys())
        totalNumberOfCases = self.getNumberOfFoundCases()
        progression = UpdatableText.UpdatableText('',' All cases are being loaded.')
        for i in range(totalNumberOfCases):
            progression.printPercent(i/totalNumberOfCases)
            progression.printProgressBar(i/totalNumberOfCases)
            try:
                allCases.append(self.loadCase(allCasesID[i]))
            finally:
                pass

        return allCases

    def loadCase(self, SCaseID, *CTDir):

        loadedSCases = []
        # Recursion if a list of SCaseID is given
        if isinstance(SCaseID, list) and len(SCaseID) >= 1:
            for i in range(len(SCaseID)):
                if CTDir:
                    loadedSCases.append(self.loadCase(SCaseID[i], CTDir[0][i]))
                else:
                    loadedSCases.append(self.loadCase(SCaseID[i]))
            return loadedSCases
        assert self.findCase(SCaseID), f"{SCaseID} not found in the database."

        # find SCase.pkl files
        if CTDir:
            print(SCaseID)
            SCaseFilePath = glob.glob(os.path.join(
                self.shoulderCasePath[SCaseID],
                CTDir[0],
                getConfig()["landmarkAndSurfaceFilesFolder"],
                "SCase.pkl"
            ))
            if not SCaseFilePath:
                print(f"No archive found for {SCaseID} in {CTDir[0]}")
                return 0
        else:
            print(SCaseID)
            smoothKernels = ["STANDARD", "DETAIL", "A", "B", "FC13", "B25s", "B26s", "B31s", "I31s"]
            for kernel in smoothKernels:
                SCaseFilePath = glob.glob(os.path.join(
                    self.shoulderCasePath[SCaseID],
                    f"CT_*_shoulder_{kernel}_*_preop",
                    getConfig()["landmarkAndSurfaceFilesFolder"],
                    "SCase.pkl"
                ))
                if SCaseFilePath:
                    break
            if not SCaseFilePath:
                SCaseFilePath = glob.glob(os.path.join(
                    self.shoulderCasePath[SCaseID],
                    "CT_*_shoulder_*",
                    getConfig()["landmarkAndSurfaceFilesFolder"],
                    "SCase.pkl"
                ))
                if not SCaseFilePath:
                    print(f"No archive found for {SCaseID}")
                    return 0

        # Load a verified ShoulderCase instance
        filename = SCaseFilePath[0]
        with open(filename, "rb") as pklSCase:
            try:
                loaded = pickle.load(pklSCase)
            except:
                raise FileNotFoundError("No SCase field found in the archive")

        SCase = loaded
        assert isinstance(loaded, ShoulderCase), "The found SCase is not a ShoulderCase instance"

        # Update data paths
        SCase.dataCTPath = os.sep.join(SCaseFilePath[0].split(os.sep)[:-2])

        return SCase

def getProgressionFraction(i,j,numberOfTypes,maxSCaseNumber):
    currentNumber = (i)*(maxSCaseNumber+1)+j
    maxNumber = numberOfTypes*(maxSCaseNumber+1)-1
    return currentNumber/maxNumber
