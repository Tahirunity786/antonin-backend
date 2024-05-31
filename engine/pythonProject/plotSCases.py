import pickle
from loadSCase import loadSCase
import os

def plotSCases(scaseList:list):
    def plotter(filename):
        try:
            with open(filename, "rb") as sc:
                scasePKL = pickle.load(sc)
                if scasePKL.shoulders["right"]["auto"].hasMeasurement != "":
                    scasePKL.plot("right", "auto")
                if scasePKL.shoulders["left"]["auto"].hasMeasurement != "":
                    scasePKL.plot("left", "auto")
        except:
            print(f"No {filename} exists")

    for scase in scaseList:
        case = loadSCase([scase])[0]
        filename = os.path.join(case.dataPythonPath(), "SCase.pkl")
        plotter(filename)



