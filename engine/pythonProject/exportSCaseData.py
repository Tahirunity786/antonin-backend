from loadSCase import loadSCase
import pandas as pd

def exportSCaseData(SCaseList=["*"], CTDir=[], saveInFile=""):
    """"
    Export multiple SCases data into a table which can be saved in a file.
    Inputs:
        ("saveInFile", filename) filename is a string with the desired extension.
        It's the full path of the destination file.
        ("SCaseList", SCaseList) SCaseList is a string array.
    Output: table containing the SCases data.
    Example:
         exportSCaseData(SCaseList=["N29", "N32", "N234"], saveInFile="myThreeCases.csv");
    """
    casesToExport = loadSCase(SCaseList, *CTDir)

    for i in casesToExport:
        if i == 0:
            casesToExport.remove(i)

    exportedDataFrame = casesToExport[0].getDataFrameOfData()
    for i in range(1, len(casesToExport)):
        exportedDataFrame = pd.concat([exportedDataFrame, casesToExport[i].getDataFrameOfData()])
    if saveInFile:
        exportedDataFrame.to_csv(saveInFile, index=False)

    return exportedDataFrame




