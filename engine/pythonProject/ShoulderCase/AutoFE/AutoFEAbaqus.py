import os
import pandas as pd
import numpy as np
from ShoulderCase.DicomVolume.DicomVolume import DicomVolume
from ShoulderCase.AutoFE.FEMesh import FEMesh
from ShoulderCase.AutoFE.FEMaterial import FEMaterial
from getConfig import getConfig
import shutil
import json
from ShoulderCase.AutoFE.TSA import TSA
from ShoulderCase.loadStl import loadStl

class AutoFEAbaqus:

    def __init__(self, shoulder):
        self.abaqusInpFile = ""
        self.shoulder = shoulder
        self.mesh = FEMesh(self.shoulder)
        self.material = FEMaterial(self.mesh, os.path.join(self.mesh.shoulder.SCase.dataCTPath, "dicom"))
        self.implant = False
        self.rTSAUnionImplantScrews = True
        if os.path.exists(os.path.join(self.shoulder.SCase.dataCTPath, "preop")):
            self.tsa = TSA(self.shoulder)
            self.implant = self.tsa.preopFileInformation["implantType"]


    def getAbaqusDir(self):
        abaqusDir = os.path.join(self.mesh.getDataPath(), "Abaqus")
        if not os.path.exists(abaqusDir):
            os.makedirs(abaqusDir)
        return abaqusDir

    def putBoundaryConditionsDataInDatabase(self):
        """
        This function will put the data of the boundary conditions in the database
        """
        #if os.path.isfile(os.path.join(self.mesh.getDataPath(), "boundaryConditions.json")):
        #    return

        boundaryConditions = {}
        outputFileName = "boundaryConditionsWithImplant.json"

        boundaryConditions["loadDir"] = self.mesh.shoulder.scapula.coordSys.ML.tolist()

        if not self.withImplant:
            outputFileName = "boundaryConditions.json"

            scapulaInpFile = os.path.join(self.mesh.getDataPath(), "refined3DMesh.inp")
            scapulaInpFileDataFrame = inpFileToDataFrame(scapulaInpFile, 'NODE', 'C3D4')

            boundaryConditions["referencePoint"] = self.mesh.shoulder.scapula.glenoid.fittedSphere.center.flatten().tolist()

            glenoidSurface = self.mesh.shoulder.scapula.glenoid.surface["points"].round(0)
            isGlenoidSurface = False
            for idx, value in enumerate(glenoidSurface):
                isGlenoidSurface = isGlenoidSurface + \
                                   scapulaInpFileDataFrame[1].eq(glenoidSurface[idx][0]) * \
                                   scapulaInpFileDataFrame[2].eq(glenoidSurface[idx][1]) * \
                                   scapulaInpFileDataFrame[3].eq(glenoidSurface[idx][2])

            glenoidSurfaceNodes = scapulaInpFileDataFrame[0].loc[isGlenoidSurface]
            glenoidSurfaceNodes = (glenoidSurfaceNodes - 1).astype(np.int32).tolist()
            boundaryConditions["glenoidSurfaceNodes"] = glenoidSurfaceNodes

        # Defining the line between AG and TS which defines the border of boundary condition
        AI = self.mesh.shoulder.scapula.angulusInferior
        TS= self.mesh.shoulder.scapula.trigonumSpinae
        slopeXY = (AI[0][1] - TS[0][1]) / (AI[0][0] - TS[0][0])
        xyLine = (scapulaInpFileDataFrame[2]) - slopeXY * (scapulaInpFileDataFrame[1])

        checkBC = xyLine.ge(TS[0][1] - slopeXY * TS[0][0]) * scapulaInpFileDataFrame[3].le(TS[0][2])

        BCBox = scapulaInpFileDataFrame[0].loc[checkBC]
        BCBox = (BCBox - 1).astype(np.int32).tolist()

        boundaryConditions["BCBox"] = BCBox

        with open(os.path.join(self.mesh.getDataPath(), outputFileName), "w") as f:
            json.dump(boundaryConditions, f)

    def runAbaqusWithoutMaterial(self, abaqusScriptPath):
        try:
            dicomWeight = self.material.dicomSetForFE.dicomInfo.PatientWeight
            SCaseWeight = dicomWeight if ( dicomWeight != " " and int(dicomWeight) > 0 ) else 70
        except:
            SCaseWeight = 70
        extraArguments = f"{self.mesh.getDataPath()},{SCaseWeight}"
        osCommand = 'abq2023 cae noGUI=' + abaqusScriptPath + ' -- ' + extraArguments

        abaqusTempDir = getConfig()["AbaqusTempDir"]
        codeRootDir = getConfig()["codeRootDir"]

        os.chdir(abaqusTempDir)
        os.system(osCommand)
        os.chdir(codeRootDir)

        abaqusTempDirFiles = os.listdir(abaqusTempDir)
        for file in abaqusTempDirFiles:
            source = os.path.join(abaqusTempDir, file)
            if file == "inputFileWithoutMaterial.inp":
                dir_ = os.path.join(self.mesh.getDataPath(), file)
                shutil.move(source, dir_)
            else:
                os.remove(source)

    def abaqusExecuteJob(self, jobScriptPath, inputInpFile):
        """
        This function will execute the simulation in command line with calling JobScripting.
        Will take inputFileWithMaterial as modelInputName.
        After copy from C:\Temp (where abaqus work) to RunFile, of the corresponding case, all the files created by the simulation
        """

        runCommand = "abaqus cae noGUI=" + jobScriptPath + ' -- ' + self.mesh.getDataPath()

        abaqusTempDir = getConfig()["AbaqusTempDir"]
        codeRootDir = getConfig()["codeRootDir"]

        os.chdir(abaqusTempDir)
        os.system(runCommand)
        os.chdir(codeRootDir)

        abaqusTempDirFiles = os.listdir(abaqusTempDir)
        for file in abaqusTempDirFiles:
            source = os.path.join(abaqusTempDir, file)
            dir_ = os.path.join(self.getAbaqusDir(), file)
            shutil.move(source, dir_)

    def runAbaqus(self):

        if not self.implant:
            self.mesh.refineTriangularMesh()
            self.mesh.convert2DMeshTo3D()
            self.mesh.areTetrahedralElementsInMesh()
            self.mesh.delete2DElements()

            self.putBoundaryConditionsDataInDatabase()

            self.runAbaqusWithoutMaterial(os.path.join(os.getcwd(), "ShoulderCase", "AutoFE", "abaqusScript.py"))

            self.material.assignMaterialFromDicom()

            self.abaqusExecuteJob(
                os.path.join(os.getcwd(), "ShoulderCase", "AutoFE", "jobScript.py"),
                self.mesh.getDataPath() + os.sep + "inputFileWithoutMaterial.inp"
            )
        elif self.implant == "Anatomic":
            if self.shoulder.hasMeasurement:
                aTSA = TSA(self.shoulder)
                aTSA.performATSAOnTheMesh()
                aTSA.putTSABoundaryConditionsDataInDatabase()
                self.material.assignMaterialFromDicom(implant=True)
                aTSA.performTSAWithAbaqus(
                    os.path.join(os.getcwd(),
                                 "ShoulderCase",
                                 "AutoFE",
                                 "aTSAAbaqusScript.py"
                                 )
                )
                aTSA.performAbaqusJob()

        elif self.implant == "Reversed" and not self.rTSAUnionImplantScrews:
            if self.shoulder.hasMeasurement:
                rTSA = TSA(self.shoulder)
                rTSA.performRTSAOnTheMesh()
                rTSA.putTSABoundaryConditionsDataInDatabase()
                self.material.assignMaterialFromDicom(implant=True)
                rTSA.performTSAWithAbaqus(
                    os.path.join(os.getcwd(),
                                 "ShoulderCase",
                                 "AutoFE",
                                 "rTSAFourScrewsAbaqusScript.py"
                                 )
                )
                rTSA.performAbaqusJob()

        elif self.implant == "Reversed" and self.rTSAUnionImplantScrews:
            if self.shoulder.hasMeasurement:
                rTSA = TSA(self.shoulder)
                rTSA.performRTSAOnTheUnionMesh(self.implant)
                rTSA.putTSABoundaryConditionsDataInDatabase()
                self.material.assignMaterialFromDicom(implant=True)
                rTSA.performTSAWithAbaqus(
                    os.path.join(
                        os.getcwd(),
                        "ShoulderCase",
                        "AutoFE",
                        "rTSAFourScrewsAbaqusScriptUnionScrewImplant.py"
                    )
                )
                rTSA.performAbaqusJob()


def inpFileToDataFrame(inpFile, firstItem, lastItem):
    with open(inpFile) as f:
        inpFileData = f.readlines()
    inpFileDataFrame = pd.DataFrame(inpFileData, columns=['data'])

    firstIdx = inpFileDataFrame.index[inpFileDataFrame['data'].str.contains(firstItem)]
    lastIdx = inpFileDataFrame.index[inpFileDataFrame['data'].str.contains(lastItem)]

    firstIdx = firstIdx[0].astype(int) + 1
    lastIdx = lastIdx[0].astype(int) - 1

    inpFileDataFrame = inpFileDataFrame[firstIdx:lastIdx]
    inpFileDataFrame = inpFileDataFrame['data'].str.split(",", expand=True)
    inpFileDataFrame = inpFileDataFrame.astype(float)
    inpFileDataFrame = inpFileDataFrame.round(0)
    inpFileDataFrame = inpFileDataFrame.reset_index()
    inpFileDataFrame = inpFileDataFrame.drop(columns=['index'])
    return inpFileDataFrame