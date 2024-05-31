import os
#import docker
import pydicom
import trimesh
#import gmsh
import numpy as np
import copy
#import pymeshlab
from getConfig import getConfig
from utils.Rotations.rotation_angle import rotation_matrix_from_vectors
import shutil
import pandas as pd
import ast
import json
from ShoulderCase.loadStl import loadStl
from ShoulderCase.DicomVolume.readDicomVolume import readDicomVolume
#import pymeshfix
import subprocess
import copy
from scipy.spatial.transform import Rotation
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
import meshio

class TSA:
    def __init__(self, shoulder):

        self.shoulder = shoulder
        self.scapulaMesh = ""
        self.glenoidMesh = ""
        self.preopFileInformation = {}
        self.implantMesh = ""
        self.implantSurfaceMesh = ""
        self.implantBoneInterfaceMesh = ""
        self.cementMesh = ""
        self.reamerMesh = ""
        self.reamerBaseMesh = ""
        self.centralScrew = ""
        self.superiorScrew = ""
        self.inferiorScrew = ""
        self.posteriorScrew = ""
        self.anteriorScrew = ""
        self.rTSALoadApplicationPoint = []
        self.rotatedImplant = ""
        self.importMeshFiles()

    def importMeshFiles(self):

        self.returnPreopFileInformation()

        scapulaPlyFilePath = os.path.join(
            self.shoulder.SCase.dataCTPath,
            getConfig()["landmarkAndSurfaceFilesFolder"],
            f"scapulaSurfaceAuto{self.shoulder.side}.ply"
        )
        if os.path.exists(scapulaPlyFilePath):
            self.scapulaMesh = trimesh.load(scapulaPlyFilePath)
        glenoidPlyFilePath = os.path.join(
            self.shoulder.SCase.dataCTPath,
            getConfig()["landmarkAndSurfaceFilesFolder"],
            f"glenoidSurfaceAuto{self.shoulder.side}.ply"
        )
        if os.path.exists(glenoidPlyFilePath):
            self.glenoidMesh = trimesh.load(glenoidPlyFilePath)

        if os.path.exists(scapulaPlyFilePath):
            implantStlFilePath = os.path.join(
                getConfig()["implantFilesDir"],
                f"{self.preopFileInformation['implantName']}.stl"
            )
            if os.path.exists(implantStlFilePath):
                self.implantMesh = trimesh.load(implantStlFilePath)

            reamerStlFilePath = os.path.join(
                getConfig()["implantFilesDir"],
                f"{self.preopFileInformation['implantName']}Reamer.stl"
            )
            if os.path.exists(reamerStlFilePath):
                self.reamerMesh = trimesh.load(reamerStlFilePath)

            reamerBaseStlFilePath = os.path.join(
                getConfig()["implantFilesDir"],
                f"{self.preopFileInformation['implantName']}ReamerBase.stl"
            )
            if os.path.exists(reamerBaseStlFilePath):
                self.reamerBaseMesh = trimesh.load(reamerBaseStlFilePath)

            implantSurfaceStlFilePath = os.path.join(
                getConfig()["implantFilesDir"],
                f"{self.preopFileInformation['implantSurfaceName']}.stl"
            )
            if os.path.exists(implantSurfaceStlFilePath):
                self.implantSurfaceMesh = trimesh.load(implantSurfaceStlFilePath)

            if self.preopFileInformation["implantType"] == "Reversed":

                implantBoneInterfaceStlFilePath = os.path.join(
                    getConfig()["implantFilesDir"],
                    f"{self.preopFileInformation['implantName']}BoneInterface.stl"
                )
                if os.path.exists(implantBoneInterfaceStlFilePath):
                    self.implantBoneInterfaceMesh = trimesh.load(implantBoneInterfaceStlFilePath)

            elif self.preopFileInformation["implantType"] == "Anatomic":

                cementMeshStlFilePath = os.path.join(
                    getConfig()["implantFilesDir"],
                    f"{self.preopFileInformation['implantName']}Cement.stl"
                )
                if os.path.exists(cementMeshStlFilePath):
                    self.cementMesh = trimesh.load(cementMeshStlFilePath)

                cementBoneInterfaceStlFilePath = os.path.join(
                    getConfig()["implantFilesDir"],
                    f"{self.preopFileInformation['implantName']}CementBoneInterface.stl"
                )
                if os.path.exists(cementBoneInterfaceStlFilePath):
                    self.cementBoneInterfaceMesh = trimesh.load(cementBoneInterfaceStlFilePath)

                cementImplantInterfaceStlFilePath = os.path.join(
                    getConfig()["implantFilesDir"],
                    f"{self.preopFileInformation['implantName']}CementImplantInterface.stl"
                )
                if os.path.exists(cementImplantInterfaceStlFilePath):
                    self.cementImplantInterfaceMesh = trimesh.load(cementImplantInterfaceStlFilePath)

                implantCementInterfaceStlFilePath = os.path.join(
                    getConfig()["implantFilesDir"],
                    f"{self.preopFileInformation['implantName']}ImplantCementInterface.stl"
                )
                if os.path.exists(implantCementInterfaceStlFilePath):
                    self.implantCementInterfaceMesh = trimesh.load(implantCementInterfaceStlFilePath)

    def returnPreopFileInformation(self):
        # This function should read the preop file and return the implant name
        preopFilePath = os.path.join(self.shoulder.SCase.dataCTPath, "preop", "data", f"{self.shoulder.SCase.id}.txt")
        preopData = pd.read_csv(
            preopFilePath,
            delimiter=';',
            header=None,
            names=["field1", "value1", "field2", "value2"]
        )

        strVal = preopData[preopData.field1 == "GlenoidImplant_AnteriorAxis"].value1.tolist()
        glenoidImplantAntAxis = ast.literal_eval(strVal[0])

        strVal = preopData[preopData.field1 == "GlenoidImplant_LateralAxis"].value1.tolist()
        glenoidImplantLatAxis = ast.literal_eval(strVal[0])

        strVal = preopData[preopData.field1 == "GlenoidImplant_SuperiorAxis"].value1.tolist()
        glenoidImplantSupAxis = ast.literal_eval(strVal[0])

        strVal = preopData[preopData.field1 == "GlenoidImplant_Center"].value1.tolist()
        glenoidImplantCenter = ast.literal_eval(strVal[0])
        # The origin of the dicom CT coordinate system is not the same as the origin of the blueprint
        # You need to add the first row of patientPositions to the origin of bleprint to have the values in the
        # CT coordinate system
        patientPositions = readDicomVolume(os.path.join(self.shoulder.SCase.dataCTPath, "dicom"))["PatientPositions"]
        patientPositions = patientPositions[0, :]
        glenoidImplantCenter = glenoidImplantCenter + patientPositions

        glenoidImplantInclination = float(preopData[preopData.field1 == "GlenoidImplant_Inclination"].value1.iloc[0])
        glenoidImplantVersion = float(preopData[preopData.field1 == "GlenoidImplant_Version"].value1.iloc[0])

        implantType = preopData[preopData.field1 == "GlenoidImplant_Type"].value1.iloc[0].split(" ")[-1]
        if implantType == "PERFORM":
            implantType = "Anatomic"

        implantName = preopData[preopData.field1 == "GlenoidImplant_PartNumber"].value1.iloc[0]

        propertiesDict = {
            "implantType": implantType,
            "implantName": implantName,
            "implantSurfaceName": f"{implantName}Surface",
            "glenoidImplantCenter": glenoidImplantCenter,
            "glenoidImplantAntAxis": glenoidImplantAntAxis,
            "glenoidImplantLatAxis": glenoidImplantLatAxis,
            "glenoidImplantSupAxis": glenoidImplantSupAxis,
            "glenoidImplantInclination": glenoidImplantInclination,
            "glenoidImplantVersion": glenoidImplantVersion
        }

        if implantType == "Reversed":
            propertiesDict = self.updatePreopDataDictForRTSA(implantType, propertiesDict)

        self.preopFileInformation = propertiesDict

    def updatePreopDataDictForRTSA(self, implantType, propertiesDict):

        preopFilePath = os.path.join(self.shoulder.SCase.dataCTPath, "preop", "data", f"{self.shoulder.SCase.id}.txt")
        preopData = pd.read_csv(
            preopFilePath,
            delimiter=';',
            header=None,
            names=["field1", "value1", "field2", "value2"]
        )

        if implantType == "Reversed":

            centralScrewDiameter = \
            preopData[preopData.field1 == "GlenoidImplant_PerformReversedScrewDiameter"].value1.iloc[0]
            centralScrewLength = \
                preopData[preopData.field1 == "GlenoidImplant_PerformReversedScrewLength"].value1.iloc[0]
            centralScrew = f"CentralScrew_{centralScrewDiameter}_{centralScrewLength}.stl"

            glenosphereDiameter = preopData[preopData.field1 == "GlenoidImplant_GlenosphereDiameter"].value1.iloc[0]

            propertiesDict.update({
                "centralScrew":        centralScrew,
                "glenosphereDiameter": glenosphereDiameter
            })

            try:
                SUPScrewType = preopData[preopData.field1 == "GlenoidImplant_PeripheralScrewType_SUP"].value1.iloc[0]
                SUPScrewSize = preopData[preopData.field1 == "GlenoidImplant_PeripheralScrewSize_SUP"].value1.iloc[0]
                SUPScrewSupInfAngle = \
                    preopData[preopData.field1 == "GlenoidImplant_PeripheralScrewAngleSupInf_SUP"].value1.iloc[0]
                SUPScrewAntPostAngle = \
                    preopData[preopData.field1 == "GlenoidImplant_PeripheralScrewAngleAntPost_SUP"].value1.iloc[0]

                ANTScrewType = preopData[preopData.field1 == "GlenoidImplant_PeripheralScrewType_ANT"].value1.iloc[0]
                ANTScrewSize = preopData[preopData.field1 == "GlenoidImplant_PeripheralScrewSize_ANT"].value1.iloc[0]
                ANTScrewSupInfAngle = \
                    preopData[preopData.field1 == "GlenoidImplant_PeripheralScrewAngleSupInf_ANT"].value1.iloc[0]
                ANTScrewAntPostAngle = \
                    preopData[preopData.field1 == "GlenoidImplant_PeripheralScrewAngleAntPost_ANT"].value1.iloc[0]

                INFScrewType = preopData[preopData.field1 == "GlenoidImplant_PeripheralScrewType_INF"].value1.iloc[0]
                INFScrewSize = preopData[preopData.field1 == "GlenoidImplant_PeripheralScrewSize_INF"].value1.iloc[0]
                INFScrewSupInfAngle = \
                    preopData[preopData.field1 == "GlenoidImplant_PeripheralScrewAngleSupInf_INF"].value1.iloc[0]
                INFScrewAntPostAngle = \
                    preopData[preopData.field1 == "GlenoidImplant_PeripheralScrewAngleAntPost_INF"].value1.iloc[0]

                POSTScrewType = preopData[preopData.field1 == "GlenoidImplant_PeripheralScrewType_POST"].value1.iloc[0]
                POSTScrewSize = preopData[preopData.field1 == "GlenoidImplant_PeripheralScrewSize_POST"].value1.iloc[0]
                POSTScrewSupInfAngle = \
                    preopData[preopData.field1 == "GlenoidImplant_PeripheralScrewAngleSupInf_POST"].value1.iloc[0]
                POSTScrewAntPostAngle = \
                    preopData[preopData.field1 == "GlenoidImplant_PeripheralScrewAngleAntPost_POST"].value1.iloc[0]

                propertiesDict.update({
                    "SUPScrewSize":          SUPScrewSize,
                    "SUPScrewSupInfAngle":   SUPScrewSupInfAngle,
                    "SUPScrewAntPostAngle":  SUPScrewAntPostAngle,
                    "ANTScrewSize":          ANTScrewSize,
                    "ANTScrewSupInfAngle":   ANTScrewSupInfAngle,
                    "ANTScrewAntPostAngle":  ANTScrewAntPostAngle,
                    "INFScrewSize":          INFScrewSize,
                    "INFScrewSupInfAngle":   INFScrewSupInfAngle,
                    "INFScrewAntPostAngle":  INFScrewAntPostAngle,
                    "POSTScrewSize":         POSTScrewSize,
                    "POSTScrewSupInfAngle":  POSTScrewSupInfAngle,
                    "POSTScrewAntPostAngle": POSTScrewAntPostAngle
                })

            except:
                propertiesDict.update({
                    "SUPScrewSize": 18,
                    "SUPScrewSupInfAngle": 15,
                    "SUPScrewAntPostAngle": 0,
                    "ANTScrewSize": 38,
                    "ANTScrewSupInfAngle": 2,
                    "ANTScrewAntPostAngle": 5,
                    "INFScrewSize": 18,
                    "INFScrewSupInfAngle": -14,
                    "INFScrewAntPostAngle": 0,
                    "POSTScrewSize": 38,
                    "POSTScrewSupInfAngle": 0,
                    "POSTScrewAntPostAngle": -3
                })

        return propertiesDict

    def dirToSaveMesh(self):
        side = "left" if self.shoulder.side == "L" else "right"
        return  rf"{os.path.join(self.shoulder.SCase.dataCTPath, getConfig()['landmarkAndSurfaceFilesFolder'], 'shoulders', side, 'auto', 'FE')}"

    def findrTSALoadApplicationPoint(self):

        implantName = self.preopFileInformation["implantName"]
        glenoSphereRadius = float(self.preopFileInformation["glenosphereDiameter"]) / 2

        if implantName == "DWJ505":
            rTSALoadApplicationPointML = 7.05 + glenoSphereRadius # 7.05 is based on the cad file
        elif implantName == "DWJ502":
            rTSALoadApplicationPointML = 7.05 + glenoSphereRadius

        self.rTSALoadApplicationPoint = np.array([rTSALoadApplicationPointML, 0, 0, 1])

    def alignScrewWithImplant(self):
        implantName = self.preopFileInformation["implantName"]

        if implantName == "DWJ505":
            implantSUPHoleCenter  = [(5.7106 + 7.4559) / 2, 0, (3.9175 + 10.431) / 2]
            implantANTHoleCenter  = [(5.701 + 7.4463) / 2, (-10.395 - 3.8817) / 2, 0]
            implantINFHoleCenter  = [(5.7106 + 7.4559) / 2, 0, (-10.431 - 3.9175) / 2]
            implantPOSTHoleCenter = [(5.3916 + 5.7056) / 2, (4.9828 + 10.975) / 2, 0]

        elif implantName == "DWJ502":
            implantSUPHoleCenter = [(5.7106 + 7.4559) / 2, 0, (3.9175 + 10.431) / 2]
            implantANTHoleCenter = [(5.701 + 7.4463) / 2, (-10.395 - 3.8817) / 2, 0]
            implantINFHoleCenter = [(5.7106 + 7.4559) / 2, 0, (-10.431 - 3.9175) / 2]
            implantPOSTHoleCenter = [(5.3916 + 5.7056) / 2, (4.9828 + 10.975) / 2, 0]

        self.centralScrew = trimesh.load(os.path.join(
            getConfig()["implantFilesDir"],
            self.preopFileInformation["centralScrew"]
        ))

        superiorScrewDiameter = self.preopFileInformation["SUPScrewSize"]
        self.superiorScrew = trimesh.load(os.path.join(
                getConfig()["implantFilesDir"],
                f"PeripheralScrew{superiorScrewDiameter}.stl"
            ))
        SUPScrewSupInfAngle  = self.preopFileInformation["SUPScrewSupInfAngle"]
        SUPScrewAntPostAngle = self.preopFileInformation["SUPScrewAntPostAngle"]
        screwToImplant = rotationMatrixFromAngles(0, int(SUPScrewSupInfAngle), int(SUPScrewAntPostAngle))
        screwToImplantTransformationMatrix = np.eye(4)
        screwToImplantTransformationMatrix[:3, :3] = screwToImplant
        screwToImplantTransformationMatrix[:3,  3] = implantSUPHoleCenter
        self.superiorScrew.apply_transform(screwToImplantTransformationMatrix)

        inferiorScrewDiameter = self.preopFileInformation["INFScrewSize"]
        self.inferiorScrew = trimesh.load(os.path.join(
            getConfig()["implantFilesDir"],
            f"PeripheralScrew{inferiorScrewDiameter}.stl"
        ))
        INFScrewSupInfAngle = self.preopFileInformation["INFScrewSupInfAngle"]
        INFScrewAntPostAngle = self.preopFileInformation["INFScrewAntPostAngle"]
        screwToImplant = rotationMatrixFromAngles(0, int(INFScrewSupInfAngle), int(INFScrewAntPostAngle))
        screwToImplantTransformationMatrix = np.eye(4)
        screwToImplantTransformationMatrix[:3, :3] = screwToImplant
        screwToImplantTransformationMatrix[:3, 3] = implantINFHoleCenter
        self.inferiorScrew.apply_transform(screwToImplantTransformationMatrix)

        anteriorScrewDiameter = self.preopFileInformation["ANTScrewSize"]
        self.anteriorScrew = trimesh.load(os.path.join(
            getConfig()["implantFilesDir"],
            f"PeripheralScrew{anteriorScrewDiameter}.stl"
        ))
        ANTScrewSupInfAngle = self.preopFileInformation["ANTScrewSupInfAngle"]
        ANTScrewAntPostAngle = self.preopFileInformation["ANTScrewAntPostAngle"]
        screwToImplant = rotationMatrixFromAngles(0, int(ANTScrewSupInfAngle), int(ANTScrewAntPostAngle))
        screwToImplantTransformationMatrix = np.eye(4)
        screwToImplantTransformationMatrix[:3, :3] = screwToImplant
        screwToImplantTransformationMatrix[:3, 3] = implantANTHoleCenter
        self.anteriorScrew.apply_transform(screwToImplantTransformationMatrix)

        posteriorScrewDiameter = self.preopFileInformation["POSTScrewSize"]
        self.posteriorScrew = trimesh.load(os.path.join(
            getConfig()["implantFilesDir"],
            f"PeripheralScrew{posteriorScrewDiameter}.stl"
        ))
        POSTScrewSupInfAngle = self.preopFileInformation["POSTScrewSupInfAngle"]
        POSTScrewAntPostAngle = self.preopFileInformation["POSTScrewAntPostAngle"]
        screwToImplant = rotationMatrixFromAngles(0, int(POSTScrewSupInfAngle), int(POSTScrewAntPostAngle))
        screwToImplantTransformationMatrix = np.eye(4)
        screwToImplantTransformationMatrix[:3, :3] = screwToImplant
        screwToImplantTransformationMatrix[:3, 3] = implantPOSTHoleCenter
        self.posteriorScrew.apply_transform(screwToImplantTransformationMatrix)

    def rotatePiecesRTSA(self):

        self.alignScrewWithImplant()

        glenoidCenter = self.preopFileInformation["glenoidImplantCenter"]

        implantML = np.array([1, 0, 0]) # check this by opening step file in a cad software
        implantIS = np.array([0, 1, 0])
        implantPA = np.array([0, 0, 1])

        self.findrTSALoadApplicationPoint()

        preopGlenoidImplantML = np.array([self.preopFileInformation["glenoidImplantLatAxis"]])
        preopGlenoidImplantPA = np.array([self.preopFileInformation["glenoidImplantAntAxis"]])
        preopGlenoidImplantIS = np.array([self.preopFileInformation["glenoidImplantSupAxis"]])
        implantToCTCoordSysRotMat = alignCoordinateSystems(
            implantML,
            implantIS,
            implantPA,
            preopGlenoidImplantML.flatten(),
            preopGlenoidImplantIS.flatten(),
            preopGlenoidImplantPA.flatten()
        )
        implantToScapulaTransformationMatrix = np.eye(4)
        implantToScapulaTransformationMatrix[:3, :3] = implantToCTCoordSysRotMat
        implantToScapulaTransformationMatrix[:3,  3] = glenoidCenter
        self.implantMesh.apply_transform(implantToScapulaTransformationMatrix)
        self.reamerMesh.apply_transform(implantToScapulaTransformationMatrix)
        self.reamerBaseMesh.apply_transform(implantToScapulaTransformationMatrix)

        #implant bone interface mesh
        self.implantBoneInterfaceMesh.apply_transform(implantToScapulaTransformationMatrix)
        implantBoneInterfaceMeshPath = os.path.join(self.dirToSaveMesh(), "implantBoneInterface.stl")
        self.implantBoneInterfaceMesh.export(implantBoneInterfaceMeshPath)

        # implant surface
        self.implantSurfaceMesh.apply_transform(implantToScapulaTransformationMatrix)
        implantSurfaceMeshPath = os.path.join(self.dirToSaveMesh(), "rotatedImplantSurface.stl")
        self.implantSurfaceMesh.export(implantSurfaceMeshPath)

        # screws
        self.centralScrew.apply_transform(implantToScapulaTransformationMatrix)
        self.superiorScrew.apply_transform(implantToScapulaTransformationMatrix)
        self.inferiorScrew.apply_transform(implantToScapulaTransformationMatrix)
        self.posteriorScrew.apply_transform(implantToScapulaTransformationMatrix)
        self.anteriorScrew.apply_transform(implantToScapulaTransformationMatrix)

        # glenosphere
        newrTSALoadApplicationPoint = np.dot(implantToScapulaTransformationMatrix, self.rTSALoadApplicationPoint)
        self.rTSALoadApplicationPoint = newrTSALoadApplicationPoint[:-1]

    def rotatePiecesATSA(self):

        glenoidCenter = self.preopFileInformation["glenoidImplantCenter"]

        implantPA = np.array([1, 0, 0]) # check this by opening step file in a CAD software
        implantML = np.array([0, 1, 0])
        implantIS = np.array([0, 0, 1])

        preopGlenoidImplantML = np.array([self.preopFileInformation["glenoidImplantLatAxis"]])
        preopGlenoidImplantPA = np.array([self.preopFileInformation["glenoidImplantAntAxis"]])
        preopGlenoidImplantIS = np.array([self.preopFileInformation["glenoidImplantSupAxis"]])
        implantToCTCoordSysRotMat = alignCoordinateSystems(
            implantML,
            implantIS,
            implantPA,
            preopGlenoidImplantML.flatten(),
            preopGlenoidImplantIS.flatten(),
            preopGlenoidImplantPA.flatten()
        )
        implantToScapulaTransformationMatrix = np.eye(4)
        implantToScapulaTransformationMatrix[:3, :3] = implantToCTCoordSysRotMat
        implantToScapulaTransformationMatrix[:3,  3] = glenoidCenter
        self.implantMesh.apply_transform(implantToScapulaTransformationMatrix)

        # reamer and reamer-base mesh
        self.reamerMesh.apply_transform(implantToScapulaTransformationMatrix)
        reamerMeshPath = os.path.join(self.dirToSaveMesh(), "reamer.stl")
        self.reamerMesh.export(reamerMeshPath)

        self.reamerBaseMesh.apply_transform(implantToScapulaTransformationMatrix)
        reamerBaseMeshPath = os.path.join(self.dirToSaveMesh(), "reamerBase.stl")
        self.reamerBaseMesh.export(reamerBaseMeshPath)

        # cement mesh
        self.cementMesh.apply_transform(implantToScapulaTransformationMatrix)
        cementMeshPath = os.path.join(self.dirToSaveMesh(), "cement.stl")
        self.cementMesh.export(cementMeshPath)

        # cement-Bone mesh
        self.cementBoneInterfaceMesh.apply_transform(implantToScapulaTransformationMatrix)
        cementBoneInterfaceMeshPath = os.path.join(self.dirToSaveMesh(), "cementBoneInterface.stl")
        self.cementBoneInterfaceMesh.export(cementBoneInterfaceMeshPath)

        # implant cement interface mesh
        self.implantCementInterfaceMesh.apply_transform(implantToScapulaTransformationMatrix)
        implantCementInterfaceMeshPath = os.path.join(self.dirToSaveMesh(), "implantCementInterface.stl")
        self.implantCementInterfaceMesh.export(implantCementInterfaceMeshPath)

        self.cementImplantInterfaceMesh.apply_transform(implantToScapulaTransformationMatrix)
        cementImplantInterfaceMeshPath = os.path.join(self.dirToSaveMesh(), "cementImplantInterface.stl")
        self.cementImplantInterfaceMesh.export(cementImplantInterfaceMeshPath)

        # implant surface
        self.implantSurfaceMesh.apply_transform(implantToScapulaTransformationMatrix)
        implantSurfaceMeshPath = os.path.join(self.dirToSaveMesh(), "rotatedImplantSurface.stl")
        self.implantSurfaceMesh.export(implantSurfaceMeshPath)

    def cutScapulaByImplantCement(self):
        # combinedMesh = trimesh.util.concatenate(self.scapulaMesh, self.implantMesh)
        # Perform the cut with boolean difference
        copiedImplantMesh = copy.deepcopy(self.implantMesh)
        copiedCementMesh = copy.deepcopy(self.cementMesh)

        scapulaMeshPath = os.path.join(self.dirToSaveMesh(), "scapula.stl")
        self.scapulaMesh.export(scapulaMeshPath)

        implantMeshPath = os.path.join(self.dirToSaveMesh(), "rotatedImplant.stl")
        copiedImplantMesh.export(implantMeshPath)

        cementMeshPath = os.path.join(self.dirToSaveMesh(), "cement.stl")
        copiedCementMesh.export(cementMeshPath)

        # We use pymesh docker for boolean operation on the meshes, meaning cutting scapula with reamer and implant
        # For this we use a temp directory for communication with the docker
        tempDirDocker = getConfig()["tempDockerDir"]

        scapulaMeshTempPath = os.path.join(tempDirDocker, "tests", "scapula.stl")
        shutil.copy(scapulaMeshPath, scapulaMeshTempPath)

        implantMeshTempPath = os.path.join(tempDirDocker, "tests", "rotatedImplant.stl")
        shutil.copy(implantMeshPath, implantMeshTempPath)

        cementMeshTempPath = os.path.join(tempDirDocker, "tests", "cement.stl")
        shutil.copy(cementMeshPath, cementMeshTempPath)

        reamerMeshPath = os.path.join(self.dirToSaveMesh(), "reamer.stl")
        reamerMeshTempPath = os.path.join(tempDirDocker, "tests", "reamer.stl")
        shutil.copy(reamerMeshPath, reamerMeshTempPath)

        reamerBaseMeshPath = os.path.join(self.dirToSaveMesh(), "reamerBase.stl")
        reamerBaseMeshTempPath = os.path.join(tempDirDocker, "tests", "reamerBase.stl")
        shutil.copy(reamerBaseMeshPath, reamerBaseMeshTempPath)

        # Create a Docker client
        client = docker.from_env()

        # Define the Docker run command
        pythonCode = [
            "import pymesh",
            "import os",
            "os.chdir('/app')",
            "scapulaMesh = pymesh.load_mesh('scapula.stl')",
            "cementMesh = pymesh.load_mesh('cement.stl')",
            "rotatedImplantMesh = pymesh.load_mesh('rotatedImplant.stl')",
            "reamerMesh = pymesh.load_mesh('reamer.stl')",
            "reamerBaseMesh = pymesh.load_mesh('reamerBase.stl')",
            "cut0 = pymesh.boolean(scapulaMesh, reamerBaseMesh, 'difference')",
            "cut1 = pymesh.boolean(cut0, reamerMesh, 'difference')",
            "cut2 = pymesh.boolean(cut1, cementMesh, 'difference')",
            "cut3 = pymesh.boolean(cut2, rotatedImplantMesh, 'difference')",
            "pymesh.save_mesh('cutScapula.stl', cut3)"
        ]
        pythonCode = ";".join(pythonCode)

        # Run the Docker container
        client.containers.run(
            'pymesh/pymesh',
            command=['python', '-c', pythonCode],
            remove=True,
            volumes={os.path.join(tempDirDocker, "tests"): {'bind': '/app', 'mode': 'rw'}},
            tty=True,
            stdin_open=True,
        )

        shutil.copy(
            os.path.join(tempDirDocker, "tests", "cutScapula.stl"),
            os.path.join(self.dirToSaveMesh(), "cutScapula.stl")
        )

        # We use pymesh docker also for remeshing scapula and implant since trimesh output mesh has intersected facets
        # Create a Docker client
        client = docker.from_env()

        # Define the Docker run command
        pythonCode = [
            "import pymesh",
            "import os",
            "os.chdir('/app')",
            "from fix_mesh import fix_mesh, tet_mesh",
            "scap = pymesh.load_mesh('cutScapula.stl')",
            "scap = fix_mesh(scap, size=5e-3)",
            "pymesh.meshio.save_mesh('cutScapulaPyMesh.stl', scap)",
            "tetgen = pymesh.tetgen()",
            "tetgen.points = scap.vertices",
            "tetgen.triangles  = scap.faces",
            "tetgen.run()",
            "mesh = tetgen.mesh",
            "pymesh.save_mesh_raw('scap.msh', mesh.vertices, mesh.faces, mesh.voxels)",
            "cementMesh = pymesh.load_mesh('cement.stl')",
            "cementMesh = fix_mesh(cementMesh, size=1e-2)",
            "pymesh.meshio.save_mesh('cementMeshPyMesh.stl', cementMesh)",
            "tetgen = pymesh.tetgen()",
            "tetgen.points = cementMesh.vertices",
            "tetgen.triangles  = cementMesh.faces",
            "tetgen.run()",
            "mesh = tetgen.mesh",
            "pymesh.save_mesh_raw('cement.msh', mesh.vertices, mesh.faces, mesh.voxels)"
        ]
        pythonCode = ";".join(pythonCode)

        # Run the Docker container
        client.containers.run(
            'pymesh/pymesh',
            command=['python', '-c', pythonCode],
            remove=True,
            volumes={os.path.join(tempDirDocker, "tests"): {'bind': '/app', 'mode': 'rw'}},
            tty=True,
            stdin_open=True,
        )
        shutil.copy(
            os.path.join(tempDirDocker, "tests", "cutScapulaPyMesh.stl"),
            os.path.join(self.dirToSaveMesh(), "cutScapulaPyMesh.stl")
        )
        shutil.copy(
            os.path.join(tempDirDocker, "tests", "cementMeshPyMesh.stl"),
            os.path.join(self.dirToSaveMesh(), "cementMeshPyMesh.stl")
        )
        shutil.copy(
            os.path.join(tempDirDocker, "tests", "scap.msh"),
            os.path.join(self.dirToSaveMesh(), "cutScapula.msh")
        )
        shutil.copy(
            os.path.join(tempDirDocker, "tests", "cement.msh"),
            os.path.join(self.dirToSaveMesh(), "cement.msh")
        )

        scapMesh = meshio.read(os.path.join(self.dirToSaveMesh(), "cutScapula.msh"))
        scapMesh.write(os.path.join(self.dirToSaveMesh(), "cutScapula3D.inp"))

        self.convert2DMeshTo3D(
            os.path.join(self.dirToSaveMesh(), "rotatedImplant.stl"),
            os.path.join(self.dirToSaveMesh(), "implant3D.inp")
        )
        self.delete2DElements(os.path.join(self.dirToSaveMesh(), "implant3D.inp"))

        cementMesh = meshio.read(os.path.join(self.dirToSaveMesh(), "cement.msh"))
        cementMesh.write(os.path.join(self.dirToSaveMesh(), "cement3D.inp"))

    def cutScapulaByImplantScrew(self):
        #combinedMesh = trimesh.util.concatenate(self.scapulaMesh, self.implantMesh)
        # Perform the cut with boolean difference
        copiedImplantMesh = copy.deepcopy(self.implantMesh)
        copiedReamerMesh = copy.deepcopy(self.reamerMesh)
        copiedReamerBaseMesh = copy.deepcopy(self.reamerBaseMesh)

        scapulaMeshPath = os.path.join(self.dirToSaveMesh(), "scapula.stl")
        self.scapulaMesh.export(scapulaMeshPath)

        implantMeshPath = os.path.join(self.dirToSaveMesh(), "rotatedImplant.stl")
        copiedImplantMesh.export(implantMeshPath)

        reamerMeshPath = os.path.join(self.dirToSaveMesh(), "reamer.stl")
        copiedReamerMesh.export(reamerMeshPath)

        reamerBaseMeshPath = os.path.join(self.dirToSaveMesh(), "reamerBase.stl")
        copiedReamerBaseMesh.export(reamerBaseMeshPath)

        # central screw
        centralScrewMeshPath = os.path.join(self.dirToSaveMesh(), "centralScrewMesh.stl")
        self.centralScrew.export(centralScrewMeshPath)

        # superior screw
        superiorScrewMeshPath = os.path.join(self.dirToSaveMesh(), "superiorScrewMesh.stl")
        self.superiorScrew.export(superiorScrewMeshPath)

        # anterior screw
        anteriorScrewMeshPath = os.path.join(self.dirToSaveMesh(), "anteriorScrewMesh.stl")
        self.anteriorScrew.export(anteriorScrewMeshPath)

        # inferior screw
        inferiorScrewMeshPath = os.path.join(self.dirToSaveMesh(), "inferiorScrewMesh.stl")
        self.inferiorScrew.export(inferiorScrewMeshPath)

        # posterior screw
        posteriorScrewMeshPath = os.path.join(self.dirToSaveMesh(), "posteriorScrewMesh.stl")
        self.posteriorScrew.export(posteriorScrewMeshPath)

        # We use pymesh docker for boolean operation on the meshes, meaning cutting scapula with reamer and implant
        # For this we use a temp directory for communication with the docker
        tempDirDocker = getConfig()["tempDockerDir"]

        scapulaMeshTempPath = os.path.join(tempDirDocker, "tests", "scapula.stl")
        shutil.copy(scapulaMeshPath, scapulaMeshTempPath)

        implantMeshTempPath = os.path.join(tempDirDocker, "tests", "rotatedImplant.stl")
        shutil.copy(implantMeshPath, implantMeshTempPath)

        reamerMeshTempPath = os.path.join(tempDirDocker, "tests", "reamer.stl")
        shutil.copy(reamerMeshPath, reamerMeshTempPath)

        reamerBaseMeshTempPath = os.path.join(tempDirDocker, "tests", "reamerBase.stl")
        shutil.copy(reamerBaseMeshPath, reamerBaseMeshTempPath)

        centralScrewMeshTempPath = os.path.join(tempDirDocker, "tests", "centralScrewMesh.stl")
        shutil.copy(centralScrewMeshPath, centralScrewMeshTempPath)

        superiorScrewMeshTempPath = os.path.join(tempDirDocker, "tests", "superiorScrewMesh.stl")
        shutil.copy(superiorScrewMeshPath, superiorScrewMeshTempPath)

        inferiorScrewMeshTempPath = os.path.join(tempDirDocker, "tests", "inferiorScrewMesh.stl")
        shutil.copy(inferiorScrewMeshPath, inferiorScrewMeshTempPath)

        anteriorScrewMeshTempPath = os.path.join(tempDirDocker, "tests", "anteriorScrewMesh.stl")
        shutil.copy(anteriorScrewMeshPath, anteriorScrewMeshTempPath)

        posteriorScrewMeshTempPath = os.path.join(tempDirDocker, "tests", "posteriorScrewMesh.stl")
        shutil.copy(posteriorScrewMeshPath, posteriorScrewMeshTempPath)

        # Create a Docker client
        client = docker.from_env()

        # Define the Docker run command
        pythonCode = [
            "import pymesh",
            "import os",
            "os.chdir('/app')",
            "scapulaMesh = pymesh.load_mesh('scapula.stl')",
            "reamerMesh = pymesh.load_mesh('reamer.stl')",
            "reamerBase = pymesh.load_mesh('reamerBase.stl')",
            "rotatedImplantMesh = pymesh.load_mesh('rotatedImplant.stl')",
            "centralScrewMesh = pymesh.load_mesh('centralScrewMesh.stl')",
            "superiorScrewMesh = pymesh.load_mesh('superiorScrewMesh.stl')",
            "inferiorScrewMesh = pymesh.load_mesh('inferiorScrewMesh.stl')",
            "anteriorScrewMesh = pymesh.load_mesh('anteriorScrewMesh.stl')",
            "posteriorScrewMesh = pymesh.load_mesh('posteriorScrewMesh.stl')",
            "cut1 = pymesh.boolean(scapulaMesh, reamerMesh, 'difference')",
            "cut2 = pymesh.boolean(cut1, reamerBase, 'difference')",
            "cut3 = pymesh.boolean(cut2, centralScrewMesh, 'difference')",
            "cut4 = pymesh.boolean(cut3, superiorScrewMesh, 'difference')",
            "cut5 = pymesh.boolean(cut4, inferiorScrewMesh, 'difference')",
            "cut6 = pymesh.boolean(cut5, anteriorScrewMesh, 'difference')",
            "cut7 = pymesh.boolean(cut6, posteriorScrewMesh, 'difference')",
            "pymesh.save_mesh('cutScapula.stl', cut7)"
        ]
        pythonCode = ";".join(pythonCode)

        # Run the Docker container
        client.containers.run(
            'pymesh/pymesh',
            command=['python', '-c', pythonCode],
            remove=True,
            volumes={os.path.join(tempDirDocker, "tests"): {'bind': '/app', 'mode': 'rw'}},
            tty=True,
            stdin_open=True,
        )

        shutil.copy(
            os.path.join(tempDirDocker, "tests", "cutScapula.stl"),
            os.path.join(self.dirToSaveMesh(), "cutScapula.stl")
        )


        # We use pymesh docker also for remeshing scapula and implant since trimesh output mesh has intersected facets
        # Create a Docker client
        client = docker.from_env()

        # Define the Docker run command
        pythonCode = [
            "import pymesh",
            "import os",
            "os.chdir('/app')",
            "from fix_mesh import fix_mesh, tet_mesh",
            "scap = pymesh.load_mesh('cutScapula.stl')",
            "scap = fix_mesh(scap)",
            "pymesh.meshio.save_mesh('cutScapulaPyMesh.stl', scap)",
            "imp = pymesh.load_mesh('rotatedImplant.stl')",
            "imp = fix_mesh(imp, size=5e-3)",
            "pymesh.meshio.save_mesh('rotatedImplantPyMesh.stl', imp)",
            "tetgen = pymesh.tetgen()",
            "tetgen.points = imp.vertices",
            "tetgen.triangles  = imp.faces",
            "tetgen.run()",
            "mesh = tetgen.mesh",
            "pymesh.save_mesh_raw('rotatedImplantPyMesh.msh', mesh.vertices, mesh.faces, mesh.voxels)"
        ]
        pythonCode = ";".join(pythonCode)

        # Run the Docker container
        client.containers.run(
            'pymesh/pymesh',
            command=['python', '-c', pythonCode],
            remove=True,
            volumes={os.path.join(tempDirDocker, "tests"): {'bind': '/app', 'mode': 'rw'}},
            tty=True,
            stdin_open=True,
        )
        shutil.copy(
            os.path.join(tempDirDocker, "tests", "cutScapulaPyMesh.stl"),
            os.path.join(self.dirToSaveMesh(), "cutScapulaPyMesh.stl")
        )
        shutil.copy(
            os.path.join(tempDirDocker, "tests", "rotatedImplantPyMesh.msh"),
            os.path.join(self.dirToSaveMesh(), "rotatedImplantPyMesh.msh")
        )

        self.convert2DMeshTo3D(
            os.path.join(self.dirToSaveMesh(), "cutScapulaPyMesh.stl"),
            os.path.join(self.dirToSaveMesh(), "cutScapula3D.inp"),
            quickFix=True
        )
        self.delete2DElements(os.path.join(self.dirToSaveMesh(), "cutScapula3D.inp"))

        implantMesh = meshio.read(os.path.join(self.dirToSaveMesh(), "rotatedImplantPyMesh.msh"))
        implantMesh.write(os.path.join(self.dirToSaveMesh(), "implant3D.inp"))

        # screws
        self.convert2DMeshTo3D(
            os.path.join(self.dirToSaveMesh(), "centralScrewMesh.stl"),
            os.path.join(self.dirToSaveMesh(), "centralScrewMesh3D.inp")
        )
        self.delete2DElements(os.path.join(self.dirToSaveMesh(), "centralScrewMesh3D.inp"))

        self.convert2DMeshTo3D(
            os.path.join(self.dirToSaveMesh(), "inferiorScrewMesh.stl"),
            os.path.join(self.dirToSaveMesh(), "inferiorScrewMesh3D.inp")
        )
        self.delete2DElements(os.path.join(self.dirToSaveMesh(), "inferiorScrewMesh3D.inp"))

        self.convert2DMeshTo3D(
            os.path.join(self.dirToSaveMesh(), "anteriorScrewMesh.stl"),
            os.path.join(self.dirToSaveMesh(), "anteriorScrewMesh3D.inp")
        )
        self.delete2DElements(os.path.join(self.dirToSaveMesh(), "anteriorScrewMesh3D.inp"))

        self.convert2DMeshTo3D(
            os.path.join(self.dirToSaveMesh(), "superiorScrewMesh.stl"),
            os.path.join(self.dirToSaveMesh(), "superiorScrewMesh3D.inp")
        )
        self.delete2DElements(os.path.join(self.dirToSaveMesh(), "superiorScrewMesh3D.inp"))

        self.convert2DMeshTo3D(
            os.path.join(self.dirToSaveMesh(), "posteriorScrewMesh.stl"),
            os.path.join(self.dirToSaveMesh(), "posteriorScrewMesh3D.inp")
        )
        self.delete2DElements(os.path.join(self.dirToSaveMesh(), "posteriorScrewMesh3D.inp"))

    def cutScapulaByImplantScrewUnionMesh(self):
        #combinedMesh = trimesh.util.concatenate(self.scapulaMesh, self.implantMesh)
        # Perform the cut with boolean difference
        copiedImplantMesh = copy.deepcopy(self.implantMesh)
        copiedReamerMesh = copy.deepcopy(self.reamerMesh)
        copiedReamerBaseMesh = copy.deepcopy(self.reamerBaseMesh)

        scapulaMeshPath = os.path.join(self.dirToSaveMesh(), "scapula.stl")
        self.scapulaMesh.export(scapulaMeshPath)

        implantMeshPath = os.path.join(self.dirToSaveMesh(), "rotatedImplant.stl")
        copiedImplantMesh.export(implantMeshPath)

        reamerMeshPath = os.path.join(self.dirToSaveMesh(), "reamer.stl")
        copiedReamerMesh.export(reamerMeshPath)

        reamerBaseMeshPath = os.path.join(self.dirToSaveMesh(), "reamerBase.stl")
        copiedReamerBaseMesh.export(reamerBaseMeshPath)

        # central screw
        centralScrewMeshPath = os.path.join(self.dirToSaveMesh(), "centralScrewMesh.stl")
        self.centralScrew.export(centralScrewMeshPath)

        # superior screw
        superiorScrewMeshPath = os.path.join(self.dirToSaveMesh(), "superiorScrewMesh.stl")
        self.superiorScrew.export(superiorScrewMeshPath)

        # anterior screw
        anteriorScrewMeshPath = os.path.join(self.dirToSaveMesh(), "anteriorScrewMesh.stl")
        self.anteriorScrew.export(anteriorScrewMeshPath)

        # inferior screw
        inferiorScrewMeshPath = os.path.join(self.dirToSaveMesh(), "inferiorScrewMesh.stl")
        self.inferiorScrew.export(inferiorScrewMeshPath)

        # posterior screw
        posteriorScrewMeshPath = os.path.join(self.dirToSaveMesh(), "posteriorScrewMesh.stl")
        self.posteriorScrew.export(posteriorScrewMeshPath)

        # We use pymesh docker for boolean operation on the meshes, meaning cutting scapula with reamer and implant
        # For this we use a temp directory for communication with the docker
        tempDirDocker = getConfig()["tempDockerDir"]

        scapulaMeshTempPath = os.path.join(tempDirDocker, "tests", "scapula.stl")
        shutil.copy(scapulaMeshPath, scapulaMeshTempPath)

        implantMeshTempPath = os.path.join(tempDirDocker, "tests", "rotatedImplant.stl")
        shutil.copy(implantMeshPath, implantMeshTempPath)

        reamerMeshTempPath = os.path.join(tempDirDocker, "tests", "reamer.stl")
        shutil.copy(reamerMeshPath, reamerMeshTempPath)

        reamerBaseMeshTempPath = os.path.join(tempDirDocker, "tests", "reamerBase.stl")
        shutil.copy(reamerBaseMeshPath, reamerBaseMeshTempPath)

        centralScrewMeshTempPath = os.path.join(tempDirDocker, "tests", "centralScrewMesh.stl")
        shutil.copy(centralScrewMeshPath, centralScrewMeshTempPath)

        superiorScrewMeshTempPath = os.path.join(tempDirDocker, "tests", "superiorScrewMesh.stl")
        shutil.copy(superiorScrewMeshPath, superiorScrewMeshTempPath)

        inferiorScrewMeshTempPath = os.path.join(tempDirDocker, "tests", "inferiorScrewMesh.stl")
        shutil.copy(inferiorScrewMeshPath, inferiorScrewMeshTempPath)

        anteriorScrewMeshTempPath = os.path.join(tempDirDocker, "tests", "anteriorScrewMesh.stl")
        shutil.copy(anteriorScrewMeshPath, anteriorScrewMeshTempPath)

        posteriorScrewMeshTempPath = os.path.join(tempDirDocker, "tests", "posteriorScrewMesh.stl")
        shutil.copy(posteriorScrewMeshPath, posteriorScrewMeshTempPath)

        # Create a Docker client
        client = docker.from_env()

        # Define the Docker run command
        pythonCode = [
            "import pymesh",
            "import os",
            "os.chdir('/app')",
            "scapulaMesh = pymesh.load_mesh('scapula.stl')",
            "reamerMesh = pymesh.load_mesh('reamer.stl')",
            "reamerBase = pymesh.load_mesh('reamerBase.stl')",
            "rotatedImplantMesh = pymesh.load_mesh('rotatedImplant.stl')",
            "centralScrewMesh = pymesh.load_mesh('centralScrewMesh.stl')",
            "superiorScrewMesh = pymesh.load_mesh('superiorScrewMesh.stl')",
            "inferiorScrewMesh = pymesh.load_mesh('inferiorScrewMesh.stl')",
            "anteriorScrewMesh = pymesh.load_mesh('anteriorScrewMesh.stl')",
            "posteriorScrewMesh = pymesh.load_mesh('posteriorScrewMesh.stl')",
            "cut1 = pymesh.boolean(scapulaMesh, reamerMesh, 'difference')",
            "cut2 = pymesh.boolean(cut1, reamerBase, 'difference')",
            "cut3 = pymesh.boolean(cut2, centralScrewMesh, 'difference')",
            "cut4 = pymesh.boolean(cut3, superiorScrewMesh, 'difference')",
            "cut5 = pymesh.boolean(cut4, inferiorScrewMesh, 'difference')",
            "cut6 = pymesh.boolean(cut5, anteriorScrewMesh, 'difference')",
            "cut7 = pymesh.boolean(cut6, posteriorScrewMesh, 'difference')",
            "pymesh.save_mesh('cutScapula.stl', cut7)",
            "implantScrews0 = pymesh.boolean(rotatedImplantMesh, centralScrewMesh, 'union')",
            "implantScrews1 = pymesh.boolean(implantScrews0, superiorScrewMesh, 'union')",
            "implantScrews2 = pymesh.boolean(implantScrews1, inferiorScrewMesh, 'union')",
            "implantScrews3 = pymesh.boolean(implantScrews2, anteriorScrewMesh, 'union')",
            "implantScrews = pymesh.boolean(implantScrews3, posteriorScrewMesh, 'union')",
            "pymesh.save_mesh('implantScrews.stl', implantScrews)"
        ]
        pythonCode = ";".join(pythonCode)

        # Run the Docker container
        client.containers.run(
            'pymesh/pymesh',
            command=['python', '-c', pythonCode],
            remove=True,
            volumes={os.path.join(tempDirDocker, "tests"): {'bind': '/app', 'mode': 'rw'}},
            tty=True,
            stdin_open=True,
        )

        shutil.copy(
            os.path.join(tempDirDocker, "tests", "cutScapula.stl"),
            os.path.join(self.dirToSaveMesh(), "cutScapula.stl")
        )


        # We use pymesh docker also for remeshing scapula and implant since trimesh output mesh has intersected facets
        # Create a Docker client
        client = docker.from_env()

        # Define the Docker run command
        pythonCode = [
            "import pymesh",
            "import os",
            "os.chdir('/app')",
            "from fix_mesh import fix_mesh, tet_mesh",
            "scap = pymesh.load_mesh('cutScapula.stl')",
            "scap = fix_mesh(scap)",
            "pymesh.meshio.save_mesh('cutScapulaPyMesh.stl', scap)",
            "imp = pymesh.load_mesh('rotatedImplant.stl')",
            "imp = fix_mesh(imp, size=5e-3)",
            "pymesh.meshio.save_mesh('rotatedImplantPyMesh.stl', imp)",
            "impScrews = pymesh.load_mesh('implantScrews.stl')",
            "impScrews = fix_mesh(impScrews, size=1e-2)",
            "pymesh.meshio.save_mesh('implantScrewsPyMesh.stl', impScrews)",
            "tetgen = pymesh.tetgen()",
            "tetgen.points = impScrews.vertices",
            "tetgen.triangles  = impScrews.faces",
            "tetgen.run()",
            "mesh = tetgen.mesh",
            "pymesh.save_mesh_raw('implantScrewsVol.msh', mesh.vertices, mesh.faces, mesh.voxels)"
        ]
        pythonCode = ";".join(pythonCode)

        # Run the Docker container
        client.containers.run(
            'pymesh/pymesh',
            command=['python', '-c', pythonCode],
            remove=True,
            volumes={os.path.join(tempDirDocker, "tests"): {'bind': '/app', 'mode': 'rw'}},
            tty=True,
            stdin_open=True,
        )
        shutil.copy(
            os.path.join(tempDirDocker, "tests", "cutScapulaPyMesh.stl"),
            os.path.join(self.dirToSaveMesh(), "cutScapulaPyMesh.stl")
        )
        shutil.copy(
            os.path.join(tempDirDocker, "tests", "rotatedImplantPyMesh.stl"),
            os.path.join(self.dirToSaveMesh(), "rotatedImplantPyMesh.stl")
        )
        shutil.copy(
            os.path.join(tempDirDocker, "tests", "implantScrewsPyMesh.stl"),
            os.path.join(self.dirToSaveMesh(), "implantScrewsPyMesh.stl")
        )
        shutil.copy(
            os.path.join(tempDirDocker, "tests", "implantScrewsVol.msh"),
            os.path.join(self.dirToSaveMesh(), "implantScrewsVol.msh")
        )

        self.convert2DMeshTo3D(
            os.path.join(self.dirToSaveMesh(), "cutScapulaPyMesh.stl"),
            os.path.join(self.dirToSaveMesh(), "cutScapula3D.inp"),
            quickFix=True
        )
        self.delete2DElements(os.path.join(self.dirToSaveMesh(), "cutScapula3D.inp"))

        self.convert2DMeshTo3D(
            os.path.join(self.dirToSaveMesh(), "rotatedImplantPyMesh.stl"),
            os.path.join(self.dirToSaveMesh(), "implant3D.inp")
        )
        self.delete2DElements(os.path.join(self.dirToSaveMesh(), "implant3D.inp"))

        # screws
        self.convert2DMeshTo3D(
            os.path.join(self.dirToSaveMesh(), "centralScrewMesh.stl"),
            os.path.join(self.dirToSaveMesh(), "centralScrewMesh3D.inp")
        )
        self.delete2DElements(os.path.join(self.dirToSaveMesh(), "centralScrewMesh3D.inp"))

        self.convert2DMeshTo3D(
            os.path.join(self.dirToSaveMesh(), "inferiorScrewMesh.stl"),
            os.path.join(self.dirToSaveMesh(), "inferiorScrewMesh3D.inp")
        )
        self.delete2DElements(os.path.join(self.dirToSaveMesh(), "inferiorScrewMesh3D.inp"))

        self.convert2DMeshTo3D(
            os.path.join(self.dirToSaveMesh(), "anteriorScrewMesh.stl"),
            os.path.join(self.dirToSaveMesh(), "anteriorScrewMesh3D.inp")
        )
        self.delete2DElements(os.path.join(self.dirToSaveMesh(), "anteriorScrewMesh3D.inp"))

        self.convert2DMeshTo3D(
            os.path.join(self.dirToSaveMesh(), "superiorScrewMesh.stl"),
            os.path.join(self.dirToSaveMesh(), "superiorScrewMesh3D.inp")
        )
        self.delete2DElements(os.path.join(self.dirToSaveMesh(), "superiorScrewMesh3D.inp"))

        self.convert2DMeshTo3D(
            os.path.join(self.dirToSaveMesh(), "posteriorScrewMesh.stl"),
            os.path.join(self.dirToSaveMesh(), "posteriorScrewMesh3D.inp")
        )
        self.delete2DElements(os.path.join(self.dirToSaveMesh(), "posteriorScrewMesh3D.inp"))

        # implant-screws

        implantScrewsMesh = meshio.read(os.path.join(self.dirToSaveMesh(), "implantScrewsVol.msh"))
        implantScrewsMesh.write(os.path.join(self.dirToSaveMesh(), "implantScrews3D.inp"))

    def convert2DMeshTo3D(self, mesh, outputName, quickFix=False, targetfacenum=20000):

        """
        meshSet = pymeshlab.MeshSet()
        try:
            meshSet.load_new_mesh(mesh)
        except:
            raise Exception("Could not find the 2D mesh")
        meshSet.meshing_merge_close_vertices()
        meshSet.meshing_remove_duplicate_faces()
        meshSet.meshing_remove_duplicate_vertices()
        meshSet.apply_filter(
            'simplification_quadric_edge_collapse_decimation',
            targetfacenum=targetfacenum,
            preservenormal=True
        )
        meshSet.save_current_mesh(mesh)
        """
        if quickFix:
            pymeshfix.clean_from_file(mesh, mesh)
        # ensure that gmsh has not a model already in from last simulation (causes PLC error)
        if (gmsh.isInitialized()):
            gmsh.finalize()
        gmsh.initialize()
        gmsh.merge(mesh)  # import stl file to gmsh
        gmsh.option.setNumber("General.Terminal", 1)
        gmsh.option.setNumber("Mesh.Algorithm3D", 1)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.01)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 10)
        gmsh.option.setNumber("Mesh.Optimize", 1)
        gmsh.option.setNumber("Mesh.QualityType", 1)

        n = gmsh.model.getDimension()
        s = gmsh.model.getEntities(n)
        l = gmsh.model.geo.addSurfaceLoop([s[i][1] for i in range(len(s))])
        gmsh.model.geo.addVolume([l])
        gmsh.model.occ.removeAllDuplicates()
        gmsh.model.geo.synchronize()
        gmsh.model.mesh.generate(3)
        gmsh.write(outputName)
        gmsh.finalize()

    def generateInpFileForMeshingScapulaAndImplantWithAbaqus(self):
        abaqusScriptPath = os.path.join(getConfig()["codeRootDir"], "ShoulderCase", "AutoFE", "inpForRTSAMeshingWithAbaqus.py")

        abaqusTempDir = getConfig()["AbaqusTempDir"]
        codeRootDir = getConfig()["codeRootDir"]
        abaqusBatDir = getConfig()["abaqusBat"]

        saveMeshDir = self.dirToSaveMesh()
        os.chdir(abaqusTempDir)
        subprocess.run([abaqusBatDir,
                        "cae",
                        rf"noGUI={abaqusScriptPath}",
                        "--",
                        rf"{saveMeshDir}"],
                       text=True,
                       shell=True)
        os.chdir(codeRootDir)
        runAbaqus(codeRootDir, abaqusScriptPath, saveMeshDir)

        abaqusTempDirFiles = os.listdir(abaqusTempDir)
        for file in abaqusTempDirFiles:
            source = os.path.join(abaqusTempDir, file)
            if file == "inpForMeshingImplantWithAbaqus.inp" or file == "inpForMeshingScapulaWithAbaqus.inp":
                dir_ = os.path.join(self.dirToSaveMesh(), file)
                shutil.move(source, dir_)
            else:
                os.remove(source)

    def delete2DElements(self, filename):

        with open(os.path.join(self.dirToSaveMesh(), filename), "r") as file:
            lines = file.readlines()
        threeDMeshDataFrame = pd.DataFrame(lines, columns=['data'])

        beginingItem = 'CPS3'  # 2D element
        endingItem = 'C3D4'  # tetrahedral element

        # find lines containing begining Item and ending Item
        begingIdx = threeDMeshDataFrame.index[threeDMeshDataFrame['data'].str.contains(beginingItem)]
        endingIdx = threeDMeshDataFrame.index[threeDMeshDataFrame['data'].str.contains(endingItem)]

        idx = begingIdx[0].astype(int)
        idx2 = endingIdx[0].astype(int)
        threeDMeshDataFrame = threeDMeshDataFrame.drop(threeDMeshDataFrame.index[idx:idx2], 0)


        np.savetxt(os.path.join(self.dirToSaveMesh(), filename),
                   threeDMeshDataFrame.values,
                   fmt="%s",
                   delimiter="\t",
                   newline="")

    def putTSABoundaryConditionsDataInDatabaseWithAbaqusMesh(self):

        boundaryConditions = {}
        outputFileName = "boundaryConditionsWithImplant.json"
        boundaryConditions["loadDir"] = self.shoulder.scapula.coordSys.ML.tolist()
        inpFileDir = os.path.join(self.dirToSaveMesh(), "inpForMeshingImplantWithAbaqus.inp")
        implantInpFileDataFrame = inpFileToDataFrame(inpFileDir, 'Node', 'C3D10')

        boundaryConditions["referencePoint"] = self.shoulder.scapula.glenoid.fittedSphere.center.flatten().tolist()

        implantSurfacePath = os.path.join(self.dirToSaveMesh(), "rotatedImplantSurface.stl")
        p, t, _ = loadStl(implantSurfacePath, mode=1)
        implantSurface = p.round(0).tolist()
        isImplantSurface = False
        for idx, value in enumerate(implantSurface):
            isImplantSurface = isImplantSurface + \
                               implantInpFileDataFrame[1].eq(implantSurface[idx][0]) * \
                               implantInpFileDataFrame[2].eq(implantSurface[idx][1]) * \
                               implantInpFileDataFrame[3].eq(implantSurface[idx][2])

        implantSurfaceNodes = implantInpFileDataFrame[0].loc[isImplantSurface]
        implantSurfaceNodes = (implantSurfaceNodes - 1).astype(np.int32).tolist()
        boundaryConditions["implantSurfaceNodes"] = implantSurfaceNodes

        inpFileDir = os.path.join(self.dirToSaveMesh(), "inpForMeshingScapulaWithAbaqus.inp")
        scapulaInpFileDataFrame = inpFileToDataFrame(inpFileDir, 'Node', 'C3D10')

        # Defining the line between AG and TS which defines the border of boundary condition

        AI = self.shoulder.scapula.angulusInferior
        TS = self.shoulder.scapula.trigonumSpinae
        slopeXY = (AI[0][1] - TS[0][1]) / (AI[0][0] - TS[0][0])
        xyLine = (scapulaInpFileDataFrame[2]) - slopeXY * (scapulaInpFileDataFrame[1])

        checkBC = xyLine.ge(TS[0][1] - slopeXY * TS[0][0]) * scapulaInpFileDataFrame[3].le(TS[0][2])

        BCBox = scapulaInpFileDataFrame[0].loc[checkBC]
        BCBox = (BCBox - 1).astype(np.int32).tolist()

        boundaryConditions["BCBox"] = BCBox

        with open(os.path.join(self.dirToSaveMesh(), outputFileName), "w") as f:
            json.dump(boundaryConditions, f)

    def putTSABoundaryConditionsDataInDatabase(self):

        boundaryConditions = {}
        outputFileName = "boundaryConditionsWithImplant.json"

        # we use the developed deep learning model to predict MSM results
        # https://www.sciencedirect.com/science/article/pii/S0021929024000290
        # Features should be provided in the following order
        # sex, weight, height, version, inclination,
        # supraspinatus_csa, infraspinatus_csa, subscapularis_csa, teres_minor_csa,
        # implant, activity, abduction_angle
        dicomFilesPath = os.path.join(self.shoulder.SCase.dataCTPath, "dicom")
        dicomFile = os.listdir(dicomFilesPath)[0]
        dicomInfo = pydicom.dcmread(os.path.join(dicomFilesPath, dicomFile))


        try:
            sex = 1 if dicomInfo.PatientSex == "M" else 2
            height = float(dicomInfo.PatientSize)
            weight = float(dicomInfo.PatientWeight)
            version = self.preopFileInformation["glenoidImplantVersion"]
            inclination = self.preopFileInformation["glenoidImplantInclination"]
            if sex == 1:
                supraspinatus_csa, infraspinatus_csa, subscapularis_csa, teres_minor_csa = 5.5e-4, 7.0e-4, 12.0e-4, 2.5e-4
            else:
                supraspinatus_csa, infraspinatus_csa, subscapularis_csa, teres_minor_csa = 4.0e-4, 5.5e-4, 10.0e-4, 2.0e-4

            if self.preopFileInformation["implantType"] == "Anatomic":
                implant = 1
            elif self.preopFileInformation["implantType"] == "Reversed":
                implant = 2

            activity = 2
            abduction_angle = 60
            X = np.array([
                [sex, weight, height, version, inclination,
                 supraspinatus_csa, infraspinatus_csa, subscapularis_csa, teres_minor_csa,
                 implant, activity, abduction_angle]]
            )
            modelPath = os.path.join(os.getcwd(), "ShoulderCase", "MSM", "model.h5")
            model = load_model(modelPath, custom_objects={"coeff_determination": coeff_determination})
            loadDir = (-model.predict(X).flatten()).tolist()

        except:
            loadMagnitude = 0.8*70*9.81
            loadDir = (self.shoulder.scapula.coordSys.ML * loadMagnitude).tolist()

        boundaryConditions["loadDir"] = loadDir

        inpFileDir = os.path.join(self.dirToSaveMesh(), "implant3D.inp")
        implantInpFileDataFrame = inpFileToDataFrame(inpFileDir, 'NODE', 'C3D4')

        if self.preopFileInformation["implantType"] == "Anatomic":
            boundaryConditions["referencePoint"] = self.shoulder.scapula.glenoid.fittedSphere.center.flatten().tolist()
        elif self.preopFileInformation["implantType"] == "Reversed":
            boundaryConditions["referencePoint"] = self.rTSALoadApplicationPoint.tolist()

        implantSurfacePath = os.path.join(self.dirToSaveMesh(), "rotatedImplantSurface.stl")
        p, t, _ = loadStl(implantSurfacePath, mode=1)
        implantSurface = p.round(0).tolist()
        isImplantSurface = False
        for idx, value in enumerate(implantSurface):
            isImplantSurface = isImplantSurface + \
                               implantInpFileDataFrame[1].eq(implantSurface[idx][0]) * \
                               implantInpFileDataFrame[2].eq(implantSurface[idx][1]) * \
                               implantInpFileDataFrame[3].eq(implantSurface[idx][2])

        implantSurfaceNodes = implantInpFileDataFrame[0].loc[isImplantSurface]
        implantSurfaceNodes = (implantSurfaceNodes - 1).astype(np.int32).tolist()
        boundaryConditions["implantSurfaceNodes"] = implantSurfaceNodes

        inpFileDir = os.path.join(self.dirToSaveMesh(), "cutScapula3D.inp")
        scapulaInpFileDataFrame = inpFileToDataFrame(inpFileDir, 'NODE', 'C3D4')

        if self.preopFileInformation["implantType"] == "Anatomic":

            inpFileDir = os.path.join(self.dirToSaveMesh(), "cement3D.inp")
            cementInpFileDataFrame = inpFileToDataFrame(inpFileDir, 'NODE', 'C3D4')

            cementBoneInterfaceMeshPath = os.path.join(self.dirToSaveMesh(), "cementBoneInterface.stl")
            p, t, _ = loadStl(cementBoneInterfaceMeshPath, mode=1)
            cementBoneInterface = p.round(0).tolist()
            isCementBoneInterface = findPointsWithinDistance(
                cementInpFileDataFrame.iloc[:, 1:].values,
                cementBoneInterface,
                threshold=1
            )
            cementBoneInterfaceNodes = cementInpFileDataFrame[0].loc[isCementBoneInterface]
            cementBoneInterfaceNodes = (cementBoneInterfaceNodes - 1).astype(np.int32).tolist()
            boundaryConditions["cementBoneInterface"] = cementBoneInterfaceNodes

            # cement-implant interface nodes
            cementImplantInterfaceMeshPath = os.path.join(self.dirToSaveMesh(), "cementImplantInterface.stl")
            p, t, _ = loadStl(cementImplantInterfaceMeshPath, mode=1)
            cementImplantInterface = p.round(0).tolist()
            isCementImplantInterface = findPointsWithinDistance(
                cementInpFileDataFrame.iloc[:, 1:].values,
                cementImplantInterface,
                threshold=1
            )
            cementImplantInterfaceNodes = cementInpFileDataFrame[0].loc[isCementImplantInterface]
            cementImplantInterfaceNodes = (cementImplantInterfaceNodes - 1).astype(np.int32).tolist()
            boundaryConditions["cementImplantInterface"] = cementImplantInterfaceNodes

            # implant-cement interface nodes
            implantCementInterfaceMeshPath = os.path.join(self.dirToSaveMesh(), "implantCementInterface.stl")
            p, t, _ = loadStl(implantCementInterfaceMeshPath, mode=1)
            implantCementInterface = p.round(0).tolist()
            isImplantCementInterface = findPointsWithinDistance(
                implantInpFileDataFrame.iloc[:, 1:].values,
                implantCementInterface,
                threshold=1
            )
            implantCementInterfaceNodes = implantInpFileDataFrame[0].loc[isImplantCementInterface]
            implantCementInterfaceNodes = (implantCementInterfaceNodes - 1).astype(np.int32).tolist()
            boundaryConditions["implantCementInterface"] = implantCementInterfaceNodes

        elif self.preopFileInformation["implantType"] == "Reversed":

            # bone-implant interface nodes
            implantBoneInterfaceMeshPath = os.path.join(self.dirToSaveMesh(), "implantBoneInterface.stl")
            p, t, _ = loadStl(implantBoneInterfaceMeshPath, mode=1)
            implantBoneInterface = p.round(0).tolist()
            isImplantBoneInterface = findPointsWithinDistance(
                scapulaInpFileDataFrame.iloc[:, 1:].values,
                implantBoneInterface,
                threshold=1
            )
            implantBoneInterfaceNodes = scapulaInpFileDataFrame[0].loc[isImplantBoneInterface]
            implantBoneInterfaceNodes = (implantBoneInterfaceNodes - 1).astype(np.int32).tolist()
            boundaryConditions["implantBoneInterface"] = implantBoneInterfaceNodes

            # bone-screws interface
            # central screw
            centralScrewMesh = os.path.join(self.dirToSaveMesh(), "centralScrewMesh.stl")
            p, t, _ = loadStl(centralScrewMesh, mode=1)
            centralScrewBoneInterface = p.round(0).tolist()
            isCentralScrewBoneInterface = findPointsWithinDistance(
                scapulaInpFileDataFrame.iloc[:, 1:].values,
                centralScrewBoneInterface,
                threshold=1
            )
            centralScrewBoneInterfaceNodes = scapulaInpFileDataFrame[0].loc[isCentralScrewBoneInterface]
            centralScrewInterfaceNodes = (centralScrewBoneInterfaceNodes - 1).astype(np.int32).tolist()

            # superior screw
            superiorScrewMesh = os.path.join(self.dirToSaveMesh(), "superiorScrewMesh.stl")
            p, t, _ = loadStl(superiorScrewMesh, mode=1)
            superiorScrewBoneInterface = p.round(0).tolist()
            isSuperiorScrewBoneInterface = findPointsWithinDistance(
                scapulaInpFileDataFrame.iloc[:, 1:].values,
                superiorScrewBoneInterface,
                threshold=1
            )
            superiorScrewBoneInterfaceNodes = scapulaInpFileDataFrame[0].loc[isSuperiorScrewBoneInterface]
            superiorScrewInterfaceNodes = (superiorScrewBoneInterfaceNodes - 1).astype(np.int32).tolist()

            # inferior screw
            inferiorScrewMesh = os.path.join(self.dirToSaveMesh(), "inferiorScrewMesh.stl")
            p, t, _ = loadStl(inferiorScrewMesh, mode=1)
            inferiorScrewBoneInterface = p.round(0).tolist()
            isInferiorScrewBoneInterface = findPointsWithinDistance(
                scapulaInpFileDataFrame.iloc[:, 1:].values,
                inferiorScrewBoneInterface,
                threshold=1
            )
            inferiorScrewBoneInterfaceNodes = scapulaInpFileDataFrame[0].loc[isInferiorScrewBoneInterface]
            inferiorScrewInterfaceNodes = (inferiorScrewBoneInterfaceNodes - 1).astype(np.int32).tolist()

            # anterior screw
            anteriorScrewMesh = os.path.join(self.dirToSaveMesh(), "anteriorScrewMesh.stl")
            p, t, _ = loadStl(anteriorScrewMesh, mode=1)
            anteriorScrewBoneInterface = p.round(0).tolist()
            isAnteriorScrewBoneInterface = findPointsWithinDistance(
                scapulaInpFileDataFrame.iloc[:, 1:].values,
                anteriorScrewBoneInterface,
                threshold=1
            )
            anteriorScrewBoneInterfaceNodes = scapulaInpFileDataFrame[0].loc[isAnteriorScrewBoneInterface]
            anteriorScrewInterfaceNodes = (anteriorScrewBoneInterfaceNodes - 1).astype(np.int32).tolist()

            # posterior screw
            posteriorScrewMesh = os.path.join(self.dirToSaveMesh(), "posteriorScrewMesh.stl")
            p, t, _ = loadStl(posteriorScrewMesh, mode=1)
            posteriorScrewBoneInterface = p.round(0).tolist()
            isPosteriorScrewBoneInterface = findPointsWithinDistance(
                scapulaInpFileDataFrame.iloc[:, 1:].values,
                posteriorScrewBoneInterface,
                threshold=1
            )
            posteriorScrewBoneInterfaceNodes = scapulaInpFileDataFrame[0].loc[isPosteriorScrewBoneInterface]
            posteriorScrewInterfaceNodes = (posteriorScrewBoneInterfaceNodes - 1).astype(np.int32).tolist()

            boundaryConditions["screwsBoneInterface"] = centralScrewInterfaceNodes + \
                                                        superiorScrewInterfaceNodes + \
                                                        anteriorScrewInterfaceNodes + \
                                                        inferiorScrewInterfaceNodes + \
                                                        posteriorScrewInterfaceNodes

        # Defining the line between AG and TS which defines the border of boundary condition
        AI = self.shoulder.scapula.angulusInferior
        TS = self.shoulder.scapula.trigonumSpinae
        slopeXY = (AI[0][1] - TS[0][1]) / (AI[0][0] - TS[0][0])
        xyLine = (scapulaInpFileDataFrame[2]) - slopeXY * (scapulaInpFileDataFrame[1])

        checkBC = xyLine.ge(TS[0][1] - slopeXY * TS[0][0]) * scapulaInpFileDataFrame[3].le(TS[0][2])

        BCBox = scapulaInpFileDataFrame[0].loc[checkBC]
        BCBox = (BCBox - 1).astype(np.int32).tolist()

        boundaryConditions["BCBox"] = BCBox

        with open(os.path.join(self.dirToSaveMesh(), outputFileName), "w") as f:
            json.dump(boundaryConditions, f)

    def performATSAOnTheMesh(self):
        self.rotatePiecesATSA()
        self.cutScapulaByImplantCement()

    def performRTSAOnTheMesh(self):
        self.rotatePiecesRTSA()
        self.cutScapulaByImplantScrew()

    def performRTSAOnTheUnionMesh(self):
        self.rotatePiecesRTSA()
        self.cutScapulaByImplantScrewUnionMesh()

    def performTSAWithAbaqus(self, jobScriptPath):

        abaqusTempDir = getConfig()["AbaqusTempDir"]
        codeRootDir = getConfig()["codeRootDir"]
        abaqusBatDir = getConfig()["abaqusBat"]

        saveMeshDir = self.dirToSaveMesh()
        os.chdir(abaqusTempDir)
        subprocess.run([abaqusBatDir,
                        "cae",
                        rf"noGUI={jobScriptPath}",
                        "--",
                        rf"{saveMeshDir}"],
                       text=True,
                       shell=True)
        os.chdir(codeRootDir)

        abaqusTempDirFiles = os.listdir(abaqusTempDir)
        for file in abaqusTempDirFiles:
            source = os.path.join(abaqusTempDir, file)
            if not os.path.exists(os.path.join(self.dirToSaveMesh(), "Abaqus")):
                os.mkdir(os.path.join(self.dirToSaveMesh(), "Abaqus"))
            dir_ = os.path.join(self.dirToSaveMesh(), "Abaqus", file)
            shutil.move(source, dir_)

    def performAbaqusJob(self):

        abaqusTempDir = getConfig()["AbaqusTempDir"]
        codeRootDir = getConfig()["codeRootDir"]
        abaqusBatDir = getConfig()["abaqusBat"]

        if self.preopFileInformation["implantType"] == "Anatomic":
            inpFileDir = os.path.join(self.dirToSaveMesh(), "Abaqus", "inpForATSA.inp")
            implant = "ATSA"
            shutil.move(inpFileDir, abaqusTempDir)
            jobScriptPath = os.path.join(os.getcwd(), "ShoulderCase", "AutoFE", "aTSASaveMetricScript.py")
            jobName = "aTSA"
        elif self.preopFileInformation["implantType"] == "Reversed":
            inpFileDir = os.path.join(self.dirToSaveMesh(), "Abaqus", "inpForRTSA.inp")
            implant = "RTSA"
            shutil.move(inpFileDir, abaqusTempDir)
            jobScriptPath = os.path.join(os.getcwd(), "ShoulderCase", "AutoFE", "rTSASaveMetricScript.py")
            jobName = "rTSA"

        os.chdir(abaqusTempDir)
        subprocess.run([abaqusBatDir,
                        f"job={jobName}",
                        rf"input=inpFor{implant}.inp",
                        "interactive"],
                       text=True,
                       shell=True)

        # save metrics, currently maximal principal strain on the bone-implant interface
        subprocess.run([abaqusBatDir,
                        "cae",
                        rf"noGUI={jobScriptPath}",
                        "--",
                        rf"{abaqusTempDir}"],
                       text=True,
                       shell=True)

        os.chdir(codeRootDir)

        abaqusTempDirFiles = os.listdir(abaqusTempDir)
        for file in abaqusTempDirFiles:
            source = os.path.join(abaqusTempDir, file)
            if not os.path.exists(os.path.join(self.dirToSaveMesh(), "Abaqus", "Job")):
                os.mkdir(os.path.join(self.dirToSaveMesh(), "Abaqus", "Job"))
            dir_ = os.path.join(self.dirToSaveMesh(), "Abaqus", "Job", file)
            shutil.move(source, dir_)


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

def alignCoordinateSystems(uA, vA, wA, uB, vB, wB):
    # Form matrices with basis vectors
    AMatrix = np.column_stack((uA, vA, wA))
    BMatrix = np.column_stack((uB, vB, wB))
    # Calculate the rotation matrix
    RMatrix = np.dot(BMatrix, np.linalg.inv(AMatrix))
    return RMatrix

def rotationMatrixToEulerAngles(matrix):
    rx = np.arctan2(matrix[2, 1],  matrix[2, 2])
    ry = np.arctan2(-matrix[2, 0], np.sqrt(matrix[2, 1]**2+matrix[2, 2]**2))
    rz = np.arctan2(matrix[1, 0],  matrix[0, 0])
    return 180/np.pi*rx, 180/np.pi*ry, 180/np.pi*rz

def findPointsWithinDistance(arr1, arr2, threshold, chunkSize=1000):
    indicesArr1 = []
    for i in range(0, len(arr1), chunkSize):
        chunkArr1 = arr1[i:i + chunkSize]

        for j in range(0, len(arr2), chunkSize):
            chunkArr2 = arr2[j:j + chunkSize]

            distances = np.linalg.norm(chunkArr1[:, np.newaxis, :] - chunkArr2, axis=2)
            indicesChunk = np.any(distances < threshold, axis=1)
            indicesArr1.extend(np.nonzero(indicesChunk)[0] + i)

    return np.array(indicesArr1)

def rotationMatrixFromAngles(thetaX, thetaY, thetaZ):
    # Euler angles in radians
    thetaX = np.radians(thetaX)  # Rotation around X axis
    thetaY = np.radians(thetaY)  # Rotation around Y axis
    thetaZ = np.radians(thetaZ)  # Rotation around Z axis

    # Create rotation matrices for each axis
    rotX = Rotation.from_euler('x', thetaX)
    rotY = Rotation.from_euler('y', thetaY)
    rotZ = Rotation.from_euler('z', thetaZ)

    # Combine the rotation matrices to get the final rotation matrix
    return (rotX * rotY * rotZ).as_matrix()

def coeff_determination(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true-y_pred )) # Sum of squared residuals
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )  # Total sum of squares
    return ( 1 - SS_res/(SS_tot + K.epsilon()) ) # Calculate R^2
