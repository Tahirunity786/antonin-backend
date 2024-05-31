import pandas as pd
import os
from ShoulderCase.AutoFE.AbaqusInpFile import AbaqusInpFile
from ShoulderCase.AutoFE.DicomVolumeForFE import DicomVolumeForFE
import numpy as np

class FEMaterial:

    def __init__(self, mesh, dicomSetPath):
        self.mesh = mesh
        self.abaqusInpFile = []
        self.dicomSetForFE = DicomVolumeForFE(dicomSetPath)

    def huToDensity(self, hu):
        """
            From Yasmine BOULANAACHE thesis: Overcorrected Implants for Total Shoulder Arthroplasty
            EPFL Thèse n° 7443, Présentée le 28 janvier 2021
            densities are in g/cm3
        """
        roCT = hu / 1460
        roApp = 2.192 * roCT + 0.007
        roAsh = 0.6 * roApp
        densityTransition = 1.2
        isCortical = roApp > densityTransition

        return roApp, roAsh, isCortical

    def densityToE(self, density, conversionFormula="L3"):
        """
            Compute Young modulus from Yasmine BOULANAACHE Overcorrected Implants for Total Shoulder Arthroplasty
            need to uncomment the chosen conversion and return its result
            make sure to match the model with the density transition defined in huToDensity()
            conversion_type = 1 -> L_28, = 2 -> L_34, 0 or other -> L_3 (default as better results)
        """
        roApp, roAsh, isCortical = density[0], density[1], density[2]

        if conversionFormula == "L28":
            E = 60 + 900 * roApp**2

        elif conversionFormula == "L34":
            E = roApp
            for i in range(len(roApp)):
                if (isCortical[i] > 0):
                    E[i] = 90 * pow(roApp[i], 7.4)
                else:
                    E[i] = 60 + 900 * pow((roApp[i]), 2)

        elif conversionFormula == "L3":
            E = np.zeros(len(roApp))
            corticalMask = isCortical > 0
            E[corticalMask] = 10200 * np.power(roAsh[corticalMask], 2.01)
            trabecularMask = ~corticalMask
            E[trabecularMask] = 15000 * np.power(roApp[trabecularMask] / 1.8, 2)

        return E

    def calculateTransitionalMatrix(self):
        return [float(self.slices[0].ImagePositionPatient[0]),
                float(self.slices[0].ImagePositionPatient[1]),
                float(self.slices[0].ImagePositionPatient[2])]

    def getBoneElasticModel(self, sampledHU, deviation):
        """
          The Elastic type is ISOTROPIC so the three columns of its table data are:
          Young's modulus | Poisson's ratio | Temperature (replaced here by HU)

          The calibration lines have been copied from previous implementation but
          are missing some explanations.

          The output array size and more generally the length of sampled HU is also
          missing explanation.

          https://classes.engineering.wustl.edu/2009/spring/mase5513/abaqus/docs/v6.5/books/usb/default.htm?startat=pt04ch10s02abm02.html#usb-mat-clinearelastic
          https://classes.engineering.wustl.edu/2009/spring/mase5513/abaqus/docs/v6.5/books/key/default.htm?startat=ch05abk03.html#usb-kws-melastic
          https://classes.engineering.wustl.edu/2009/spring/mase5513/abaqus/docs/v6.5/books/ker/default.htm?startat=pt01ch22pyo01.html
        """
        # Poisson's ratio might become a config variable
        poissonRatio = 0.26

        # Elastic model is initialised with calibration values
        elasticModel = np.array([[1, poissonRatio / (1 + deviation / 100), -2000],
                                 [1, poissonRatio / (1 + deviation / 100), 0]])

        # Elastic model next values are built with the input parameters
        density = self.huToDensity(sampledHU)

        youngModulus = self.densityToE(density)

        sampledElasticModel = np.stack([youngModulus, poissonRatio / (1 + deviation / 100) * np.ones(len(sampledHU)), sampledHU]).transpose().squeeze()
        elasticModel = np.append(elasticModel, sampledElasticModel, axis=0)

        # Elastic model final line also contains calibration values
        elasticModel = np.append(elasticModel, [[20000, poissonRatio / (1 + deviation / 100), 2000]], axis=0) * (1 + deviation / 100)

        # removes lines that are there twices to avoid error in abaqus
        # ABAQUS "The independent variables must be arranged in ascending order"

        elasticModel = np.unique(elasticModel, axis=0)
        return elasticModel

    def exportPartDensity(self, partName):
        """
        Export density values for the given part of the given model.
        """
        nodesHUFilename = os.path.join(self.mesh.getDataPath(), f"{partName}HU.inp")
        nodesHU = pd.read_csv(nodesHUFilename, header=None)[1].to_numpy()
        nodesDensity = self.huToDensity(nodesHU)
        np.savetxt(os.path.join(self.mesh.getDataPath(), f"{partName}Density.inp"),
                   np.c_[nodesDensity[0], nodesDensity[1], nodesDensity[2]], delimiter=",", fmt="%.3f")

    def exportPartYoungModulus(self, partName):
        """
        Export Young modulus values for the given part of the given model.
        """
        densityFilename = os.path.join(self.mesh.getDataPath(), f"{partName}Density.inp")
        ro_app = pd.read_csv(densityFilename, header=None)[0].to_numpy()
        ro_ash = pd.read_csv(densityFilename, header=None)[1].to_numpy()
        is_cortical = pd.read_csv(densityFilename, header=None)[2].to_numpy()
        density = [ro_app, ro_ash, is_cortical]
        E = self.densityToE(density)
        np.savetxt(os.path.join(self.mesh.getDataPath(), f"{partName}E.inp"), E, fmt="%.3f")

    def exportBoneElasticModel(self, partName):
        """
        Create and export an Abaqus linear elastic model for the "Bone" material
        using the HU values of the given part of the given model.
        The HU values are used to measure the bone density scaled by the given
        gender and age of the patient.
        """
        nodesHUFilename = os.path.join(self.mesh.getDataPath(), f"{partName}HU.inp")
        nodesHU = pd.read_csv(nodesHUFilename, header=None)[1].to_numpy()

        # Choose 50 positive values of HU to create the material elastic model
        nodesPositiveHU = nodesHU[nodesHU > 0]
        nodesIndices = np.linspace(0, nodesPositiveHU.size, 50 + 1)[:-1].astype("int")
        sampledNodesHU = np.sort(nodesPositiveHU)[nodesIndices]
        deviation = 0
        boneElasticModel = self.getBoneElasticModel(sampledNodesHU, deviation)
        exportFilename = "BoneElasticModel.inp"
        np.savetxt(os.path.join(self.mesh.getDataPath(), exportFilename), boneElasticModel, delimiter=",", fmt="%.3f")

    def assignMaterialAsTemperature(self):
        """
        Assign material as temperature based on hu values
        """
        with open(self.mesh.getDataPath() + os.sep + "inputFileWithoutMaterial.inp", "r") as file:
            lines = file.readlines()
        threeDMeshDataFrame = pd.DataFrame(lines, columns=['data'])
        beginingTetSearch = "elset=VOLUME1, instance=PART-1-1"
        endingTetSearch = 'End Part'
        endAssemblySearch = 'End Assembly'

        beginingTet = threeDMeshDataFrame.index[threeDMeshDataFrame['data'].str.contains(beginingTetSearch)]
        endingTet = threeDMeshDataFrame.index[threeDMeshDataFrame['data'].str.contains(endingTetSearch)]
        endAssembly = threeDMeshDataFrame.index[threeDMeshDataFrame['data'].str.contains(endAssemblySearch)]
        beginingTetIdx = beginingTet[0].astype(int) + 1
        endingTetIdx = endingTet[0].astype(int) - 1

        sectionAssigning = '*Solid Section, elset=ES_VOLUME_0_MAT100, material=PART-1\n'
        creatingSection = '** Section: Section-1-ES_VOLUME_0_MAT100\n'
        numbersInSectionSet = threeDMeshDataFrame['data'][beginingTetIdx]
        definingSectionSet = '*Elset, elset=ES_VOLUME_0_MAT100, generate\n'
        HUTable = '*INCLUDE, INPUT=PART-1_HU.inp\n'
        initialConditions = '*INITIAL CONDITION, type = temperature\n'
        elasticModel = '*INCLUDE, INPUT=BoneElasticModel.inp\n'
        materialType = '*Elastic\n'
        materialName = '*Material, name=PART-1\n'
        starsText = '**\n'
        MaterialSectionText = '**Material\n'

        threeDMeshDataFrame = insertRow(endingTetIdx, threeDMeshDataFrame, sectionAssigning)
        threeDMeshDataFrame = insertRow(endingTetIdx, threeDMeshDataFrame, creatingSection)
        threeDMeshDataFrame = insertRow(endingTetIdx, threeDMeshDataFrame, numbersInSectionSet)
        threeDMeshDataFrame = insertRow(endingTetIdx, threeDMeshDataFrame, definingSectionSet)

        endAssembly = threeDMeshDataFrame.index[threeDMeshDataFrame['data'].str.contains(endAssemblySearch)]
        endAssemblyIdx = endAssembly[0].astype(int) + 1

        # "fills the input file with the material properties based on HU values"
        threeDMeshDataFrame = insertRow(endAssemblyIdx, threeDMeshDataFrame, initialConditions)
        threeDMeshDataFrame = insertRow(endAssemblyIdx, threeDMeshDataFrame, HUTable)
        threeDMeshDataFrame = insertRow(endAssemblyIdx, threeDMeshDataFrame, initialConditions)
        threeDMeshDataFrame = insertRow(endAssemblyIdx, threeDMeshDataFrame, elasticModel)
        threeDMeshDataFrame = insertRow(endAssemblyIdx, threeDMeshDataFrame, materialType)
        threeDMeshDataFrame = insertRow(endAssemblyIdx, threeDMeshDataFrame, materialName)
        threeDMeshDataFrame = insertRow(endAssemblyIdx, threeDMeshDataFrame, starsText)
        threeDMeshDataFrame = insertRow(endAssemblyIdx, threeDMeshDataFrame, MaterialSectionText)

        # "Fills the input file with an output request to see the temperature distribution"
        endFieldOutputSearch = 'HISTORY OUTPUT'
        endFieldOutput = threeDMeshDataFrame.index[threeDMeshDataFrame['data'].str.contains(endFieldOutputSearch)]
        endFieldOutputIdx = endFieldOutput[0].astype(int) - 1

        fieldOutput = '** FIELD OUTPUT: F-Output-2\n'
        nodeOutput = '*Node Output\n'
        paramsOutput = 'CF, NT, RF, U\n'
        starsText = '**\n'

        threeDMeshDataFrame = insertRow(endFieldOutputIdx, threeDMeshDataFrame, starsText)
        threeDMeshDataFrame = insertRow(endFieldOutputIdx, threeDMeshDataFrame, paramsOutput)
        threeDMeshDataFrame = insertRow(endFieldOutputIdx, threeDMeshDataFrame, nodeOutput)
        threeDMeshDataFrame = insertRow(endFieldOutputIdx, threeDMeshDataFrame, fieldOutput)

        # need to change PART-1-1 to PART-1 to avoid errors linking material and temperatures
        threeDMeshDataFrame = threeDMeshDataFrame.replace(regex=['PART-1-1'], value=['PART-1'])

        threeDMeshList = threeDMeshDataFrame.values.tolist()

        fileName = os.path.join(self.mesh.getDataPath(), "inputFileWithMaterial.inp")

        with open(fileName, "w") as f:
            for element in threeDMeshList:
                f.write(f"{element[0]}")

    def exportPartHU(self, inputInpFile, partName):

        self.abaqusInpFile = AbaqusInpFile(inputInpFile)
        nodesOfPart = self.abaqusInpFile.getNodesCoordinatesOfPart(partName)
        nodesOfPartLabels = [label for label in nodesOfPart.keys()]
        nodesOfPartCoordinates = np.array([coordinates for coordinates in nodesOfPart.values()])
        translatedNodesCoordinates = nodesOfPartCoordinates - self.dicomSetForFE.calculateTransitionalMatrix()
        transformedNodesCoordinates = (self.dicomSetForFE.calculateTransformationMatrix().dot(np.transpose(translatedNodesCoordinates))).T
        if (self.dicomSetForFE.patientPositionZ[0] > self.dicomSetForFE.patientPositionZ[-1]):
            transformedNodesCoordinates = transformedNodesCoordinates * [1, 1, -1]
        nodesHU = self.dicomSetForFE.getFilteredHUAtCoordinates(transformedNodesCoordinates)

        # HU value is associated to the part's name and the node's label
        labeledHU = pd.DataFrame(
            {"label": [partName + "." + str(label) for label in nodesOfPartLabels],
             "value": nodesHU}
        )
        HUFilename = os.path.join(self.mesh.getDataPath(), f"{partName}HU.inp")
        labeledHU.to_csv(HUFilename, header=False, index=False)

    def exportPartDensity(self, partName):
        """
        Export density values for the given part of the given model.
        """
        nodesHUFilename = os.path.join(self.mesh.getDataPath(), f"{partName}HU.inp")
        nodesHU = pd.read_csv(nodesHUFilename, header=None)[1].to_numpy()
        nodesDensity = self.huToDensity(nodesHU)
        np.savetxt(os.path.join(self.mesh.getDataPath(), f"{partName}Density.inp"),
                   np.c_[nodesDensity[0], nodesDensity[1], nodesDensity[2]], delimiter=",", fmt="%.3f")

    def exportPartYoungModulus(self, partName):
        """
        Export Young modulus values for the given part of the given model.
        """
        densityFilename = os.path.join(self.mesh.getDataPath(), f"{partName}Density.inp")
        ro_app = pd.read_csv(densityFilename, header=None)[0].to_numpy()
        ro_ash = pd.read_csv(densityFilename, header=None)[1].to_numpy()
        is_cortical = pd.read_csv(densityFilename, header=None)[2].to_numpy()
        density = [ro_app, ro_ash, is_cortical]
        E = self.densityToE(density)
        np.savetxt(os.path.join(self.mesh.getDataPath(), f"{partName}E.inp"), E, fmt="%.3f")

    def saveBoneElasticModelToInpTSA(self):

        inputInpFile = os.path.join(self.mesh.getDataPath(), "inpForMeshingScapulaWithAbaqus.inp")
        partName = "scapulaPart"
        self.exportPartHU(inputInpFile, partName)
        # Part density and young modulus can be exported for further analysis
        saveValuesForAnalysis = True
        if (saveValuesForAnalysis):
            self.exportPartDensity(partName)
            self.exportPartYoungModulus(partName)

        self.exportBoneElasticModel(partName)


    def assignMaterialFromDicom(self, implant=False):

        if not implant:
            inputInpFile = os.path.join(self.mesh.getDataPath(), "inputFileWithoutMaterial.inp")
            partName = "PART-1"
        else:
            inputInpFile = os.path.join(self.mesh.getDataPath(), "cutScapula3D.inp")
            partName = "scapulaPart"

        self.exportPartHU(inputInpFile, partName)

        # Part density and young modulus can be exported for further analysis
        saveValuesForAnalysis = True
        if (saveValuesForAnalysis):
            self.exportPartDensity(partName)
            self.exportPartYoungModulus(partName)

        self.exportBoneElasticModel(partName)

        if not implant:
            self.assignMaterialAsTemperature()

    def exportBoneMaterialFromDicomToInpFile(self):

        inputInpFile = os.path.join(self.mesh.getDataPath(), "aTSAInputFileWithoutMaterial.inp")
        partName = "scapula#PART-1"

        self.exportPartHU(inputInpFile, partName)

        # Part density and young modulus can be exported for further analysis
        saveValuesForAnalysis = True
        if (saveValuesForAnalysis):
            self.exportPartDensity(partName)
            self.exportPartYoungModulus(partName)

        self.exportBoneElasticModel(partName)

def insertRow(rowNumber, df, value):
    line = pd.DataFrame({"data": value}, index=[rowNumber + 0.5])
    df = pd.concat([df, line], ignore_index=False)
    df = df.sort_index().reset_index(drop=True)
    return df