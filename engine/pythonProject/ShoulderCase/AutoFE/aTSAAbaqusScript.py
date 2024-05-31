from abaqus import *
from abaqusConstants import *
from part import *
from material import *
from section import *
from optimization import *
from assembly import *
from step import *
from interaction import *
from load import *
from mesh import *
from job import *
from sketch import *
from visualization import *
from connectorBehavior import *
import os
import sys
import json
import displayGroupMdbToolset as dgm
import math
import numpy as np

def calculateCenter(points):
    # Use NumPy to sum the coordinates along each axis
    totalCoordinates = np.sum(points, axis=0)
    # Calculate the average coordinates to get the center
    center = totalCoordinates / len(points)
    return center

def calculate3dDistance(point1, point2):
    # Extract coordinates
    x1, y1, z1 = point1
    x2, y2, z2 = point2
    # Calculate Euclidean distance
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
    return distance

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

def aTSAWithAbaqus(inputPath):

    scapula = os.path.join(inputPath, 'cutScapula3D.inp')
    scapulaModel = mdb.ModelFromInputFile(inputFileName=scapula, name='scapula')
    scapulaRootAssembly = mdb.models['scapula'].rootAssembly
    scapulaInstance = mdb.models['scapula'].rootAssembly.instances['PART-1-1']
    scapulaNodes = scapulaInstance.nodes
    scapulaElements = scapulaInstance.elements

    implant = os.path.join(inputPath, 'implant3D.inp')
    implantModel = mdb.ModelFromInputFile(inputFileName=implant, name='implant')
    implantRootAssembly = mdb.models['implant'].rootAssembly
    implantInstance = mdb.models['implant'].rootAssembly.instances['PART-1-1']
    implantNodes = implantInstance.nodes

    cement = os.path.join(inputPath, 'cement3D.inp')
    cementModel = mdb.ModelFromInputFile(inputFileName=cement, name='cement')
    cementRootAssembly = mdb.models['cement'].rootAssembly
    cementInstance = mdb.models['cement'].rootAssembly.instances['PART-1-1']
    cementNodes = cementInstance.nodes

    boundaryConditionsFile = os.path.join(inputPath, 'boundaryConditionsWithImplant.json')
    with open(boundaryConditionsFile) as f:
        boundaryConditionsData = json.load(f)

    implantSurfaceNode = boundaryConditionsData["implantSurfaceNodes"]

    cementBoneInterface = boundaryConditionsData["cementBoneInterface"]
    cementImplantInterface = boundaryConditionsData["cementImplantInterface"]

    implantCementInterface = boundaryConditionsData["implantCementInterface"]

    BCBoxNode = boundaryConditionsData["BCBox"]

    referencePoint = boundaryConditionsData["referencePoint"]
    loadDirection = boundaryConditionsData["loadDir"]

    # implant surface node
    abaqusImplantSurfaceNodeList = implantNodes[implantSurfaceNode[0]:implantSurfaceNode[0] + 1]
    for idx, nodeIn in enumerate(implantSurfaceNode):
        abaqusImplantSurfaceNodeList = abaqusImplantSurfaceNodeList + \
                                       implantNodes[implantSurfaceNode[idx]:implantSurfaceNode[idx] + 1]

    abaqusCementImplantSurfaceNodeList = cementNodes[cementImplantInterface[0]:cementImplantInterface[0] + 1]
    for idx, nodeIn in enumerate(cementImplantInterface):
        abaqusCementImplantSurfaceNodeList = abaqusCementImplantSurfaceNodeList + \
                                             cementNodes[cementImplantInterface[idx]:cementImplantInterface[idx] + 1]

    abaqusImplantCementSurfaceNodeList = implantNodes[implantCementInterface[0]:implantCementInterface[0] + 1]
    for idx, nodeIn in enumerate(implantCementInterface):
        abaqusImplantCementSurfaceNodeList = abaqusImplantCementSurfaceNodeList + \
                                             implantNodes[implantCementInterface[idx]:implantCementInterface[idx] + 1]

    abaqusBCNodeList = scapulaNodes[BCBoxNode[0]:BCBoxNode[0] + 1]
    for idx2, nodeIn2 in enumerate(BCBoxNode):
        abaqusBCNodeList = abaqusBCNodeList + scapulaNodes[BCBoxNode[idx2]:BCBoxNode[idx2] + 1]

    abaqusCementBoneInterfaceNodeList = cementNodes[cementBoneInterface[0]:cementBoneInterface[0] + 1]
    for idx2, nodeIn2 in enumerate(cementBoneInterface):
        abaqusCementBoneInterfaceNodeList = abaqusCementBoneInterfaceNodeList + \
                                            cementNodes[cementBoneInterface[idx2]:cementBoneInterface[idx2] + 1]

    elementCenters = np.zeros((len(scapulaElements), 3))
    for elementIndex, element in enumerate(scapulaElements):
        elementNodes = element.connectivity
        elementNodesCoordinates = np.array([scapulaNodes[i].coordinates for i in elementNodes])
        elementCenters[elementIndex] = calculateCenter(elementNodesCoordinates)

    threshold = 2  # mm
    abaqusCementBoneInterfaceNodeCoordinates = np.array(
        [node.coordinates for node in abaqusCementBoneInterfaceNodeList]
    )
    abaqusCementBoneInterfaceElementIndices = findPointsWithinDistance(
        elementCenters,
        abaqusCementBoneInterfaceNodeCoordinates,
        threshold
    )
    abaqusCementBoneInterfaceElementList = scapulaElements[abaqusCementBoneInterfaceElementIndices[0]: \
                                                            abaqusCementBoneInterfaceElementIndices[0] + 1]
    for idx, elementIn in enumerate(abaqusCementBoneInterfaceElementIndices):
        abaqusCementBoneInterfaceElementList = abaqusCementBoneInterfaceElementList + \
                                                scapulaElements[abaqusCementBoneInterfaceElementIndices[idx]: \
                                                                abaqusCementBoneInterfaceElementIndices[idx] + 1]

    # abaqus set for BC
    scapulaRootAssembly.Set(name='BC', nodes=abaqusBCNodeList)

    # abaqus set for bone-implant interface elements
    scapulaRootAssembly.Set(name='boneCementInterfaceElements', elements=abaqusCementBoneInterfaceElementList)

    # abaqus set for RF
    implantRootAssembly.ReferencePoint(point=(referencePoint[0], referencePoint[1], referencePoint[2]))
    implantRootAssembly.Set(name='referencePoint', referencePoints=(implantRootAssembly.referencePoints[5],))
    implantRootAssembly.Set(name='MPCSlave', nodes=abaqusImplantSurfaceNodeList)
    implantRootAssembly.Set(name='implantCementInterface', nodes=abaqusImplantCementSurfaceNodeList)

    # cement
    cementRootAssembly.Set(name='cementImplantInterface', nodes=abaqusCementImplantSurfaceNodeList)
    cementRootAssembly.Set(name='cementBoneInterface', nodes=abaqusCementBoneInterfaceNodeList)

    # assembly model
    assemblyModel = mdb.models['Model-1'].rootAssembly
    assemblyModel.Instance(name='scapula', model=scapulaModel)
    assemblyModel.Instance(name='implant', model=implantModel)
    assemblyModel.Instance(name='cement', model=cementModel)
    scapulaInstance = assemblyModel.instances['scapula.PART-1-1']
    implantInstance = assemblyModel.instances['implant.PART-1-1']
    cementInstance  = assemblyModel.instances['cement.PART-1-1']

    mdb.models['Model-1'].StaticStep(name='Step-1', previous='Initial')
    mdb.models['Model-1'].fieldOutputRequests['F-Output-1'].setValues(
        variables=('S', 'PE', 'PEEQ', 'PEMAG', 'LE', 'U', 'RF', 'CF', 'CSTRESS', 'CDISP', 'EVOL')
    )
    # create MPC constraint which the master is the refrence point and the
    # slave is the glenoid surface with using link for connecting the master and slave
    mdb.models['Model-1'].MultipointConstraint(
        controlPoint=assemblyModel.sets['implant.referencePoint'],
        csys=None,
        mpcType=LINK_MPC,
        name='Constraint-1',
        surface=assemblyModel.sets['implant.MPCSlave'],
        userMode=DOF_MODE_MPC,
        userType=0
    )

    # creating load based on the magnitude and the direction of ML
    mdb.models['Model-1'].ConcentratedForce(
        cf1=loadDirection[0],
        cf2=loadDirection[1],
        cf3=loadDirection[2],
        createStepName='Step-1',
        distributionType=UNIFORM,
        field='',
        localCsys=None,
        name='Load-1',
        region=assemblyModel.sets['implant.referencePoint']
    )

    mdb.models['Model-1'].DisplacementBC(
        amplitude=UNSET,
        createStepName='Step-1',
        distributionType=UNIFORM,
        fieldName='',
        fixed=OFF,
        localCsys=None,
        name='BC-1',
        region=assemblyModel.sets['scapula.BC'],
        u1=0.0,
        u2=0.0,
        u3=0.0,
        ur1=UNSET,
        ur2=UNSET,
        ur3=UNSET
    )

    # implant material
    mdb.models['implant'].Material(name='imp')
    # material properties of the Tornier implant https://onlinelibrary.wiley.com/doi/full/10.1002/jor.23115
    mdb.models['implant'].materials['imp'].Elastic(table=((110000.0,0.3),))
    mdb.models['implant'].HomogeneousSolidSection(material='imp', name='Section-1', thickness=None)
    mdb.models['implant'].parts['PART-1'].SectionAssignment(
        offset=0.0,
        offsetField='',
        offsetType=MIDDLE_SURFACE,
        region=Region(elements=mdb.models['implant'].parts['PART-1'].elements),
        sectionName='Section-1',
        thicknessAssignment=FROM_SECTION
    )

    # cement material
    mdb.models['cement'].Material(name='cem')
    # cement material properties from https://jeo-esska.springeropen.com/articles/10.1186/s40634-022-00494-8#Sec2
    mdb.models['cement'].materials['cem'].Elastic(table=((2000.0, 0.3),))
    mdb.models['cement'].HomogeneousSolidSection(material='cem', name='Section-1', thickness=None)
    mdb.models['cement'].parts['PART-1'].SectionAssignment(
        offset=0.0,
        offsetField='',
        offsetType=MIDDLE_SURFACE,
        region=Region(elements=mdb.models['cement'].parts['PART-1'].elements),
        sectionName='Section-1',
        thicknessAssignment=FROM_SECTION
    )

    # scapula model
    mdb.models['scapula'].Material(name='bone')
    boneElasticModelFile = os.path.join(inputPath, 'BoneElasticModel.inp')
    boneElasticModelData = []
    with open(boneElasticModelFile) as f:
        for line in f:
            line = line.strip()
            values = line.split(',')
            floatValues = tuple(map(float, values))
            boneElasticModelData.append(floatValues)
    boneElasticModelTable = tuple(boneElasticModelData)

    mdb.models['scapula'].materials['bone'].Elastic(table=boneElasticModelTable, temperatureDependency=ON)
    mdb.models['scapula'].HomogeneousSolidSection(material='bone', name='Section-1', thickness=None)
    mdb.models['scapula'].parts['PART-1'].SectionAssignment(
        offset=0.0,
        offsetField='', offsetType=MIDDLE_SURFACE,
        region=Region(elements=mdb.models['scapula'].parts['PART-1'].elements),
        sectionName='Section-1',
        thicknessAssignment=FROM_SECTION
    )

    scapulaPartHUFile = os.path.join(inputPath, 'scapulaPartHU.inp')
    huData = []
    nodesNumbers = []
    with open(scapulaPartHUFile) as f:
        for i, line in enumerate(f):
            line = line.strip()
            values = line.split(',')[-1]
            floatValues = float(values)
            huData.append(floatValues)
            nodesNumbers.append(i + 1)
    discFieldData = ('scapula#PART-1-1', 1, tuple(nodesNumbers), tuple(huData))
    mdb.models['Model-1'].DiscreteField(
        data=(discFieldData,),
        dataWidth=1,
        defaultValues=(0.0,),
        description='',
        fieldType=SCALAR,
        location=NODES,
        name='DiscField-1'
    )

    mdb.models['scapula'].parts['PART-1'].Set(
        name='scapulaNodes',
        nodes=mdb.models['scapula'].parts['PART-1'].nodes,
    )

    implantNodesLabels = set(node.label for node in mdb.models['implant'].parts['PART-1'].nodes)
    MPCNodes = set(node.label for node in assemblyModel.sets['implant.MPCSlave'].nodes)
    implantNodesWithoutMPCNodes = implantNodesLabels - MPCNodes
    implantNodesWithoutMPCNodes = list(implantNodesWithoutMPCNodes)
    mdb.models['implant'].parts['PART-1'].SetFromNodeLabels(
        name='implantNodes',
        nodeLabels=implantNodesWithoutMPCNodes,
    )

    mdb.models['Model-1'].Temperature(
        createStepName='Initial',
        distributionType=DISCRETE_FIELD,
        field='DiscField-1',
        magnitudes=(1.0,),
        name='Predefined Field-1',
        region=mdb.models['Model-1'].rootAssembly.sets['scapula.PART-1-1.scapulaNodes']
    )

    # define tie
    mdb.models['Model-1'].Tie(
        adjust=OFF,
        main=mdb.models['Model-1'].rootAssembly.instances['scapula.PART-1-1'].sets['scapulaNodes'],
        name='tie1',
        positionToleranceMethod=COMPUTED,
        secondary=mdb.models['Model-1'].rootAssembly.sets['cement.cementBoneInterface'],
        thickness=ON,
        tieRotations=ON
    )

    cementBoneInterfaceNodes = set(node.label for node in assemblyModel.sets['cement.cementBoneInterface'].nodes)
    cementImplantInterfaceNodes = set(node.label for node in assemblyModel.sets['cement.cementImplantInterface'].nodes)
    remainedCementImplantInterfaceNodes = cementImplantInterfaceNodes - cementBoneInterfaceNodes
    remainedCementImplantInterfaceNodes = list(remainedCementImplantInterfaceNodes)
    mdb.models['cement'].parts['PART-1'].SetFromNodeLabels(
        name='remainedCementImplantInterface',
        nodeLabels=remainedCementImplantInterfaceNodes,
    )

    mdb.models['Model-1'].rootAssembly.regenerate()

    mdb.models['Model-1'].Tie(
        adjust=OFF,
        main=mdb.models['Model-1'].rootAssembly.sets['implant.implantCementInterface'],
        name='tie2',
        positionToleranceMethod=COMPUTED,
        secondary=mdb.models['Model-1'].rootAssembly.sets['cement.PART-1-1.remainedCementImplantInterface'],
        thickness=ON,
        tieRotations=ON
    )

    # define contact
    """
    mdb.models['Model-1'].ContactProperty('IntProp-1')
    mdb.models['Model-1'].interactionProperties['IntProp-1'].TangentialBehavior(
        dependencies=0,
        directionality=ISOTROPIC,
        elasticSlipStiffness=None,
        formulation=PENALTY,
        fraction=0.005,
        maximumElasticSlip=FRACTION,
        pressureDependency=OFF,
        shearStressLimit=None,
        slipRateDependency=OFF,
        table=((0.7,),),
        temperatureDependency=OFF
    )
    mdb.models['Model-1'].interactionProperties['IntProp-1'].NormalBehavior(
        allowSeparation=ON,
        constraintEnforcementMethod=DEFAULT,
        pressureOverclosure=HARD
    )
    mdb.models['Model-1'].ContactStd(createStepName='Initial', name='Int-1')
    mdb.models['Model-1'].interactions['Int-1'].includedPairs.setValuesInStep(stepName='Initial', useAllstar=ON)
    mdb.models['Model-1'].interactions['Int-1'].contactPropertyAssignments.appendInStep(
        assignments=((GLOBAL, SELF, 'IntProp-1'),),
        stepName='Initial'
    )
    """

    job1 = mdb.Job(
        atTime=None,
        contactPrint=OFF,
        description='',
        echoPrint=OFF,
        explicitPrecision=SINGLE,
        getMemoryFromAnalysis=True,
        historyPrint=OFF,
        memory=90,
        memoryUnits=PERCENTAGE,
        model='Model-1',
        modelPrint=OFF,
        multiprocessingMode=DEFAULT,
        name='inpForATSA',
        nodalOutputPrecision=SINGLE,
        numCpus=1,
        numGPUs=0,
        queue=None,
        resultsFormat=ODB,
        scratch='',
        type=ANALYSIS,
        userSubroutine='',
        waitHours=0,
        waitMinutes=0
    )

    job1.writeInput()


if __name__ == "__main__":
    path = str(sys.argv[-1])
    print(path)
    aTSAWithAbaqus(path)