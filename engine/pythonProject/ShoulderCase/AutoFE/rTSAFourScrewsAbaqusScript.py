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

def rTSAWithAbaqus(inputPath):

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

    centralScrew = os.path.join(inputPath, 'centralScrewMesh3D.inp')
    centralScrewModel = mdb.ModelFromInputFile(inputFileName=centralScrew, name='centralScrew')
    centralScrewRootAssembly = mdb.models['centralScrew'].rootAssembly
    centralScrewInstance = mdb.models['centralScrew'].rootAssembly.instances['PART-1-1']
    centralScrewNodes = centralScrewInstance.nodes

    superiorScrew = os.path.join(inputPath, 'superiorScrewMesh3D.inp')
    superiorScrewModel = mdb.ModelFromInputFile(inputFileName=superiorScrew, name='superiorScrew')
    superiorScrewRootAssembly = mdb.models['superiorScrew'].rootAssembly
    superiorScrewInstance = mdb.models['superiorScrew'].rootAssembly.instances['PART-1-1']
    superiorScrewNodes = superiorScrewInstance.nodes

    anteriorScrew = os.path.join(inputPath, 'anteriorScrewMesh3D.inp')
    anteriorScrewModel = mdb.ModelFromInputFile(inputFileName=anteriorScrew, name='anteriorScrew')
    anteriorScrewRootAssembly = mdb.models['anteriorScrew'].rootAssembly
    anteriorScrewInstance = mdb.models['anteriorScrew'].rootAssembly.instances['PART-1-1']
    anteriorScrewNodes = anteriorScrewInstance.nodes

    inferiorScrew = os.path.join(inputPath, 'inferiorScrewMesh3D.inp')
    inferiorScrewModel = mdb.ModelFromInputFile(inputFileName=inferiorScrew, name='inferiorScrew')
    inferiorScrewRootAssembly = mdb.models['inferiorScrew'].rootAssembly
    inferiorScrewInstance = mdb.models['inferiorScrew'].rootAssembly.instances['PART-1-1']
    inferiorScrewNodes = inferiorScrewInstance.nodes

    posteriorScrew = os.path.join(inputPath, 'posteriorScrewMesh3D.inp')
    posteriorScrewModel = mdb.ModelFromInputFile(inputFileName=posteriorScrew, name='posteriorScrew')
    posteriorScrewRootAssembly = mdb.models['posteriorScrew'].rootAssembly
    posteriorScrewInstance = mdb.models['posteriorScrew'].rootAssembly.instances['PART-1-1']
    posteriorScrewNodes = posteriorScrewInstance.nodes

    boundaryConditionsFile = os.path.join(inputPath, 'boundaryConditionsWithImplant.json')
    with open(boundaryConditionsFile) as f:
        boundaryConditionsData = json.load(f)

    implantSurfaceNode = boundaryConditionsData["implantSurfaceNodes"]
    implantBoneInterface = boundaryConditionsData["implantBoneInterface"]
    screwsBoneInterface = boundaryConditionsData["screwsBoneInterface"]
    BCBoxNode = boundaryConditionsData["BCBox"]
    referencePoint = boundaryConditionsData["referencePoint"]
    loadDirection = boundaryConditionsData["loadDir"]

    abaqusImplantSurfaceNodeList = implantNodes[implantSurfaceNode[0]:implantSurfaceNode[0] + 1]
    for idx, nodeIn in enumerate(implantSurfaceNode):
        abaqusImplantSurfaceNodeList = abaqusImplantSurfaceNodeList +\
                                       implantNodes[implantSurfaceNode[idx]:implantSurfaceNode[idx] + 1]

    abaqusBCNodeList = scapulaNodes[BCBoxNode[0]:BCBoxNode[0] + 1]
    for idx2, nodeIn2 in enumerate(BCBoxNode):
        abaqusBCNodeList = abaqusBCNodeList + scapulaNodes[BCBoxNode[idx2]:BCBoxNode[idx2] + 1]

    abaqusImplantBoneInterfaceNodeList = scapulaNodes[implantBoneInterface[0]:implantBoneInterface[0] + 1]
    for idx2, nodeIn2 in enumerate(implantBoneInterface):
        abaqusImplantBoneInterfaceNodeList = abaqusImplantBoneInterfaceNodeList +\
                                             scapulaNodes[implantBoneInterface[idx2]:implantBoneInterface[idx2] + 1]

    abaqusScrewsBoneInterfaceNodeList = scapulaNodes[screwsBoneInterface[0]:screwsBoneInterface[0] + 1]
    for idx2, nodeIn2 in enumerate(screwsBoneInterface):
        abaqusScrewsBoneInterfaceNodeList = abaqusScrewsBoneInterfaceNodeList + \
                                             scapulaNodes[screwsBoneInterface[idx2]:screwsBoneInterface[idx2] + 1]

    elementCenters = np.zeros((len(scapulaElements), 3))
    for elementIndex, element in enumerate(scapulaElements):
        elementNodes = element.connectivity
        elementNodesCoordinates = np.array([scapulaNodes[i].coordinates for i in elementNodes])
        elementCenters[elementIndex] = calculateCenter(elementNodesCoordinates)

    threshold = 2  # mm
    abaqusImplantBoneInterfaceNodeCoordinates = np.array([node.coordinates for node in abaqusImplantBoneInterfaceNodeList])
    abaqusImplantBoneInterfaceElementIndices = findPointsWithinDistance(
        elementCenters,
        abaqusImplantBoneInterfaceNodeCoordinates,
        threshold
    )
    abaqusImplantBoneInterfaceElementList = scapulaElements[abaqusImplantBoneInterfaceElementIndices[0]:\
                                                            abaqusImplantBoneInterfaceElementIndices[0] + 1]
    for idx, elementIn in enumerate(abaqusImplantBoneInterfaceElementIndices):
        abaqusImplantBoneInterfaceElementList = abaqusImplantBoneInterfaceElementList + \
                                                scapulaElements[abaqusImplantBoneInterfaceElementIndices[idx]:\
                                                                abaqusImplantBoneInterfaceElementIndices[idx] + 1]

    threshold = 2  # mm
    abaqusScrewsBoneInterfaceNodeCoordinates = np.array(
        [node.coordinates for node in abaqusScrewsBoneInterfaceNodeList])
    abaqusScrewsBoneInterfaceElementIndices = findPointsWithinDistance(
        elementCenters,
        abaqusScrewsBoneInterfaceNodeCoordinates,
        threshold
    )
    abaqusScrewsBoneInterfaceElementList = scapulaElements[abaqusScrewsBoneInterfaceElementIndices[0]: \
                                                            abaqusScrewsBoneInterfaceElementIndices[0] + 1]
    for idx, elementIn in enumerate(abaqusScrewsBoneInterfaceElementIndices):
        abaqusScrewsBoneInterfaceElementList = abaqusScrewsBoneInterfaceElementList + \
                                                scapulaElements[abaqusScrewsBoneInterfaceElementIndices[idx]: \
                                                                abaqusScrewsBoneInterfaceElementIndices[idx] + 1]

    # abaqus set for BC
    scapulaRootAssembly.Set(name='BC', nodes=abaqusBCNodeList)

    # abaqus set for implantBoneInterface
    scapulaRootAssembly.Set(name='boneImplantInterfaceNodes', nodes=abaqusImplantBoneInterfaceNodeList)

    # abaqus set for bone-implant interface elements
    scapulaRootAssembly.Set(name='boneImplantInterfaceElements', elements=abaqusImplantBoneInterfaceElementList)
    scapulaRootAssembly.Set(name='boneScrewsInterfaceElements', elements=abaqusScrewsBoneInterfaceElementList)

    # abaqus set for RF
    implantRootAssembly.ReferencePoint(point=(referencePoint[0], referencePoint[1], referencePoint[2]))
    try:
        implantRootAssembly.Set(name='referencePoint', referencePoints=(implantRootAssembly.referencePoints[5],))
    except:
        implantRootAssembly.Set(name='referencePoint', referencePoints=(implantRootAssembly.referencePoints[4],))
    implantRootAssembly.Set(name='MPCSlave', nodes=abaqusImplantSurfaceNodeList)

    # assembly model
    assemblyModel = mdb.models['Model-1'].rootAssembly
    assemblyModel.Instance(name='scapula', model=scapulaModel)
    assemblyModel.Instance(name='implant', model=implantModel)
    assemblyModel.Instance(name='centralScrew', model=centralScrewModel)
    assemblyModel.Instance(name='superiorScrew', model=superiorScrewModel)
    assemblyModel.Instance(name='anteriorScrew', model=anteriorScrewModel)
    assemblyModel.Instance(name='inferiorScrew', model=inferiorScrewModel)
    assemblyModel.Instance(name='posteriorScrew', model=posteriorScrewModel)
    scapulaInstance = assemblyModel.instances['scapula.PART-1-1']
    implantInstance = assemblyModel.instances['implant.PART-1-1']
    centralScrewInstance = assemblyModel.instances['centralScrew.PART-1-1']
    superiorScrewInstance = assemblyModel.instances['superiorScrew.PART-1-1']
    anteriorScrewInstance = assemblyModel.instances['anteriorScrew.PART-1-1']
    inferiorScrewInstance = assemblyModel.instances['inferiorScrew.PART-1-1']
    posteriorScrewInstance = assemblyModel.instances['posteriorScrew.PART-1-1']

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

    # Load based on the litreature the magnitude of the load is going to be 0.9 of
    # average body weight
    #g = 9.81
    #F_coeff = 0.9
    #loadMag = g*F_coeff*patientWeight
    #loadMag = g * F_coeff * 90
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
    mdb.models['implant'].materials['imp'].Elastic(table=((110000.0, 0.3),)) # material properties of the Tornier implant https://onlinelibrary.wiley.com/doi/full/10.1002/jor.23115
    mdb.models['implant'].HomogeneousSolidSection(material='imp', name='Section-1',thickness=None)
    mdb.models['implant'].parts['PART-1'].SectionAssignment(
        offset=0.0,
        offsetField='',
        offsetType=MIDDLE_SURFACE,
        region=Region(elements=mdb.models['implant'].parts['PART-1'].elements),
        sectionName='Section-1',
        thicknessAssignment=FROM_SECTION
    )

    # screw material
    mdb.models['centralScrew'].Material(name='CS')
    mdb.models['centralScrew'].materials['CS'].Elastic(table=((110000.0,0.3),))
    mdb.models['centralScrew'].HomogeneousSolidSection(material='CS', name='Section-1', thickness=None)
    mdb.models['centralScrew'].parts['PART-1'].SectionAssignment(
        offset=0.0,
        offsetField='',
        offsetType=MIDDLE_SURFACE,
        region=Region(elements=mdb.models['centralScrew'].parts['PART-1'].elements),
        sectionName='Section-1',
        thicknessAssignment=FROM_SECTION
    )

    mdb.models['superiorScrew'].Material(name='SS')
    mdb.models['superiorScrew'].materials['SS'].Elastic(table=((110000.0, 0.3),))
    mdb.models['superiorScrew'].HomogeneousSolidSection(material='SS', name='Section-1', thickness=None)
    mdb.models['superiorScrew'].parts['PART-1'].SectionAssignment(
        offset=0.0,
        offsetField='',
        offsetType=MIDDLE_SURFACE,
        region=Region(elements=mdb.models['superiorScrew'].parts['PART-1'].elements),
        sectionName='Section-1',
        thicknessAssignment=FROM_SECTION
    )

    mdb.models['anteriorScrew'].Material(name='AS')
    mdb.models['anteriorScrew'].materials['AS'].Elastic(table=((110000.0, 0.3),))
    mdb.models['anteriorScrew'].HomogeneousSolidSection(material='AS', name='Section-1', thickness=None)
    mdb.models['anteriorScrew'].parts['PART-1'].SectionAssignment(
        offset=0.0,
        offsetField='',
        offsetType=MIDDLE_SURFACE,
        region=Region(elements=mdb.models['anteriorScrew'].parts['PART-1'].elements),
        sectionName='Section-1',
        thicknessAssignment=FROM_SECTION
    )

    mdb.models['inferiorScrew'].Material(name='IS')
    mdb.models['inferiorScrew'].materials['IS'].Elastic(table=((110000.0, 0.3),))
    mdb.models['inferiorScrew'].HomogeneousSolidSection(material='IS', name='Section-1', thickness=None)
    mdb.models['inferiorScrew'].parts['PART-1'].SectionAssignment(
        offset=0.0,
        offsetField='',
        offsetType=MIDDLE_SURFACE,
        region=Region(elements=mdb.models['inferiorScrew'].parts['PART-1'].elements),
        sectionName='Section-1',
        thicknessAssignment=FROM_SECTION
    )

    mdb.models['posteriorScrew'].Material(name='PS')
    mdb.models['posteriorScrew'].materials['PS'].Elastic(table=((110000.0, 0.3),))
    mdb.models['posteriorScrew'].HomogeneousSolidSection(material='PS', name='Section-1', thickness=None)
    mdb.models['posteriorScrew'].parts['PART-1'].SectionAssignment(
        offset=0.0,
        offsetField='',
        offsetType=MIDDLE_SURFACE,
        region=Region(elements=mdb.models['posteriorScrew'].parts['PART-1'].elements),
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
    mdb.models['centralScrew'].parts['PART-1'].Set(
        name='centralScrewNodes',
        nodes=mdb.models['centralScrew'].parts['PART-1'].nodes,
    )
    mdb.models['anteriorScrew'].parts['PART-1'].Set(
        name='anteriorScrewNodes',
        nodes=mdb.models['anteriorScrew'].parts['PART-1'].nodes,
    )
    mdb.models['inferiorScrew'].parts['PART-1'].Set(
        name='inferiorScrewNodes',
        nodes=mdb.models['inferiorScrew'].parts['PART-1'].nodes,
    )
    mdb.models['posteriorScrew'].parts['PART-1'].Set(
        name='posteriorScrewNodes',
        nodes=mdb.models['posteriorScrew'].parts['PART-1'].nodes,
    )
    mdb.models['superiorScrew'].parts['PART-1'].Set(
        name='superiorScrewNodes',
        nodes=mdb.models['superiorScrew'].parts['PART-1'].nodes,
    )

    mdb.models['Model-1'].Temperature(
        createStepName='Initial',
        distributionType=DISCRETE_FIELD,
        field='DiscField-1',
        magnitudes=(1.0,),
        name='Predefined Field-1',
        region=mdb.models['Model-1'].rootAssembly.sets['scapula.PART-1-1.scapulaNodes']
    )

    #define tie
    mdb.models['Model-1'].Tie(
        adjust=OFF,
        main=mdb.models['Model-1'].rootAssembly.instances['scapula.PART-1-1'].sets['scapulaNodes'],
        name='tie1',
        positionToleranceMethod=COMPUTED,
        secondary=mdb.models['Model-1'].rootAssembly.instances['implant.PART-1-1'].sets['implantNodes'],
        thickness=ON,
        tieRotations=ON
    )

    mdb.models['Model-1'].Tie(
        adjust=OFF,
        main=mdb.models['Model-1'].rootAssembly.instances['scapula.PART-1-1'].sets['scapulaNodes'],
        name='tie2',
        positionToleranceMethod=COMPUTED,
        secondary=mdb.models['Model-1'].rootAssembly.instances['centralScrew.PART-1-1'].sets['centralScrewNodes'],
        thickness=ON,
        tieRotations=ON
    )

    mdb.models['Model-1'].Tie(
        adjust=OFF,
        main=mdb.models['Model-1'].rootAssembly.instances['scapula.PART-1-1'].sets['scapulaNodes'],
        name='tie3',
        positionToleranceMethod=COMPUTED,
        secondary=mdb.models['Model-1'].rootAssembly.instances['superiorScrew.PART-1-1'].sets['superiorScrewNodes'],
        thickness=ON,
        tieRotations=ON
    )

    mdb.models['Model-1'].Tie(
        adjust=OFF,
        main=mdb.models['Model-1'].rootAssembly.instances['scapula.PART-1-1'].sets['scapulaNodes'],
        name='tie4',
        positionToleranceMethod=COMPUTED,
        secondary=mdb.models['Model-1'].rootAssembly.instances['anteriorScrew.PART-1-1'].sets['anteriorScrewNodes'],
        thickness=ON,
        tieRotations=ON
    )

    mdb.models['Model-1'].Tie(
        adjust=OFF,
        main=mdb.models['Model-1'].rootAssembly.instances['scapula.PART-1-1'].sets['scapulaNodes'],
        name='tie5',
        positionToleranceMethod=COMPUTED,
        secondary=mdb.models['Model-1'].rootAssembly.instances['inferiorScrew.PART-1-1'].sets['inferiorScrewNodes'],
        thickness=ON,
        tieRotations=ON
    )

    mdb.models['Model-1'].Tie(
        adjust=OFF,
        main=mdb.models['Model-1'].rootAssembly.instances['scapula.PART-1-1'].sets['scapulaNodes'],
        name='tie6',
        positionToleranceMethod=COMPUTED,
        secondary=mdb.models['Model-1'].rootAssembly.instances['posteriorScrew.PART-1-1'].sets['posteriorScrewNodes'],
        thickness=ON,
        tieRotations=ON
    )


    #define contact
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
        name='inpForRTSA',
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
    rTSAWithAbaqus(path)