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
import displayGroupOdbToolset as dgo
import math
import numpy as np

def aTSASaveMetricOnBoneCementInterface(inputPath):
    myViewport = session.Viewport(name='Output model', origin=(10, 10), width=150, height=100)
    modelName = os.path.join(inputPath, "aTSA.odb")
    myOdb = session.openOdb(name=modelName)
    myViewport.setValues(displayedObject=myOdb)
    myViewport.makeCurrent()
    myViewport.maximize()
    leaf = dgo.LeafFromElementSets(elementSets=('SCAPULA#BONECEMENTINTERFACEELEMENTS',))
    dg = session.DisplayGroup(leaf=leaf, name='BIInterface')
    dg1 = session.displayGroups['BIInterface']
    myViewport.odbDisplay.setPrimaryVariable(
        variableLabel='E',
        outputPosition=INTEGRATION_POINT,
        refinement=(INVARIANT, 'Max. Principal (Abs)'),
    )
    myViewport.odbDisplay.setValues(visibleDisplayGroups=(dg1,))
    odb = session.odbs[modelName]
    session.writeFieldReport(
        fileName='EMaxPrincipalBoneImplantInterface.csv',
        append=OFF,
        sortItem='Element Label',
        odb=odb,
        step=0,
        frame=1,
        outputPosition=ELEMENT_CENTROID,
        variable=(('E', INTEGRATION_POINT, ((INVARIANT, 'Max. Principal (Abs)'),)),)
    )
    session.writeFieldReport(
        fileName='EVOLBoneImplantInterface.csv',
        append=ON,
        sortItem='Element Label',
        odb=odb,
        step=0,
        frame=1,
        outputPosition=WHOLE_ELEMENT,
        variable=(('EVOL', WHOLE_ELEMENT),)
    )

if __name__ == "__main__": #to have the possibility to call the file in function list with an argument
    path = str(sys.argv[-1])
    print(path)
    aTSASaveMetricOnBoneCementInterface(path)