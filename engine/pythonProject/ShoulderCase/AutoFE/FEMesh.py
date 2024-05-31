#import pymeshlab
#import gmsh
import os
import pandas as pd
import numpy as np
from getConfig import getConfig

class FEMesh:

    def __init__(self, shoulder):
        self.shoulder = shoulder
        self.twoDMeshFileName = self.get2DMeshFilePath()
        self.threeDMeshFileName = "refined3DMesh.inp"
        self.threeDMeshFilePath = self.get3DMeshFilePath()

    def getDataPath(self):
        side = "left" if self.shoulder.side == "L" else "right"
        return os.path.join(self.shoulder.SCase.dataCTPath,
                            getConfig()["landmarkAndSurfaceFilesFolder"],
                            "shoulders",
                            side,
                            "auto",
                            "FE"
                            )


    def get2DMeshFilePath(self):
        return os.path.join(self.shoulder.SCase.dataCTPath,
                            getConfig()["landmarkAndSurfaceFilesFolder"],
                            f"scapulaSurfaceAuto{self.shoulder.side}.ply"
                            )

    def get2DRefinedMeshFilePath(self):
        return os.path.join(self.getDataPath(),
                            f"scapulaSurfaceAuto{self.shoulder.side}Refined.stl"
                            )

    def get3DMeshFilePath(self):
        if not os.path.isdir(self.getDataPath()):
            os.mkdir(self.getDataPath())
        return os.path.join(self.getDataPath(), self.threeDMeshFileName)

    def refineTriangularMesh(self):

        meshSet = pymeshlab.MeshSet()
        try:
            meshSet.load_new_mesh(self.get2DMeshFilePath())
        except:
            raise Exception("Could not find the 2D mesh")
        meshSet.meshing_merge_close_vertices()
        meshSet.meshing_remove_duplicate_faces()
        meshSet.meshing_remove_duplicate_vertices()

        meshSet.apply_filter('simplification_quadric_edge_collapse_decimation', targetfacenum=20000, preservenormal=True)

        meshSet.save_current_mesh(self.get2DRefinedMeshFilePath())

    def convert2DMeshTo3D(self):

        # ensures that gmash has not a model already in from last simulation (causes PLC error)
        if (gmsh.isInitialized()):
            gmsh.finalize()
        gmsh.initialize()

        gmsh.merge(self.get2DRefinedMeshFilePath())  # import stl file to gmsh
        gmsh.option.setNumber("General.Terminal", 1)
        gmsh.option.setNumber("Mesh.Algorithm3D", 1)

        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 1)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 10)

        gmsh.option.setNumber("Mesh.Optimize", 1)
        gmsh.option.setNumber("Mesh.QualityType", 2)

        n = gmsh.model.getDimension()
        s = gmsh.model.getEntities(n)
        l = gmsh.model.geo.addSurfaceLoop([s[i][1] for i in range(len(s))])
        gmsh.model.geo.addVolume([l])

        gmsh.model.geo.synchronize()

        gmsh.model.mesh.generate(3)

        gmsh.write(self.get3DMeshFilePath())

        gmsh.finalize()

        binDirectory = r''
        os.environ['PATH'] += os.path.pathsep + binDirectory

    def areTetrahedralElementsInMesh(self):

        with open(self.get3DMeshFilePath(), "r") as file:
            lines = file.readlines()
        for line in lines:
            if "C3D4" in line:
                return True
        raise Exception("No tetrahedral elements in mesh")

    def delete2DElements(self):

        with open(self.get3DMeshFilePath(), "r") as file:
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


        np.savetxt(self.get3DMeshFilePath(),
                   threeDMeshDataFrame.values,
                   fmt="%s",
                   delimiter="\t",
                   newline="")