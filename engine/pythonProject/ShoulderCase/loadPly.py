from plyfile import PlyData
import numpy as np

def loadPly(filename):
    """
    LOADPLY Load the ply files and output verticies and faces
    """
    plydata = PlyData.read(filename)
    faces = []
    for face in plydata["face"].data:
        faces.append(face[0])
    faces = np.array(faces)
    #vertices = []
    #for vertex in plydata["vertex"].data:
    #    vertices.append(vertex[0])
    x = plydata["vertex"]["x"]
    y = plydata["vertex"]["y"]
    z = plydata["vertex"]["z"]
    vertices = np.hstack((x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)))
    return vertices, faces
