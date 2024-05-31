from stl import mesh
import numpy as np

def loadStl(filename, mode):
    """
    INPUT:

    filename: string representing the name fo the file
    
    mode:
    mode=1 (if omitted is automatically set to one)
    set the the output to: 
        output=[p,t,tnorm]
    where
        p=points (unique) of the model nx3 array
        t=triangles indexes of the model
        tnorm= normals of triangles 
        
    mode=2
    set the the output to:
        output=[v,tnorm]
     where
         v=  vertex of the model(not unique points) of the model nx3 array. Each
             three points we have a triangle in consecutive order.
         tnorm= normals of triangles
    """
    mesh_ = mesh.Mesh.from_file(filename)
    v = mesh_.vectors.reshape(-1, 3)
    tnorm = mesh_.normals
    if mode==1:
        t = np.arange(v.shape[0]).reshape(3, -1, order='F')
        p, i, j = np.unique(v, axis=0, return_index=True, return_inverse=True)
        t = j[t.flatten()].reshape(3, -1).T
        return p, t, tnorm      
    elif mode == 2:
        return v, tnorm