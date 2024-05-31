import numpy as np

def findShortest3DVector(vectors):
    if vectors.shape[1] != 3:
        raise Exception("You should provide an Nx3 vector array")
        return 0
    norms = np.linalg.norm(vectors, ord=2, axis=1)
    index = np.where(norms == np.min(norms))[0][0]
    return index