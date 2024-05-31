import numpy as np

def projectVectorArrayOnVector(vectorArray, vector):
    return (vectorArray@vector.reshape(-1, 1)) * vector / np.linalg.norm(vector)**2