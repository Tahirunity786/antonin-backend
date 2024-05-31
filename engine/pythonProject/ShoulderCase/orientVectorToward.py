import numpy as np

def orientVectorToward(vector, orientation):
    if np.dot(vector, orientation) < 0:
        vector = -vector
    return vector
    