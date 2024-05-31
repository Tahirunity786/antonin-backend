import numpy as np

def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    if any(v): #if not all zeros then 
        c = np.dot(a, b)
        s = np.linalg.norm(v)
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))

    else:
        return np.eye(3) #cross of all zeros only occurs on identical directions   

    
def angle_of_rotation(rotation_matrix):
    """
    Convert rotation matrix to axis of rotation and calculate the angle of rotation
    """
    axis_of_rotation = np.array([rotation_matrix[2,1]-rotation_matrix[1,2],
                                 rotation_matrix[0,2]-rotation_matrix[2,0],
                                 rotation_matrix[1,0]-rotation_matrix[0,1]])
    rotation_angle = np.arcsin(np.linalg.norm(axis_of_rotation)/2)*180/np.pi
    return rotation_angle

def angle_of_rotation_from_vectors(vec1, vec2):
    """
    Convert rotation matrix to axis of rotation and calculate the angle of rotation
    """
    vec1_n = vec1 / np.linalg.norm(vec1)
    vec2_n = vec2 / np.linalg.norm(vec2)
    vec1_norm = np.linalg.norm(vec1)
    vec2_norm = np.linalg.norm(vec2)
    axis_of_rotation = np.cross(vec1_n, vec2_n) / np.linalg.norm(np.cross(vec1_n, vec2_n))
    rotation_angle = np.arccos(np.dot(vec1_n, vec2_n))*180/np.pi
    return rotation_angle


def axis_of_rotation_from_vectors(vec1, vec2):
    """
    Convert rotation matrix to axis of rotation and calculate the angle of rotation
    """
    vec1_n = vec1 / np.linalg.norm(vec1)
    vec2_n = vec2 / np.linalg.norm(vec2)
    vec1_norm = np.linalg.norm(vec1)
    vec2_norm = np.linalg.norm(vec2)

    if any(np.cross(vec1_n, vec2_n)):
        axis_of_rotation = np.cross(vec1_n, vec2_n) / np.linalg.norm(np.cross(vec1_n, vec2_n))
        return axis_of_rotation
    else:
        vec1_n_abs = np.abs(vec1_n)
        minIndex = np.argmin(vec1_n_abs)
        res = np.zeros(3)
        res[minIndex] = 1
        return np.cross(vec1_n, res) / np.linalg.norm(np.cross(vec1_n, res))

def axis_of_rotation(rotation_matrix):
    """
    Convert rotation matrix to axis of rotation and calculate the angle of rotation
    """
    axis_of_rotation = np.array([rotation_matrix[2,1]-rotation_matrix[1,2],
                                 rotation_matrix[0,2]-rotation_matrix[2,0],
                                 rotation_matrix[1,0]-rotation_matrix[0,1]])
    return axis_of_rotation