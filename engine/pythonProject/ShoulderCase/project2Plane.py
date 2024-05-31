import numpy as np

def project2Plane(P, N, Q):
    """
    Let P be the m x 3 array of the 3D points to be projected, let Q be the
    1 x 3 vector of the given point on the plane, let N be the 1 x 3 vector 
    of the normal direction to the plane, and let P0 be the m x 3 array of 
    points orthogonally projected from P onto the plane. Then do this:
    """
    N = N / np.linalg.norm(N)
    N2 = N.reshape(-1, 1) @ N.reshape(1, -1)
    P0 = P @ (np.eye(3)-N2) + Q @ N2
    return P0