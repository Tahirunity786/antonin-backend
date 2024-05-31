import numpy as np
import matplotlib.pyplot as plt
from utils.Rotations import rotation_angle


class Vector:
    def __init__(self, *args):
        if len(args) == 1:
            self.origin = np.array([[0,0,0]])
            self.target = args[0]
        elif len(args) == 2:
            self.origin = args[0]
            self.target = args[1]
        else:
            raise Exception("Vector constructor only accepts up to 2 arguments")

    def __neg__(self):
        return Vector(self.target, self.origin)

    def __add__(self, addedVector):
        if isinstance(addedVector, Vector):
            return Vector(self.origin, self.target + addedVector.vector())
        elif addedVector.shape == (1, 3):
            return Vector(self.origin, self.target + addedVector)

    def __sub__(self, substractedVector):
        if isinstance(substractedVector, Vector):
            return self + -substractedVector
        elif substractedVector.shape == (1, 3):
            return Vector(self.origin, self.target - substractedVector)

    def __mul__(self, coeff):
        return Vector(self.origin, self.origin + coeff * self.vector())

    def __truediv__(self, coeff):
        return Vector(self.origin, self.origin + self.vector() / coeff)
            
    def copy(self):
        return Vector(self.origin, self.target)
    
    def set_origin(self, value):
        assert value.shape == (1, 3), "Origin must be a 1x3 numpy array"
        self.origin = value
        
    def set_target(self, value):
        assert value.shape == (1, 3), "Target must be a 1x3 numpy array"
        self.target = value   
        
    def points(self):
        return np.concatenate([self.origin, self.target], axis=0)
    
    def vector(self):
        return self.target - self.origin
    
    def norm(self):
        return np.linalg.norm(self.vector())
    
    def direction(self):
        return self.vector() / self.norm()
    
    def orientToward(self, orientationElement):
        if isinstance(orientationElement, Vector):
            orientation = orientationElement.vector()
        else: 
            orientation = orientationElement
        if np.dot(self.vector(), orientation) >= 0:
            return self.copy()
        else:
            return -self.copy()
     
    def project(self, projectedElement):
        """
        projectedElement must be either a point in 3D space (1x3 numpy array)
        or another Vector.
        """
        if isinstance(projectedElement, Vector):
            projectedVector = projectedElement
        elif projectedElement.shape == (1, 3):
            projectedVector = Vector(self.origin, projectedElement)
        return Vector(self.origin,
                      self.origin+self.direction().ravel()*np.dot(projectedVector.vector(),self.direction().ravel()))
    
    def orthogonalComplementTo(self, elementToReach):
        if isinstance(elementToReach, Vector):
            elementToReach = Vector(self.origin,
                                    self.origin + elementToReach.vector())
        else:
            elementToReach = Vector(self.origin, elementToReach)
        return elementToReach - self.project(elementToReach)  
    
    def plot(self, *args):
        points = self.points
        ax = plt.axes(projection='3d')
        ax.plot3D(points[:,0], points[:,1], points[:,2], *args)
    
    def dot(self, inputVector):
        if isinstance(inputVector, Vector):
            array = inputVector.vector()
        else:
            array = inputVector
        return np.dot(self.vector(), array.ravel())
    
    def cross(self, inputVector):
        if isinstance(inputVector, Vector):
            array = inputVector.vector()
        else:
            array = inputVector
        return Vector(self.origin, self.origin + np.cross(self.vector(), array))
    
    def angle(self, inputVector):
        if isinstance(inputVector, Vector):
            array = inputVector.vector()
        else:
            array = inputVector
        #rotation_matrix = rotation_angle.rotation_matrix_from_vectors(array, self.vector())
        return rotation_angle.angle_of_rotation_from_vectors(array, self.vector())
        #return np.abs(RRR.from_matrix(rotation_matrix).as_euler('xyz', degrees=True)[2])
    
    def normalised(self):
        return self.copy() / self.norm()
