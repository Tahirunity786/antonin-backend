import numpy as np
from utils.Plane.Plane import Plane
import matplotlib.pyplot as plt
import plotly.graph_objects as go

class CoordinateSystem:
    """
    Useful to define, use, and test a coordinate system.

    Can be tested on orthogonality, right or left handedness.

    Can be used to express global points in local coordinates
    and vice versa.
    Can be used to project points on planes spanned by the
    CoordinateSystem axes.
    """
    def __init__(self):
        self.origin = []
        self._xAxis = []
        self._yAxis = []
        self._zAxis = []

        self.rotationMatrix = np.array([])

    def isEmpty(self):
        return len(self.origin) == 0 or \
               len(self._xAxis) == 0 or \
               len(self._yAxis) == 0 or \
               len(self._zAxis) == 0

    def set_xAxis(self, value):
        self._xAxis = value
        self.rotationMatrix = np.zeros((value.shape[0], 3))
        self.rotationMatrix[:, 0] = value/np.linalg.norm(value)

    def set_yAxis(self, value):
        self._yAxis = value
        self.rotationMatrix[:, 1] = value/np.linalg.norm(value)

    def set_zAxis(self, value):
        self._zAxis = value
        self.rotationMatrix[:, 2] = value/np.linalg.norm(value)

    def getRotationMatrix(self):
        assert self.isOrthogonal(), "CoordinateSystem is not orthogonal. Can't get its rotation matrix"
        return self.rotationMatrix

    def express(self, globalPoints):
        # Give the coordinates of global cartesian points in present coordinate system
        assert not self.isEmpty(), "Coordinate System has not been initialized yet."
        assert globalPoints.shape[1] == 3, "Points have to be a (N,3) numpy array."
        return (globalPoints-self.origin.reshape(1, -1)) @ self.rotationMatrix

    def getGlobalCoordinatesOfLocalPoints(self, localPoints):
        # Give the coordinates of local points in global cartesian coordinate system
        assert not self.isEmpty(), "Coordinate System has not been initialized yet."
        assert localPoints.shape[1] == 3, "Points have to be a (N,3) numpy array."
        return (localPoints @ self.rotationMatrix.T) + self.origin

    def projectOnXYPlane(self, points):
        assert not self.isEmpty(), "Coordinate System has not been initialized yet."
        projectionPlane = Plane()
        projectionPlane.fit(np.concatenate([self.origin.reshape(1, -1),
                        self.origin.reshape(1, -1)+self._xAxis.reshape(1, -1),
                        self.origin.reshape(1, -1)+self._yAxis.reshape(1, -1)], axis=0))
        return projectionPlane.projectOnPlane(points)

    def projectOnYZPlane(self, points):
        assert not self.isEmpty(), "Coordinate System has not been initialized yet."
        projectionPlane = Plane()
        projectionPlane.fit(np.concatenate([self.origin.reshape(1, -1),
                        self.origin.reshape(1, -1) + self._yAxis.reshape(1, -1),
                        self.origin.reshape(1, -1) + self._zAxis.reshape(1, -1)], axis=0))
        return projectionPlane.projectOnPlane(points)

    def projectOnZXPlane(self, points):
        assert not self.isEmpty(), "Coordinate System has not been initialized yet."
        projectionPlane = Plane()
        projectionPlane.fit(np.concatenate([self.origin.reshape(1, -1),
                        self.origin.reshape(1, -1) + self._zAxis.reshape(1, -1),
                        self.origin.reshape(1, -1) + self._xAxis.reshape(1, -1)], axis=0))
        return projectionPlane.projectOnPlane(points)

    def isOrthogonal(self):
        """
        The axes comparison can't be done looking for strict equality due to
        rounded values. Therefore the values must be evaluated with a given
        tolerance
        """
        assert not self.isEmpty(), "Coordinate System has not been initialized yet."
        minAngle = (np.pi/2)-(10**(-6)) # Arbitrarily defined as a deviation of
                                        # a millionth of a radian
        if np.dot(self._xAxis, self._yAxis) > \
            np.linalg.norm(self._xAxis)*np.linalg.norm(self._yAxis)*np.cos(minAngle):
                return False
        if np.dot(self._xAxis, self._zAxis) > \
            np.linalg.norm(self._xAxis)*np.linalg.norm(self._zAxis)*np.cos(minAngle):
                return False
        if np.dot(self._zAxis, self._yAxis) > \
            np.linalg.norm(self._zAxis)*np.linalg.norm(self._yAxis)*np.cos(minAngle):
                return False
        return True

    def isRightHanded(self):
        """
        The comparison between a theoretical right-handed axis and the actual
        z Axis can't be done looking for strict vector equality due to rounded
        values. Therefore the norm of the sum of the two vectors is evaluated.
        """
        assert not self.isEmpty(), "Coordinate System has not been initialized yet."
        if not self.isOrthogonal():
            return False
        rightHandedAxis = np.cross(self._xAxis, self._yAxis)
        if np.linalg.norm(rightHandedAxis + self._zAxis) < np.linalg.norm(rightHandedAxis):
            return False
        return True

    def isLeftHanded(self):
        assert not self.isEmpty(), "Coordinate System has not been initialized yet."
        if not self.isOrthogonal():
            return False
        if self.isRightHanded():
            return False
        return True

    def plot(self, center):
        axisLength = 50
        if center:
            origin = np.arryay([0, 0, 0])
        else:
            origin = self.origin

        X = np.concatenate([origin.reshape(1, -1), (origin + self._xAxis*axisLength).reshape(1, -1)], axis=0)
        Y = np.concatenate([origin.reshape(1, -1), (origin + self._yAxis*axisLength).reshape(1, -1)], axis=0)
        Z = np.concatenate([origin.reshape(1, -1), (origin + self._zAxis*axisLength).reshape(1, -1)], axis=0)

        fx = go.Figure(data=[go.Scatter3d(x=X[:, 0], y=X[:, 1], z=X[:, 2],
                                          mode='lines',
                                          marker=dict(color="red"),
                                          name="PA")])

        fy = go.Figure(data=[go.Scatter3d(x=Y[:, 0], y=Y[:, 1], z=Y[:, 2],
                                          mode='lines',
                                          marker=dict(color="green"),
                                          name="IS")])

        fz = go.Figure(data=[go.Scatter3d(x=Z[:, 0], y=Z[:, 1], z=Z[:, 2],
                                          mode='lines',
                                          marker=dict(color="blue"),
                                          name="ML")])
        fx = fx.update_traces(line_width=5)
        fy = fy.update_traces(line_width=5)
        fz = fz.update_traces(line_width=5)

        fx.add_trace(go.Cone(
            x = [X[1, 0]],
            y = [X[1, 1]],
            z = [X[1, 2]],
            u = [0.3*(X[1, 0] - X[0, 0])],
            v = [0.3*(X[1, 1] - X[0, 1])],
            w = [0.3*(X[1, 2] - X[0, 2])],
            showlegend=False,
            showscale=False,
            colorscale=[[0, 'red'], [1, 'red']]
        ))

        fy.add_trace(go.Cone(
            x=[Y[1, 0]],
            y=[Y[1, 1]],
            z=[Y[1, 2]],
            u=[0.3 * (Y[1, 0] - Y[0, 0])],
            v=[0.3 * (Y[1, 1] - Y[0, 1])],
            w=[0.3 * (Y[1, 2] - Y[0, 2])],
            showlegend=False,
            showscale=False,
            colorscale=[[0, 'green'], [1, 'green']]
        ))

        fz.add_trace(go.Cone(
            x=[Z[1, 0]],
            y=[Z[1, 1]],
            z=[Z[1, 2]],
            u=[0.3 * (Z[1, 0] - Z[0, 0])],
            v=[0.3 * (Z[1, 1] - Z[0, 1])],
            w=[0.3 * (Z[1, 2] - Z[0, 2])],
            showlegend=False,
            showscale=False,
            colorscale=[[0, 'blue'], [1, 'blue']]
        ))


        fig = go.Figure(data=fx.data + fy.data + fz.data)


        return fig

