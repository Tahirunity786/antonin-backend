import numpy as np
from sklearn.decomposition import PCA

class Plane:
    """
     A plane is decribed by a point and a normal.
     Plane can define the desired plane's normal and point
     by fitting an array of points.
     Plane has a method to project points on the current plane.
    """
    def __init__(self):
        self.normal = []
        self.point = []
        self.fitPerformance = {"points":[],
                               "residuals":[],
                               "RMSE":[],
                               "R2":[]}
    def fit(self, points):
        # Fit current plane to given points
        pca = PCA(n_components=points.shape[1]).fit(points)
        normal = np.cross(pca.components_.T[:, 0], pca.components_.T[:, 1])
        normal = -normal/np.linalg.norm(normal)

        meanPoint = np.mean(points, axis=0)

        scores = pca.transform(points)
        scores = np.hstack((-scores[:,0].reshape(-1, 1), scores[:, 1].reshape(-1, 1)))

        coeff = pca.components_
        coeff = np.vstack((-coeff[0, :], coeff[1, :]))
        estimatedPoints = meanPoint + scores@coeff
        residuals = points - estimatedPoints
        error = np.linalg.norm(residuals, ord=2, axis=1)
        rmse = np.linalg.norm(error)/np.sqrt(points.shape[0])
        sumSquareError = np.sum(error**2)
        total = np.linalg.norm(points - meanPoint, ord=2, axis=1)
        sumSquareTotal = np.sum(total**2)
        R2 = 1-(sumSquareError/sumSquareTotal)
        self.normal = normal
        self.point = meanPoint
        self.setFitPerformance(points, residuals, rmse, R2)

    def setFitPerformance(self,points,residuals,RMSE,R2):
        self.fitPerformance["points"] = points
        self.fitPerformance["residuals"] = residuals
        self.fitPerformance["RMSE"] = RMSE
        self.fitPerformance["R2"] = R2

    def setPointOnPlane(self, point):
        self.point = point

    def projectOnPlane(self, points):
        N2 = self.normal.reshape(-1, 1)*self.normal.reshape(1, -1)
        return points@(np.eye(3)-N2) + self.point.reshape(1, -1)@N2