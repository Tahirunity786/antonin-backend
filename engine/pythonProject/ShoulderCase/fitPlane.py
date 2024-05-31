from sklearn.decomposition import PCA
import numpy as np
from scipy.spatial.distance import cdist

def fitPlane(X):
    Xn, Xm = X.shape
    pca = PCA(n_components=Xm).fit(X)
    normal = np.outer(pca.components_.T[:, 0], pca.components_.T[:, 1])                 
    meanX = np.mean(X, axis=0)
    Xfit = np.vstack([meanX for i in range(Xn)]) + pca.transform(X)[:, :2]@pca.components_[:, :2]
    residuals = X-Xfit
    error = np.diag(cdist(residuals, np.zeros(Xn,Xm)))
    sse = np.sum(error**2)
    rmse = np.linalg.norm(error)/np.sqrt(Xn)
    
    tot = []
    for i in range(Xn):
        tot.append(np.linalg.norm(meanX-X[i, :]))
    tot = np.array(tot)
    
    sst = np.sum(tot**2)
    
    R2 = 1 - (sse/sst) 
    return normal, meanX, residuals, rmse,R2  