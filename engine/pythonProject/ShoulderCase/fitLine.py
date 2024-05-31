from sklearn.decomposition import PCA
import numpy as np
from scipy.spatial.distance import cdist

def fitLine(X):
    """
    Fits a line to a group of points in X
    """
    Xn, Xm = X.shape
    pca = PCA(n_components=Xm).fit(X)
    coeff = pca.components_.T[:, 0]
    score = pca.transform(X)[:, 0]
    dirVect = coeff
    meanX = np.mean(X, axis=0)

    Xfit1 = meanX + np.dot(score.reshape(-1, 1), coeff.reshape(1, -1))
    residuals = X-Xfit1
    error = np.diag(cdist(residuals, np.zeros((Xn,Xm))))
    sse = np.sum(error**2)
    rmse = np.linalg.norm(error)/np.sqrt(Xn)

    
    tot = []
    for i in range(Xn):
        tot.append(np.linalg.norm(meanX-X[i, :]))
    tot = np.array(tot)
    
    sst = np.sum(tot**2)
    
    R2 = 1 - (sse/sst)
    return dirVect, meanX, residuals, rmse,R2 
