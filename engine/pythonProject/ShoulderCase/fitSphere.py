import numpy as np

def fitSphere(*args):
    """
    SPHEREFIT find least squares sphere

    Fit a sphere to a set of xyz data points
    [center,radius,residuals] = fitSphere(X)
    [center,radius,residuals] = fitSphere(x,y,z);
    Input
    x,y,z Cartesian data, n x 3 matrix or three vectors (n x 1 or 1 x n)
    Output
    center: least squares sphere center coordinates, == [xc yc zc]
    radius: radius of curvature
    residuals: residuals in the radial direction

    Fit the equation of a sphere in Cartesian coordinates to a set of xyz
    data points by solving the overdetermined system of normal equations,
    ie, x^2 + y^2 + z^2 + a*x + b*y + c*z + d = 0
    The least squares sphere has radius R = sqrt((a^2+b^2+c^2)/4-d) and
    center coordinates (x,y,z) = (-a/2,-b/2,-c/2)
    """
    if not len(args) <= 3 or not len(args) >= 1:
        raise Exception("Incorrect number of arguments")
    if len(args) == 1: # n x 3 numpy array
        if args[0].shape[1] != 3:
            raise Exception("input data must have three columns")
        else:
            z = args[0][:, 2] # save columns as x,y,z vectors
            y = args[0][:, 1]
            x = args[0][:, 0]
    elif len(args) == 3: # three x,y,z vectors
        x = args[0].reshape(-1, 1)
        y = args[1].reshape(-1, 1)
        z = args[2].reshape(-1, 1)
        if not len(x) == len(y) == len(z):
            raise Exception("input vectors must be same length")
    else: # must have one or three inputs
        raise Exception("invalid input, n x 3 np.array or 3 n x 1 np.array expected")
    
    # need four or more data points
    if len(x) < 4:
        raise("must have at least four points to fit a unique sphere")
    
    # solve linear system of normal equations
    A = np.concatenate([x.reshape(-1, 1), y.reshape(-1, 1),
                        z.reshape(-1, 1), np.ones_like(x).reshape(-1, 1)], axis=1)
    b = -(x**2 + y**2 + z**2).reshape(-1, 1)
    a = np.linalg.lstsq(A, b)[0].ravel()
    
    # return center coordinates and sphere radius
    center = -a[:3]/2
    radius = np.sqrt(np.sum(center**2)-a[3])
    
    # calculate residuals
    residuals = radius - np.sqrt(np.sum((np.concatenate([x.reshape(-1, 1),
                                                         y.reshape(-1, 1),
                                                         z.reshape(-1, 1)], axis=1)-center)**2, axis=1))
    sse = np.sum(residuals**2)
    meanX = np.array([np.mean(x), np.mean(y), np.mean(z)])
    
    tot = []
    for i in range(x.shape[0]):
        X = np.array([x[i], y[i], z[i]]) 
        tot.append(np.linalg.norm(meanX-X))
    
    tot = np.array(tot)
    sst = np.sum(tot**2)
    R2 = 1 - sse/sst
    return center,radius,residuals,R2        
        
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        