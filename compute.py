"""linear regression functions"""
import numpy as np

def cost(X, y, theta):
    """compute linear regression cost function (one variable)"""
    # J = (X*theta - y)' * (X*theta - y) / 2*m
    m = len(y)
    # note about pandas' dot product: column names must be same as row names
    Xtheta_minus_y = X.dot(theta).sub(y.squeeze(), axis = 0)
    J = Xtheta_minus_y.T.dot(Xtheta_minus_y) / (2*m)
    return J.values[0][0] # return single value, not DF 

def descent(X, y, theta, alpha, num_iters):
    """compute gradient descent (one variable)"""
    m = len(y)
    J_history = np.zeros(num_iters)
    for ii in range(num_iters):
        # theta -= alpha * X' * (X*theta - y) / m
        Xtheta_minus_y = X.dot(theta).sub(y.squeeze(), axis = 0)
        theta = theta.sub((alpha * X.T.dot(Xtheta_minus_y) / m).squeeze(), axis = 0)
        J_history[ii] = cost(X, y, theta)
    return (theta, J_history)

def featureNormalize(X):
    """normalize X"""
    X_norm = X
    mu = X_norm.mean()
    sigma = X_norm.std()
    X_norm = X_norm - mu
    X_norm = X_norm/sigma
    X_norm = X_norm.fillna(1.0) # warning! - fix this so it only fills NaNs in first column
    return (X_norm, mu, sigma)
    

    
