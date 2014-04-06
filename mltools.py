"""linear regression functions"""
import numpy as np
import pdb

def addx0(X):
    """add column of ones to act as x0, the intercept value"""
    return np.c_[ np.ones(len(X)), X ]

def addxsquare(X, col):
    """add squared feature"""
    return np.c_[ X, np.multiply(X[:,col],X[:,col]) ]

def cost(X, y, theta):
    """compute linear regression cost function (one variable)"""
    # J = (X*theta - y)' * (X*theta - y) / 2*m
    m = len(y)
    J = (X*theta - y).T * (X*theta - y) / (2*m)
    return J[0,0] # return single value

def descent(X, y, theta, alpha, num_iters, lreg = 0.):
    """compute gradient descent"""
    m = len(y)
    J_history = np.zeros(num_iters)
    for ii in range(num_iters):
        # matlab: theta -= alpha * X' * (X*theta - y) / m
        # linear regularization TURNED OFF
        #theta[0] = theta[0] - alpha*( X[:,0].T * (X[:,0] * theta[0] - y) )/m
        #theta[1:] = theta[1:]*(1.0 - alpha*lreg/m) - alpha*( X[:,1:].T * (X[:,1:] * theta[1:] - y) )/m
        theta = theta - alpha * ( X.T * (X*theta - y) ) / m
        J_history[ii] = cost(X, y, theta)
    return (theta, J_history)

def featureNormalize(X):
    """normalize X"""
    X_norm = X
    mu = X_norm.mean(axis = 0)
    sigma = X_norm.std(axis = 0)
    X_norm = X_norm - mu
    X_norm = X_norm/sigma
    return (X_norm, mu, sigma)

def normalEqn(X, y, lreg = 0.):
    """the normal equation, returns theta"""
    # theta = (X.T*X - lambda*regIdentity).I * X.T * y
    I = np.eye(X.shape[1])
    I[0,0] = 0.
    theta = (X.T*X - lreg*I).I * X.T * y
    return theta
