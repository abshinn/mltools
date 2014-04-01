"""linear regression functions"""
import numpy as np
import pdb

def addx0(X):
    """add column of ones to act as x0, the intercept value"""
    return np.c_[ np.ones(len(X)), X ]

def cost(X, y, theta):
    """compute linear regression cost function (one variable)"""
    # J = (X*theta - y)' * (X*theta - y) / 2*m
    m = len(y)
    Xtheta_minus_y = X * theta - y
    J = Xtheta_minus_y.T * Xtheta_minus_y / (2*m)
    return J[0,0] # return single value

def descent(X, y, theta, alpha, num_iters):
    """compute gradient descent (one variable)"""
    m = len(y)
    J_history = np.zeros(num_iters)
    for ii in range(num_iters):
        # theta -= alpha * X' * (X*theta - y) / m
        Xtheta_minus_y = X * theta - y
        theta = theta - (alpha * X.T * Xtheta_minus_y / m)
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
