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
    return J

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
    X_norm[:,0] = 1. # leave intercept column unaffected by normalization
    return (X_norm, mu, sigma)

def normalEqn(X, y):
    """the normal equation, returns theta"""
    theta = np.dot(np.dot(np.linalg.inv(X.T.dot(X)), X.T), y)
    theta = (X.T * X).I * X.T * y
    return theta
