"""linear regression functions"""
import numpy as np
import pdb

def addx0(X):
    """add column of ones to act as x0, the intercept value"""
    return np.c_[ np.ones(len(X)), X ]

def addxsquare(X, col):
    """add squared feature"""
    return np.c_[ X, np.multiply(X[:,col],X[:,col]) ]

def polyfeatures(X1, X2, degree = 6):
    """build a polynomial data set from two features"""
# NOT TESTED YET
    X_out = np.mat(np.ones(len(X1))).T
    for ii in range(1,degree+1):
        for jj in range(ii):
            X_out = np.c[  X_out, np.multiply( np.power(X1,ii-jj), np.power(X2,jj) )  ]
    return X_out

def cost(X, y, theta, lreg = 0.):
    """compute linear regression cost function (one variable)"""
    # J = (X*theta - y)' * (X*theta - y) / 2*m
    m = len(y)
    #J = ( (X*theta - y).T * (X*theta - y) ) / (2*m)
    J = ( (X*theta - y).T * (X*theta - y) + lreg*(theta.T * theta) ) / (2.*m)
    return J[0,0] # return single value

def descent(X, y, theta, alpha, num_iters, lreg = 0.):
    """compute gradient descent"""
    m = len(y)
    J_history = np.zeros(num_iters)
    for ii in range(num_iters):
        # matlab: theta -= alpha * X' * (X*theta - y) / m
        # linear regularization TURNED ON 
        grad = theta - alpha * ( X.T * (X*theta - y) ) / m
        temp = theta 
        temp[0] = 0.
        theta = grad + alpha*lreg*temp/m
        # linear regularization TURNED OFF
        #theta = theta - alpha * ( X.T * (X*theta - y) ) / m
        J_history[ii] = cost(X, y, theta)
        #pdb.set_trace()
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
