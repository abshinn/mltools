"""linear regression functions"""

import numpy as np
import pdb

def add_x0(X):
    """add x0, the bias feature"""
    return np.c_[ np.ones(len(X)), X ]

def add_quadratic(X, col):
    """add quadratic feature of column 'col' to X"""
    return np.c_[ X, np.multiply(X[:,col],X[:,col]) ]

def cost(X, y, theta, rlambda = 0.):
    """compute linear regression cost function (one variable)"""
    m = len(y)
    #J = ( (X*theta - y).T * (X*theta - y) ) / (2*m)
    J = ( (X*theta - y).T * (X*theta - y) + rlambda*(theta[1:,:].T * theta[1:,:]) ) / (2.*m)
    return J[0,0] # return single value

def descent(X, y, theta, alpha, num_iters, rlambda = 0.):
    """compute gradient descent"""
    m = len(y)
    J_history = np.zeros(num_iters)
    grad = np.matrix.copy(theta) # initial theta
    for ii in range(num_iters):
        # linear regularization TURNED OFF
        #grad = grad - alpha * ( X.T * (X*grad - y) ) / m
        # linear regularization TURNED ON
        grad_temp = np.matrix.copy(grad)
        grad_temp[0,0] = 0.
        grad = grad - alpha * ( X.T * (X*grad - y) - rlambda*grad_temp) / m
        J_history[ii] = cost(X, y, grad, rlambda = rlambda)
    return (grad, J_history)

def featureNormalize(X):
    """normalize X"""
    X_norm = np.matrix.copy(X)
    mu = X_norm.mean(axis = 0)
    sigma = X_norm.std(axis = 0)
    X_norm = X_norm - mu
    X_norm = X_norm/sigma
    return (X_norm, mu, sigma)

def normalEqn(X, y, rlambda = 0.):
    """the normal equation, returns theta"""
    I = np.eye(X.shape[1])
    I[0,0] = 0.
    theta = (X.T*X - rlambda*I).I * X.T * y
    return theta

def polyfeatures(X1, X2, degree = 6):
    """build a polynomial data set from two features"""
    X_out = np.mat(np.ones(len(X1))).T
    for ii in range(1,degree+1):
        for jj in range(ii+1):
            X_out = np.c_[  X_out, np.multiply( np.power(X1,ii-jj), np.power(X2,jj) )  ]
    return X_out

def sigmoid(z):
    """sigmoid function"""
    return 1./(1. + np.exp(-z))

def LRcost(theta, X, y, rlambda = 0.):
    """logistic regression cost function"""
    m, n = X.shape
    theta = np.copy(theta.reshape(n, 1))
    theta = np.matrix(theta)
    I = np.eye(n)
    I[0,0] = 0.
    H = sigmoid(X*theta) # hypothesis
    cost = (-y.T * np.log(H) - (1 - y).T * np.log(1 - H) + rlambda*(theta[1:,:].T * theta[1:,:])/2.) / m
    grad = (X.T * (H - y) + rlambda*I*theta) / m
    return cost.A.flatten(), grad.A.flatten()
