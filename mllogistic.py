"""Machine Learning Tools: logistic regression tools module"""

import numpy as np

def sigmoid(z):
    """sigmoid function"""
    return 1./(1. + np.exp(-z))

def cost(theta, X, y, rlambda = 0.):
    """logistic regression cost function"""
    m, n = X.shape
    theta = np.copy(theta.reshape(n, 1))
    theta = np.matrix(theta)
    H = sigmoid(X*theta) # hypothesis
    cost = (-y.T * np.log(H) - (1 - y).T * np.log(1 - H) + rlambda*(theta[1:,:].T * theta[1:,:])/2.) / m
    return cost.A.flatten()

def grad(theta, X, y, rlambda = 0.):
    """logistic regression gradient"""
    m, n = X.shape
    theta = np.copy(theta.reshape(n, 1))
    theta = np.matrix(theta)
    I = np.eye(n)
    I[0,0] = 0.
    H = sigmoid(X*theta) # hypothesis
    grad = (X.T * (H - y) + rlambda*I*theta) / m
    return grad.A.flatten()
