"""Machine Learning Tools: feature tools module"""

import numpy as np

def add_x0(X):
    """add x0, the bias feature"""
    return np.c_[ np.ones(len(X)), X ]

def add_quadratic(X, col):
    """add quadratic feature of column 'col' to X"""
    return np.c_[ X, np.multiply(X[:,col],X[:,col]) ]

def normalize(X):
    """normalize X, returns tuple: (X_norm, mu, sigma)"""
    X_norm = np.matrix.copy(X)
    mu = X_norm.mean(axis = 0)
    sigma = X_norm.std(axis = 0)
    X_norm = X_norm - mu
    X_norm = X_norm/sigma
    return (X_norm, mu, sigma)

def polynomial(X1, X2, degree = 6):
    """build polynomial data set from two features, bias feature included"""
    X_out = np.mat(np.ones(len(X1))).T
    for ii in range(1,degree+1):
        for jj in range(ii+1):
            X_out = np.c_[  X_out, np.multiply(np.power(X1,ii-jj), np.power(X2,jj))  ]
    return X_out
