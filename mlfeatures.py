"""Machine Learning Tools: feature tools module"""

import numpy as np

class Feature(object):
    """
    Machine Learning Tools Feature Class
    - instantiate with feature matrix X, methods modify self.X and keep self.orig
    """
    def __init__(self, X):
        self.orig = X
        self.X = self.orig

    def get_original(self):
        """return original X"""
        return self.orig

    def get_X(self):
        """return updated X"""
        return self.X

    def add_bias(self):
        """add x0, the bias feature"""
        self.X = np.c_[ np.ones(len(self.X)), self.X ]

    def add_quadratic(self, col):
        """add quadratic feature of column 'col' to X"""
        self.X = np.c_[ self.X, np.multiply(self.X[:,col], self.X[:,col]) ]

    def normalize(self):
        """normalize X, returns tuple: (X_norm, mu, sigma)"""
        X_norm = np.matrix.copy(self.X)
        self.mu = X_norm.mean(axis = 0)
        self.sigma = X_norm.std(axis = 0)
        X_norm = X_norm - mu
        X_norm = X_norm/sigma
        self.X = X_norm

    def polynomial(self, col1, col2, degree = 6):
        """create polynomial data set from two features of self.X"""
        for ii in xrange(1,degree+1):
            for jj in xrange(ii+1):
                X_out = np.c_[ np.multiply(np.power(self.X[:,col1],ii-jj), np.power(self.X[:,col2],jj)) ]
        return X_out
