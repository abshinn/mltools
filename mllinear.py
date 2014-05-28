"""Machine Learning Tools: linear regression tools module"""

import numpy as np

def hyp(X, theta):
    """linear regression hypothesis"""
    return X*theta

def costf(X, y, theta, rlambda = 0., hypothesis = hyp):
    """linear regression cost function"""
    m, n = X.shape
    H = hypothesis(X, theta)
    J = ( (H - y).T * (H - y) + rlambda*(theta[1:,0].T * theta[1:,0]) ) / (2.*m)
    return J[0,0] # return single value

def gradientf(X, y, theta, alpha = 0.01, rlambda = 0., hypothesis = hyp):
    """gradient of linear regression cost function"""
    m, n = X.shape
    theta_reg = np.matrix.copy(theta)
    theta_reg[0,0] = 0.
    H = hypothesis(X, theta)
    grad = -alpha * ( X.T*(H - y) - rlambda*theta_reg) / m
    return grad

def descentf(X, y, theta, num_iters, alpha = 0.01, rlambda = 0.):
    """compute gradient descent"""
    m, n = X.shape
    J_history = np.zeros(num_iters)
    grad = np.matrix.copy(theta)
    for ii in range(num_iters):
        # linear regularization TURNED OFF
        #grad = grad - alpha * ( X.T * (X*grad - y) ) / m
        # linear regularization TURNED ON
        grad_temp = np.matrix.copy(grad)
        grad_temp[0,0] = 0.
        H = hypothesis(X, grad)
        grad = grad - alpha * ( X.T * (H - y) - rlambda*grad_temp) / m
        J_history[ii] = cost(X, y, grad, rlambda = rlambda)
    return (grad, J_history)

def normalEqn(X, y, rlambda = 0.):
    """the normal equation, returns theta"""
    if np.linalg.det(X.T*X) == 0.:
        # perhaps use a try/except down below so to not compute X.T*X twice
        print("X.T*X is non-invertible")
        return
    I = np.eye(X.shape[1])
    I[0,0] = 0.
    theta = (X.T*X - rlambda*I).I * X.T * y
    return theta


class Descent(object):
    """Descent
    instantiate with
        X: m x n numpy matrix where m is the number of observations
           and n is the number of features including the bias feature
        y: m x 1 numpy matrix of y values for every observation
    """
    def __init__(self, X, y):
        # to add: check if matrices
        self.X = X
        self.y = y
        self.theta = np.mat(np.zeros((3,1)))
        self.cost_history = []

    def cost(self, rlambda = 0.):
        """regularized cost"""
        return costf(self.X, self.y, self.theta, rlambda = rlambda)

    def update(self, alpha = 0.01, rlambda = 0.):
        """update regularized gradient: return gradient given instance theta"""
        return gradientf(self.X, self.y, self.theta, alpha = alpha, rlambda = rlambda)

    def run(self, alpha = 0.01, rlambda = 0., iterations = 1000):
        """run gradient descent: change theta instance variable"""
        for ii in xrange(iterations):
            self.theta += self.update(alpha = alpha, rlambda = rlambda)
            self.cost_history.append(self.cost(rlambda = rlambda))
        return (self.theta, self.cost_history)
