"""Machine Learning Tools: linear regression tools module"""

import numpy as np

def hypothesis(X, theta):
    """linear regression hypothesis"""
    return X*theta

def cost(X, y, theta, rlambda = 0.):
    """linear regression cost function"""
    m = len(y)
    H = hypothesis(X, theta)
    J = ( (H - y).T * (H - y) + rlambda*(theta[1:,:].T * theta[1:,:]) ) / (2.*m)
    return J[0,0] # return single value

def descent(X, y, theta, alpha, num_iters, rlambda = 0.):
    """compute gradient descent"""
    m = len(y)
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
    I = np.eye(X.shape[1])
    I[0,0] = 0.
    theta = (X.T*X - rlambda*I).I * X.T * y
    return theta
