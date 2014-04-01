#!/usr/bin/env python3
"""mltools linear regression example 2 script; multiple variable linear regression"""

import numpy as np
#import pandas as pd
from ggplot import mtcars
import compute
import matplotlib.pyplot as plt
import pdb

# set print precision
np.set_printoptions(precision = 3)

# example for handpicking from a pandas column, assume y is last column:
#y = np.matrix(data.values[:,-1]).T
#X = np.matrix(data.values[:,0:-1])

# let us pick mpg to be y so that we can use regression to guess the mpg given cyl
X = np.mat(mtcars.values[:,2:4], dtype = float) # cyl and displ columns
y = np.mat(mtcars.mpg.values, dtype = float).T

# normalize features
X, mu, sigma = compute.featureNormalize(X)

# add x0, all set to 1
X = compute.addx0(X)

# initialize fitting parameters, array of 0's
theta = np.matrix(np.zeros(X.shape[1])).T

# ----------- Gradient Descent ------------
# initialize gradient descent parameters
iterations = 50
alpha = 0.1

# compute initial cost
print("Initial cost: J = {}".format(compute.cost(X, y, theta)))

# compute gradient descent
theta, J_history = compute.descent(X, y, theta, alpha, iterations)
print("Cost, theta found using gradient decent: {:.3f}, {}".format(J_history[-1], theta.T))

# ----------- Normal Equation ------------
theta = compute.normalEqn(X, y)
J_final = compute.cost(X, y, theta)
print("Cost, theta found using normal equation: {:.3f}, {}".format(J_final, theta.T))

# ----------- Plots -----------
# J history
plt.figure(1)
plt.plot(J_history)
plt.ylabel("J")
plt.title("alpha = {}".format(alpha))

plt.show()

# use debug tools to explore variables and plots before script terminates
pdb.set_trace()
