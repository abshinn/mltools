#!/usr/bin/env python
"""mltools linear regression example 2 script; multiple variable linear regression"""

import numpy as np
from ggplot import mtcars
import mltools
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
X, mu, sigma = mltools.featureNormalize(X)

# add x0, all set to 1
X = mltools.addx0(X)

# initialize fitting parameters, array of 0's
theta = np.mat(np.zeros(X.shape[1])).T

# ----------- Gradient Descent ------------
# initialize gradient descent parameters
iterations = 50
alpha = 0.1

# compute initial cost
print("Initial cost: J = {}".format(mltools.cost(X, y, theta)))

# compute gradient descent
theta, J_history = mltools.descent(X, y, theta, alpha, iterations)
print("Cost, theta found using gradient decent: {:.3f}, {}".format(J_history[-1], theta.T))

# estimate mpg using computed gradient descent parameters
### first column of X should not be "un-normalized"...
#cyl4 = (sigma * np.mat("[1.0 4.0 80.0]") - mu) * theta
#print("MPG for 4 cylinders: {}".format(cyl4))

# ----------- Normal Equation ------------
theta_NEq = mltools.normalEqn(X, y, lreg = 0.)
J_final = mltools.cost(X, y, theta_NEq)
print("Cost, theta found using normal equation: {:.3f}, {}".format(J_final, theta_NEq.T))

# ----------- Plots -----------
# scatter and best fit
fig, (ax1, ax2, ax3) = plt.subplots(nrows = 3, ncols = 1)

# cylinder v mpg
ax1.scatter(X[:,1].A, y.A)
theta_cyl = theta
theta_cyl[2] = 0.
ax1.plot(X[:,1].A, (X*theta_cyl).A, "r-")
ax1.set_xlabel("cyl")
ax1.set_ylabel("mpg")

# disp v mpg
ax2.scatter(X[:,2].A, y.A)
theta_disp = theta
theta_disp[1] = 0.
ax2.plot(X[:,2].A, (X*theta_disp).A, "r-")
ax2.set_xlabel("disp")
ax2.set_ylabel("mpg")

# J history
ax3.plot(J_history)
ax3.set_ylabel("J")
ax3.set_title("alpha = {}".format(alpha))

plt.show()

# use debug tools to explore variables and plots before script terminates
pdb.set_trace()
