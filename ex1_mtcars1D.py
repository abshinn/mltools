#!/usr/bin/env python
"""mltools linear regression example 1 script; single variable gradient descent with numpy"""

import mltools
import numpy  as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from ggplot import mtcars
import pdb # debugging

# set print precision
np.set_printoptions(precision = 3)

# example for handpicking from a pandas column:
#y = np.matrix(data.values[:,-1]).T
#X = np.matrix(data.values[:,0:-1])

# let us pick mpg to be y so that we can use regression to guess the mpg given disp
X = np.mat(mtcars.disp.values, dtype = float).T
y = np.mat(mtcars.mpg.values, dtype = float).T

# add x0, all set to 1
X = mltools.addx0(X)

# initialize fitting parameters, array of 0's
theta = np.mat(np.zeros(X.shape[1])).T

# ----------- Gradient Descent ------------
# initialize gradient descent parameters
iterations = 200
alpha = 0.0000003

# compute initial cost
print("Initial cost: J = {:.3f}".format(mltools.cost(X, y, theta)))

# compute gradient descent
theta, J_history = mltools.descent(X, y, theta, alpha, iterations)
print("Theta found using gradient decent: {}".format(theta.T))

# display final cost
print("Final cost: J = {:.3f}".format(J_history[-1]))

# normal equation
theta_NEq = mltools.normalEqn(X, y, lreg = 0.)
J_final = mltools.cost(X, y, theta_NEq)
print("Cost, theta found using normal equation: {:.3f}, {}".format(J_final, theta_NEq.T))

# now, let us see how well the learning algorithm did
print("MPG for a disp of {}:  {}".format( 80, np.mat("[1  80]")*theta_NEq))
print("MPG for a disp of {}:  {}".format(140, np.mat("[1 140]")*theta_NEq))
print("MPG for a disp of {}:  {}".format(200, np.mat("[1 200]")*theta_NEq))

# ----------- Plots -----------
# scatter and best fit
fig, (ax1, ax2) = plt.subplots(nrows = 2, ncols = 1)
ax1.scatter(X[:,1].A, y.A)
ax1.plot(X[:,1].A, (X*theta).A, "g-", label = "descent")
ax1.plot(X[:,1].A, (X*theta_NEq).A, "r-", label = "normal eq")
ax1.legend(bbox_to_anchor=(1.05, 1), loc = 1, borderaxespad = 0.)
ax1.set_ylabel("mpg")
ax1.set_xlabel("disp")

# Decent J history
ax2.plot(J_history)
ax2.set_ylabel("Cost Function")
ax2.set_title("Descent, alpha = {}".format(alpha))

# cost-space contour
# intialize thetas
theta0_vals = np.linspace(25, 35, 100)
theta1_vals = np.linspace( -4,  4, 100)

# intialize J values
J_vals = np.zeros((len(theta0_vals), len(theta1_vals)))

# fill in J_val
for ii in range(len(theta0_vals)):
    for jj in range(len(theta1_vals)):
        t = np.mat([ theta0_vals[ii], theta1_vals[jj] ]).T
        J_vals[ii,jj] = mltools.cost(X, y, t)

theta0_vals, theta1_vals = np.meshgrid(theta0_vals, theta1_vals)

fig = plt.figure(figsize = plt.figaspect(2.0))
ax = fig.add_subplot(1, 1, 1, projection = "3d")
surf = ax.plot_surface(theta0_vals, theta1_vals, J_vals)

plt.show()

# use debug tools to explore variables and plots before script terminates
pdb.set_trace()

# Discussion
#   Gradient descent converges, but at an incorrect local minimum. However, the normal equation
#   comes to the rescue in this case.
