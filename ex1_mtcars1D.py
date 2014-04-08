#!/usr/bin/env python
"""mltools linear regression example 1 script; single variable gradient descent with numpy"""

import pdb
import mltools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from ggplot import mtcars

# set print precision
np.set_printoptions(precision = 3)

# let us pick mpg to be y so that we can use regression to guess the mpg given disp
X = np.mat(mtcars.disp.values, dtype = float).T
y = np.mat(mtcars.mpg.values, dtype = float).T

# add x2, the square of x1
#X = mltools.addxsquare(X, 0)

# normalize disp
X, mu, sigma = mltools.featureNormalize(X)

# add x0
X = mltools.addx0(X)

# initialize fitting parameters, array of 0's
theta = np.mat(np.zeros(X.shape[1])).T

# ----------- Gradient Descent ------------
# initialize gradient descent parameters
iterations = 20 
alpha = 0.3
lreg = 0.0

# compute initial cost
print("Initial cost: J = {:.3f}".format(mltools.cost(X, y, theta, lreg = lreg)))

# compute gradient descent
theta, J_history = mltools.descent(X, y, theta, alpha, iterations, lreg = lreg)
print("Theta found using gradient decent: {}".format(theta.T))

# display final cost
print("Final cost: J = {:.3f}".format(J_history[-1]))

# normal equation
theta_NEq = mltools.normalEqn(X, y)
J_final = mltools.cost(X, y, theta_NEq)
print("Cost, theta found using normal equation: {:.3f}, {}".format(J_final, theta_NEq.T))

# now, let us see how well the learning algorithm did
print("MPG for a disp of {}:  {}".format( 80, np.mat([1, ( 80 - mu)/sigma])*theta_NEq))
print("MPG for a disp of {}:  {}".format(250, np.mat([1, (250 - mu)/sigma])*theta_NEq))
print("MPG for a disp of {}:  {}".format(400, np.mat([1, (400 - mu)/sigma])*theta_NEq))

# ----------- Plots -----------
# scatter and best fit
fig, (ax1, ax2) = plt.subplots(nrows = 2, ncols = 1)
ax1.scatter(X[:,1].A, y.A)
ax1.plot(X[:,1].A, (X*theta).A, "g-", label = "descent")
ax1.plot(X[:,1].A, (X*theta_NEq).A, "r-", label = "normal eq")
ax1.legend(bbox_to_anchor=(1.05, 1), loc = 1, borderaxespad = 0.)
ax1.set_ylabel("mpg")
ax1.set_xlabel("disp")

# descent J history
ax2.plot(J_history)
ax2.set_ylabel("Cost Function")
ax2.set_title("Descent, alpha = {}".format(alpha))

# fix crowding in above plots
fig.set_tight_layout(True)

# cost-space contour and surface plots
# intialize thetas
theta0_vals = np.linspace(10, 30, 100)
theta1_vals = np.linspace(-15, 5, 100)

# intialize J values
J_vals = np.zeros((len(theta0_vals), len(theta1_vals)))

# fill in J_val
for ii in range(len(theta0_vals)):
    for jj in range(len(theta1_vals)):
        t = np.mat([ theta0_vals[ii], theta1_vals[jj] ]).T
        J_vals[ii,jj] = mltools.cost(X, y, t)

# J_vals needs to be transposed
J_vals = J_vals.T

# contour plot
plt.figure()
ctr = plt.contour(theta0_vals, theta1_vals, J_vals, np.logspace(-1, 2, 10))
plt.clabel(ctr, inline = 1, fontsize = 10)
plt.plot(theta_NEq[0,0], theta_NEq[1,0], 'rx', ms = 10, mew = 2)
plt.plot(theta[0,0], theta[1,0], 'bo')
plt.xlabel(r"$\theta_0$")
plt.ylabel(r"$\theta_1$")

# mesh grid for surface plot
theta0_vals, theta1_vals = np.meshgrid(theta0_vals, theta1_vals)

# surface plot
fig2 = plt.figure(figsize = plt.figaspect(2.0))
ax = fig2.add_subplot(1, 1, 1, projection = "3d")
surf = ax.plot_surface(theta0_vals, theta1_vals, J_vals, cmap = cm.coolwarm, antialiased = False)

# show plots
plt.show()

# use debug tools to explore variables before script terminates
#pdb.set_trace()

# Discussion
#   Interestingly, without featrure normalization, gradient descent fails to descend in the 
#   direction of the optimal theta0.
