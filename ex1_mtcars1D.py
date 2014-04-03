#!/usr/bin/env python3
"""mltools linear regression example 1 script; single variable gradient descent with numpy"""

import mltools
import numpy  as np
import matplotlib.pyplot as plt
from ggplot import mtcars
import pdb # debugging

# set print precision
np.set_printoptions(precision = 3)

# example for handpicking from a pandas column:
#y = np.matrix(data.values[:,-1]).T
#X = np.matrix(data.values[:,0:-1])

# let us pick mpg to be y so that we can use regression to guess the mpg given cyl
X = np.mat(mtcars.cyl.values, dtype = float).T
y = np.mat(mtcars.mpg.values, dtype = float).T

length = len(y)

# add x0, all set to 1
X = mltools.addx0(X)

# initialize fitting parameters, array of 0's
theta = np.mat(np.zeros(X.shape[1])).T

# ----------- Gradient Descent ------------
# initialize gradient descent parameters
iterations = 1500
alpha = 0.03

# compute initial cost
print("Initial cost: J = {:.3f}".format(mltools.cost(X, y, theta)))

# compute gradient descent
theta, J_history = mltools.descent(X, y, theta, alpha, iterations)
print("Theta found using gradient decent: {}".format(theta.T))

# compute final cost
print("Final cost: J = {:.3f}".format(J_history[-1]))

# now, let us see how well the learning algorithm did
cyl4 = np.mat("[1 4]") * theta
print("MPG for 4 cylinders: {}".format(cyl4))

# ----------- Plots -----------
# scatter and best fit
fig, (ax1, ax2) = plt.subplots(nrows = 2, ncols = 1)
ax1.scatter(X[:,1].A, y.A)
ax1.plot(X[:,1].A, (X*theta).A, "r-")
ax1.set_ylabel("mpg")
ax1.set_xlabel("cyl")

# J history
ax2.plot(J_history)
ax2.set_ylabel("Cost Function")
ax2.set_title("alpha = {}".format(alpha))

plt.show()

# use debug tools to explore variables and plots before script terminates
pdb.set_trace()

# note:
# because of the discrete nature of the X variable, cyl, this is a crude
# regression example... the next step up should include the displ variable
# in a multi-variable regression in order to make better predictions
