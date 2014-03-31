#!/usr/bin/env python3
"""mltools linear regression example 1 script; single variable gradient descent with numpy"""

import numpy  as np
import pandas as pd
import compute
import ggplot as gg
import matplotlib.pyplot as plt
import pdb # debugging

# set print precision
np.set_printoptions(precision = 3)

# use pandas to read in csv
#data = pd.read_csv('data_ex1.txt', header = None)

# example for handpicking from a pandas column:
#y = np.matrix(data.values[:,-1]).T
#X = np.matrix(data.values[:,0:-1])

# let us pick mpg to be y so that we can use regression to guess the mpg given cyl
X = np.mat(gg.mtcars.cyl.values).T
y = np.mat(gg.mtcars.mpg.values).T

length = len(y)

# add x0, all set to 1
X = compute.addx0(X)

# initialize fitting parameters, array of 0's
theta = np.matrix(np.zeros(X.shape[1])).T

# ----------- Gradient Descent ------------
# initialize gradient descent parameters
iterations = 1500
alpha = 0.03

# compute initial cost
print("Initial cost: J = {:.3f}".format(compute.cost(X, y, theta)))

# compute gradient descent
theta, J_history = compute.descent(X, y, theta, alpha, iterations)
print("Theta found using gradient decent: {}".format(theta.T))

# compute final cost
print("Final cost: J = {:.3f}".format(J_history[-1]))

# now, let us see how well the learning algorithm did
cyl4 = np.mat("[1 4]") * theta
print("MPG for 4 cylinders: {}".format(cyl4))

# ----------- Plots -----------
# matplotlib line and fit
plt.figure(1)
plt.scatter(X[:,1].A, y.A)
plt.plot(X[:,1].A, (X*theta).A, "r-")
plt.ylabel("y")
plt.xlabel("x")

# J history
plt.figure(2)
plt.plot(J_history)
plt.ylabel("J")
plt.title("alpha = {}".format(alpha))

plt.show()

# use debug tools to explore variables and plots before script terminates
pdb.set_trace()

# note:
# because of the discrete nature of the X variable, cyl, this is a crude
# regression example... the next step up should include the displ variable
# in a multi-variable regression in order to make better predictions
