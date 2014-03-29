#!/usr/bin/env python3
"""mltools linear regression example 1 script; single variable gradient descent with numpy"""

import numpy  as np
import pandas as pd
import compute
import matplotlib.pyplot as plt
import pdb # debugging

# use pandas to read in csv
data = pd.read_csv('data_ex1.txt', header = None)

# assume data is formated such that y is the last column
y = np.matrix(data.values[:,-1]).T
X = np.matrix(data.values[:,0:-1])

length = len(y)

# add x0, all set to 1
X = compute.addx0(X)

# initialize fitting parameters, array of 0's
theta = np.matrix(np.zeros(X.shape[1])).T

# ----------- Gradient Descent ------------
# initialize gradient descent parameters
iterations = 1500
alpha = 0.01

# compute initial cost
print("Initial cost: J = {}".format(compute.cost(X, y, theta)))

# compute gradient descent
theta, J_history = compute.descent(X, y, theta, alpha, iterations)
print("Theta found using gradient decent: {}".format(theta))

# ----------- Plots -----------
# matplotlib line and fit
plt.figure(1)
plt.scatter(np.array(X[:,1]), np.array(y))
plt.plot(np.array(X[:,1]), np.array(X*theta), "r-")
plt.ylabel("y")
plt.xlabel("x")

# J history
plt.figure(2)
plt.plot(J_history)
plt.ylabel("J")
plt.title("alpha = {}".format(alpha))

plt.show()
