#!/usr/bin/env python3
"""mltools linear regression example 2 script; multiple variable linear regression"""

import numpy  as np
import pandas as pd
import compute
import matplotlib.pyplot as plt
import pdb

# use pandas to read in csv
data = pd.read_csv('data_ex2.txt', header = None)

# assume data is formated such that y is the last column
y = data.values[:,-1]
X = data.values[:,0:-1]

length = len(y)

# add x0, all set to 1
X = compute.addx0(X)

# initialize fitting parameters, array of 0's
theta = np.zeros(X.shape[1])

# normalize features
X, mu, sigma = compute.featureNormalize(X)

# ----------- Gradient Descent ------------
# initialize gradient descent parameters
iterations = 1500
alpha = 0.01

# compute initial cost
print("Initial cost: J = {}".format(compute.cost(X, y, theta)))

# compute gradient descent
theta, J_history = compute.descent(X, y, theta, alpha, iterations)
print("Theta found using gradient decent: {}".format(theta))

# ----------- Normal Equation ------------
theta = compute.normalEqn(X, y)
print("Theta found using normal equation: {}".format(theta))

# ----------- Plots -----------
