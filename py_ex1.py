#!/usr/bin/env python3
"""mltools linear regression example 1; single variable with pandas"""

import numpy  as np
import pandas as pd
import compute
#import ggplot as gg
import matplotlib.pyplot as plt
import pdb

data = pd.read_csv('data_ex1.txt', header = None)
data.columns = ["X", "y"]
length = len(data)


# ----------- Gradient Descent ------------
# add x0, all set to 1
data = pd.concat([pd.DataFrame([1]*length), data], axis = 1)
data.columns = [0, 1, "y"]

X = data[[0, 1]]
y = data[["y"]]

# initialize fitting parameters, array of 0's
theta = pd.DataFrame(np.zeros(X.shape[1]))

# initialize gradient descent parameters
iterations = 1500
alpha = 0.01

# compute initial cost
print("Initial cost: J = {}".format(compute.cost(X, y, theta)))

# compute gradient descent
theta, J_history = compute.descent(X, y, theta, alpha, iterations)
print("Theta found using gradient decent: {}".format(theta.T.values))


# ----------- Plots -----------
# ggplot for python
#pointplot = gg.ggplot(data, gg.aes("X", "y")) + gg.geom_point(colour="steelblue")
#print(pointplot)

# matplotlib line and fit
plt.figure(1)
plt.scatter(X[[1]], data.y) # data no longer has column name X
plt.plot(X[[1]], X.dot(theta)[[0]], "r-")
plt.ylabel("y")
plt.xlabel("x")

# J history
plt.figure(2)
plt.plot(J_history)
plt.ylabel("J")
plt.title("alpha = {}".format(alpha))

plt.show()

if __name__ == "__main__":
    pass
