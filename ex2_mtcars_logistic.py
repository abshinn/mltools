#!/usr/bin/env python2.7
"""
mltools
Machine Learning Tools

Logistic regression example using mtcars data set:
- cost minimization using scipy optimization routines
"""

import pdb
import mltools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import fmin_bfgs, fmin_ncg
import ggplot as gg
mtcars = gg.mtcars

# set print precision
np.set_printoptions(precision = 4)

# example for handpicking from a pandas column:
X = np.mat(mtcars.values[:,[3,1]], dtype = float)
y = np.mat(mtcars.cyl.values, dtype = float).T

# one vs all: choose 8 cylinders to be 1, else 0
y = (y == 8.0) + 0.0

# add bias feature and polynomial features
degree = 2
X = mltools.polyfeatures(X[:,0], X[:,1], degree = degree)

# save pre-normalized X
X_prenorm = np.matrix.copy(X)

# normalize features
X, mu, sigma = mltools.featureNormalize(X[:,1:])

# add x0
X = mltools.add_x0(X)

# initialize fitting parameters, array of 0's
initial_theta = np.mat(np.zeros(X.shape[1])).T

# ----------- Gradient Descent ------------
# regularization parameter
rlambda = 50.

# compute initial cost
cost, grad = mltools.LRcost(initial_theta, X, y, rlambda)
print("Initial cost = {}".format(cost))
#print("Initial gradient:\n{}".format(grad))

m, n = X.shape

# note: duplicated effort, need to separate cost and gradient into two different functions
def decorateCost(theta):
    cost, grad = mltools.LRcost(theta, X, y, rlambda)
    return cost
def decorateGrad(theta):
    cost, grad = mltools.LRcost(theta, X, y, rlambda)
    return grad

# optimize
#theta = fmin_bfgs(decorateCost, fprime = decorateGrad, x0 = initial_theta, maxiter = 400)
theta, all_theta = fmin_bfgs(decorateCost, fprime = decorateGrad, x0 = initial_theta, maxiter = 400, retall = True)
#theta, all_theta = fmin_bfgs(decorateCost, x0 = initial_theta, maxiter = 400, retall = True)
theta = theta.reshape(n, 1)
theta = np.mat(theta)

# optimization costs:
print("optimization costs:")
for t in all_theta:
    t = np.mat(t.reshape(n, 1))
    print(decorateCost(t))

# calculate prediction efficiency
predict = np.round(mltools.sigmoid(X*theta))
print (predict == y).mean()*100

# scatter ggplot
mtcars.cyl = mtcars.cyl.astype(str) # changes cyl to a discrete value
point = gg.ggplot(mtcars, gg.aes("disp", "mpg", colour = "cyl")) + gg.geom_point(size = 35)
print point

# scatter pyplot
cyl46 = np.where(y.A1 == 0)
cyl8  = np.where(y.A1 == 1)
fig, ax = plt.subplots()
ax.plot(X_prenorm[cyl46, 1].A, X_prenorm[cyl46, 2].A, "ko", markerfacecolor = "b", markersize = 7, label = "cyl4")
ax.plot(X_prenorm[cyl8 , 1].A, X_prenorm[cyl8 , 2].A, "ko", markerfacecolor = "r", markersize = 7, label = "cyl6")
ax.set_xlabel("disp")
ax.set_ylabel("mpg")

# here is the grid range
margin = 20
u = np.linspace(min(X_prenorm[:,1]).A1 - margin, max(X_prenorm[:,1]).A1 + margin, 50)
v = np.linspace(min(X_prenorm[:,2]).A1 - margin, max(X_prenorm[:,2]).A1 + margin, 50)

# initialize z, the prediction
z = np.zeros(shape = (len(u), len(v)))

# evaluate z = theta*x over the grid
for ii in range(len(u)):
    for jj in range(len(v)):
        uv_poly = mltools.polyfeatures(np.mat(u[ii]), np.mat(v[jj]), degree = degree)
        uv_poly[0,1:] = (uv_poly[0,1:] - mu)/sigma
        z[ii,jj] = uv_poly*theta

# what we want is z transpose
z = z.T

# turn z into probability function
# (values above .5 are 1, otherwise 0)
z = mltools.sigmoid(z)

# contour plot
ax.contourf(u, v, z, [.5, 1], colors = "0.9")
ax.set_title(r"Detecting 8 cylinders; $\lambda$ = {}; {}-degree polynomial".format(rlambda, degree))

# show plot
plt.show()

# explore namespace
print("Exploring namespace...\ntype [c] or [cntrl-d] to exit")
pdb.set_trace()

discussion = """
 TODO
 - explore lambda/degree space
 - train to identify other cylinders
 - find more data to improve learning algorithm?
"""
print(discussion)
