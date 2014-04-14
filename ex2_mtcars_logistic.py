#!/usr/bin/env python
"""mltools example 2, logistic regression"""

import pdb
import mltools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import fmin_bfgs
#from matplotlib import cm
#from mpl_toolkits.mplot3d import Axes3D
#from ggplot import mtcars
import ggplot as gg
mtcars = gg.mtcars

# set print precision
np.set_printoptions(precision = 4)

# example for handpicking from a pandas column:
y = np.mat(mtcars.cyl.values, dtype = float).T
X = np.mat(mtcars.values[:,[3,1]], dtype = float)

# one vs all
y = (y == 8.0) + 0.0




# add bias feature and polynomial features
X = mltools.polyfeatures(X[:,0], X[:,1])

# initialize fitting parameters, array of 0's
initial_theta = np.mat(np.zeros(X.shape[1])).T

# ----------- Gradient Descent ------------
# regularization parameter
rlambda = 0

# compute initial cost
cost, grad = mltools.LRcost(initial_theta, X, y, rlambda)
print("Initial cost: J = {}".format(cost))

m, n = X.shape

# reinitialize theta
initial_theta = np.zeros(shape = (n, 1))

# regularization parameter
rlambda = 1

def decorateCost(theta):
    cost, grad = mltools.LRcost(theta, X, y, rlambda)
    return cost
def decorateGrad(theta):
    cost, grad = mltools.LRcost(theta, X, y, rlambda)
    return grad

# optimize
#theta = fmin_bfgs(decorateCost, fprime = decorateGrad, x0 = initial_theta, maxiter = 400)
theta = fmin_bfgs(decorateCost, x0 = initial_theta, maxiter = 400)
theta = theta.reshape(n, 1)
theta = np.mat(theta)

# calculate prediction efficiency
predict = np.round(mltools.sigmoid(X*theta))
print (predict == y).mean()*100

pdb.set_trace()

# scatter plot
mtcars.cyl = mtcars.cyl.astype(str) # changes cyl to a discrete value
point = gg.ggplot(mtcars, gg.aes("disp", "mpg", colour = "cyl")) + gg.geom_point(size = 35)
#print point

## scatter plot
cyl46 = np.where(y.A1 == 0)
cyl8  = np.where(y.A1 == 1)
fig, ax = plt.subplots()
ax.plot(X[cyl46, 1].A, X[cyl46, 2].A, "kd", markerfacecolor = "r", markersize = 7, label = "cyl4")
ax.plot(X[cyl8 , 1].A, X[cyl8 , 2].A, "kH", markerfacecolor = "g", markersize = 7, label = "cyl6")
ax.set_xlabel("disp")
ax.set_ylabel("mpg")
#ax.legend(loc = 1)
#plt.show()

# here is the grid range
u = np.linspace(min(X[:,1]).A1, max(X[:,1]).A1, 50)
v = np.linspace(min(X[:,2]).A1, max(X[:,2]).A1, 50)

z = np.zeros(shape = (len(u), len(v)))


# evaluate z = theta*x over the grid
for ii in range(len(u)):
    for jj in range(len(v)):
        z[ii,jj] = mltools.polyfeatures(np.mat(u[ii]), np.mat(v[jj]))*theta
z = z.T # important to transpose z before calling contour

# contour plot
#plt.figure()
ax.contour(u, v, z, [0, 0])
#plt.clabel(ctr, inline = 1, fontsize = 10)
#ax.plot(theta[0,0], theta[1,0], 'rx')
#ax.legend(["y = 1", "y = 0", "Decision Boundary"])

plt.show()

pdb.set_trace()
