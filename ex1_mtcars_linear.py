#!/usr/bin/env python2.7
"""
mltools
Machine Learning Tools

Linear regression example using mtcars data set:
    single variable (+quadratic term) gradient descent and normal equation
"""

import pdb
import mlfeatures
import mllinear
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# from matplotlib import cm
# from mpl_toolkits.mplot3d import Axes3D
from ggplot import mtcars

# set plot size
plt.rcParams["figure.figsize"] = 16, 5

# set print precision
np.set_printoptions(precision = 3)

# let us pick mpg to be y so that we can use regression to guess the mpg given disp
X = np.mat(mtcars.disp.values, dtype = float).T
y = np.mat(mtcars.mpg.values, dtype = float).T

# add x1*x1
X = mlfeatures.add_quadratic(X, 0)

# normalize disp and disp^2
X, mu, sigma = mlfeatures.normalize(X)

# add x0
X = mlfeatures.add_x0(X)

# shape of X
m, n = X.shape

# ----------- Gradient Descent ------------
# initialize gradient descent parameters
iterations = 45
alpha = 0.4
rlambda = 4.0 

# instantiate descent
desc = mllinear.Descent(X, y)

# compute initial cost
print("Initial cost: J = {:.3f}".format( desc.cost(rlambda) ))

# compute gradient descent
theta, J_history = desc.run(alpha, rlambda, iterations)
print("Theta found using gradient decent: {}".format( theta.T ))

# display final cost
print("Final cost: J = {:.3f}".format(J_history[-1]))

# normal equation
Norm = mllinear.Ridge()
theta_norm = Norm.run(X, y)
J_final = mllinear.costf(X, y, theta_norm)
print("Cost, theta found using normal equation: {:.3f}, {}".format(J_final, theta_norm.T))

# now, let us see how well the learning algorithm did
predict = np.mat([80, 250, 400]).T
predict = mlfeatures.add_quadratic(predict, 0)
predict = (predict - mu)/sigma
predict = mlfeatures.add_x0(predict)
print("MPG for a disp of {}:  {}".format( 80, predict[0,:]*theta_norm))
print("MPG for a disp of {}:  {}".format(250, predict[1,:]*theta_norm))
print("MPG for a disp of {}:  {}".format(400, predict[2,:]*theta_norm))

# ----------- Plots -----------
# create continuous X variable
xcont = np.mat(np.linspace(min(mtcars.disp), max(mtcars.disp), m)).T
xquad = mlfeatures.add_quadratic(xcont, 0)
xquad = (xquad - mu)/sigma
xquad = mlfeatures.add_x0(xquad)

# scatter and best fit
fig, (ax1, ax2, ax3) = plt.subplots(nrows = 1, ncols = 3)
ax1.scatter(np.array(mtcars.disp), y.A1)
ax1.plot(xcont.A1, (xquad*theta).A1, "g-", label = "descent")
ax1.plot(xcont.A1, (xquad*theta_norm).A1, "r-", label = "normal eq")
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
theta0_vals = np.linspace(theta_norm[0,0]-20, theta_norm[0,0]+20, 100)
theta1_vals = np.linspace(theta_norm[1,0]-20, theta_norm[1,0]+20, 100)
theta2_val = theta_norm[2,0]

# intialize J values
J_vals = np.zeros((len(theta0_vals), len(theta1_vals)))

# fill in J_val
for ii in range(len(theta0_vals)):
    for jj in range(len(theta1_vals)):
            t = np.mat([ theta0_vals[ii], theta1_vals[jj], theta2_val ]).T
            J_vals[ii,jj] = mllinear.costf(X, y, t, rlambda = rlambda)

# J_vals needs to be transposed
J_vals = J_vals.T

# contour plot
ctr = plt.contour(theta0_vals, theta1_vals, J_vals, np.logspace(-1, 2, 10))
ax3.clabel(ctr, inline = 1, fontsize = 10)
ax3.plot(theta_norm[0,0], theta_norm[1,0], 'rx', ms = 10, mew = 2)
ax3.plot(theta[0,0], theta[1,0], 'bo')
ax3.set_xlabel(r"$\theta_0$")
ax3.set_ylabel(r"$\theta_1$")

# if False:
#     # mesh grid for surface plot
#     theta0_vals, theta1_vals = np.meshgrid(theta0_vals, theta1_vals)
# 
#     # surface plot
#     fig2 = plt.figure(figsize = plt.figaspect(2.0))
#     ax = fig2.add_subplot(1, 1, 1, projection = "3d")
#     surf = ax.plot_surface(theta0_vals, theta1_vals, J_vals, cmap = cm.coolwarm, antialiased = False)

# show plots
plt.show()

print("""
Discussion
1D:
  Without featrure normalization, gradient descent fails to find the optimal theta.
Quadratic:
  Descent with regularization gets very close to global optimum. Oddly, the final
  cost found in descent differs from the cost found by the normal equation. This
  is perhaps due to regularization. Also, the cost starts to increase towards the
  end of num_iterations...
""")

# explore namespace
print("Exploring namespace using Pdb...\ntype [c] or [cntrl-d] to exit")
pdb.set_trace()
