import numpy as np
import matplotlib.pyplot as plt

from misc_utils.normalization import normalize_vector
from envs.utils import move_points_gaussian2d

mu,sigma,n = 0.,0.5,10

def normal(x,mu,sigma):
    return ( 2.*np.pi*sigma**2.)**-.5 * np.exp( -.5 * (x-mu)**2. / sigma**2. )
#
# x = np.random.normal(mu,sigma,n)
# y = normal(x,mu,sigma)
#
#
# fig = plt.figure()
# plt.scatter(x,y)
# plt.show()
# plt.close()

def normal2(x, mu, sigma):
    return (1/(sigma*np.sqrt(2*np.pi))) * (np.e * -0.5 * ((x-mu)**2) / (sigma**2))

# x = np.arange(11)
# x = np.array([0,1,2,8,9,10,11,20,26,31,66])
x = np.random.normal(0,2,100)
# sorted_x = np.sort(x)
# x = np.sort(x)
y = np.linspace(0,3,100)
x_copy = x.copy()
y_copy = y.copy()

xy = np.array(list(zip(x, y)))
center = xy[50]
# dist = np.linalg.norm(center-xy, axis=1, ord=1)
dist = (center[0] - x) + (center[1] - y)

sorted_dist = np.argsort(dist)
sorted_dist = np.sort(dist)
y_distr = normal(dist, mu, sigma)
y_distr = normalize_vector(y_distr)
fig = plt.figure()
plt.scatter(x,y, s=[80]*len(x) ,c='blue')
plt.scatter(x[50],y[50], s=[80], c='yellow')
plt.scatter(dist,y_distr, s=[80]*len(x), c='red')
plt.scatter(x + y_distr, y + y_distr, s=[30]*len(x), c='green')
plt.show()
plt.close()

x, y = move_points_gaussian2d(50, x, y, 1, 1, 10, mu, sigma)
fig = plt.figure()
plt.scatter(x,y, s=[80]*len(x) ,c='blue')
plt.scatter(x[50],y[50], s=[80] ,c='yellow')
plt.scatter(dist,y_distr, s=[80]*len(x), c='red')
plt.scatter(x, y, s=[30]*len(x), c='green')
plt.show()
plt.close()


print('AHAHAH')

''' TEST TO SEE IF PUTTING THE "CENTER" IN THE FIRST POSITION THE NORMAL DISTR KEEPS WORKING'''
x = x_copy
y = y_copy
xy = list(zip(x, y))
center = xy.pop(50)
xy = [center] + xy
xy = np.array(xy)

# dist = np.linalg.norm(center-xy, axis=1, ord=1)
dist = (center[0] - x) + (center[1] - y)

sorted_dist = np.argsort(dist)
sorted_dist = np.sort(dist)
y_distr = normal(dist, mu, sigma)
y_distr = normalize_vector(y_distr)
fig = plt.figure()
plt.scatter(x,y, s=[80]*len(x) ,c='blue')
plt.scatter(x[50],y[50], s=[80], c='yellow')
plt.scatter(dist,y_distr, s=[80]*len(x), c='red')
plt.scatter(x + y_distr, y + y_distr, s=[30]*len(x), c='green')
plt.show()
plt.close()
exit(5)




original_y = y.copy()
print(len(y))
center_x = x[50]
# dist = np.linalg.norm(center-x, axis=0)
dist_x = abs(x - center_x)
# TODO: normalize y to be within a certain range
y_distr = normal(dist_x, mu, sigma)
y_distr = normalize_vector(y_distr)
print(center_x)
print(dist_x)
print(y_distr)
fig = plt.figure()
plt.scatter(x,y_distr, s=[80]*len(x))
plt.scatter(x,y, s=[80]*len(x))
plt.scatter(x,y + normalize_vector(y*y_distr), s=[30]*len(x))
plt.scatter(x,y + y_distr, s=[30]*len(x), c='red')
plt.show()
plt.close()

y = y + normalize_vector(y*y_distr)

center_y = y[50]
dist_y = x = np.sort(np.random.normal(0,2,100)) # abs(y - center_y)
x_distr = normal(np.sort(dist_y), mu, sigma)
x_distr = normalize_vector(x_distr)
fig = plt.figure()
plt.scatter(x_distr,y, s=[80]*len(x))
plt.scatter(x,y, s=[80]*len(x))
plt.scatter(x+normalize_vector(x*x_distr),y, s=[30]*len(x))
plt.show()
plt.close()

plt.scatter(original_x,original_y, s=[30]*len(x), c='blue')
center_x = original_x[50]
center_y = original_y[50]
plt.scatter(center_x, center_y, c='green', s=100)
plt.scatter(x+normalize_vector(x*x_distr),y, s=[30]*len(x), c='red')
x = x+normalize_vector(x*x_distr)
center_x = x[50]
center_y = y[50]
plt.scatter(center_x, center_y, c='yellow', s=100)
plt.show()
plt.close()
exit(5)
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D

#Parameters to set
mu_x = 0
variance_x = 3

mu_y = 0
variance_y = 15

#Create grid and multivariate normal
x = np.linspace(-10,10,30)
y = np.linspace(-10,10,30)
X, Y = np.meshgrid(x,y)
pos = np.empty(X.shape + (2,))
pos[:, :, 0] = X; pos[:, :, 1] = Y
rv = multivariate_normal([mu_x, mu_y], [[variance_x, 0], [0, variance_y]])
print(X.shape)
#Make a 3D plot
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, rv.pdf(pos),cmap='viridis',linewidth=0)
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
plt.show()
print(rv.pdf(pos).shape)
ax.scatter(x,y,rv.pdf(pos))