from misc_utils.plotting.plot3d import *
from misc_utils.shapes import *
from misc_utils.shapes.io import *


m = load_mesh('shapes/tr_reg_000_rem.ply')
v = np.array(m.vertices)
f = np.array(m.triangles)

n = load_mesh('shapes/smpl_base_neutro_rem.obj')
w = np.array(n.vertices)
g = np.array(n.triangles)

v0 = v[:,0]
v1 = v[:,1]
v2 = v[:,2]
w0 = w[:,0]
w1 = w[:,1]
w2 = w[:,2]

print(f.shape)
print(g.shape)
print(v0.min(), v0.max())
print(v1.min(), v1.max())
print(v2.min(), v2.max())

print(w0.min(), w0.max())
print(w1.min(), w1.max())
print(w2.min(), w2.max())

c0 = np.arange(len(v0))
c1 = np.arange(len(w0))

plot_point_cloud([(v0, v1, v2), (w0, w1, w2)], colors=[c0, c1])

for i in range(len(g)):
    print(f[i], g[i])