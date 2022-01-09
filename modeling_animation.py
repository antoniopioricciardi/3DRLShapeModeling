import os
import pickle
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from open3d.cpu.pybind.utility import Vector3dVector, Vector3iVector
import time
from open3d.visualization import Visualizer

from misc_utils.shapes.io import *
from misc_utils.normalization import normalize_vector
from misc_utils.shapes.operations import *
from misc_utils.plotting.plot3d import *
import matplotlib.pyplot as plt
from read_shape import load_mesh_new
from inits import get_coordinates

def from_shape(spread=5, num_triangles=0, shape_path='./shapes/'):
    target_vert, canvas_vert, tri_list = load_mesh()
    t_x = target_vert[:,0]
    t_y = target_vert[:,1]
    t_z = target_vert[:,2]
    c_x = canvas_vert[:,0]
    c_y = canvas_vert[:,1]
    c_z = canvas_vert[:,2]
    return t_x * spread, t_y * spread, t_z * spread, c_x * spread, c_y * spread, c_z * spread, tri_list

RESULTS_PATH = 'results'

spread = 5


shape_name1 = 'tr_reg_000_rem.ply'# '25.ply' # 'tr_reg_007'
shape_path1 = os.path.join('shapes', shape_name1)
shape_name2 ='smpl_base_neutro_rem.ply' # 'sphere_curvature_flow.ply' #
shape_path2 = os.path.join('shapes', shape_name2)

# open a file, where you stored the pickled data
transition_path = os.path.join(RESULTS_PATH, shape_name1 + '--' + shape_name2)
transition_path = os.path.join(transition_path, 'transitions')
model_name = 'gaussian_model-seed_42-spread_5-lr_0.0001-sigma_0.5-neighborsmovementscale_1'
file = open(os.path.join(transition_path, model_name), 'rb')

# read information from file
transitions = pickle.load(file)
file.close()

transition_matrix = transitions['canvas_transition_matrix']
masks_transitions = transitions['vertex_masks']
config_33 = transitions['33_config']
config_66 = transitions['66_config']
final_configuration = transitions['final_configuration']
canvas_vertices_transitions = transitions['canvas_vertices_transitions']
actions_transitions = transitions['actions_transitions']
print(final_configuration.shape)

# plot_point_cloud([(final_configuration[0], final_configuration[1], final_configuration[2])], colors=[(np.arange(len(final_configuration[0])))])

mesh1, V, F = load_mesh_new(shape_path1, simplify=True, normalize=True)
mesh2, _, G = load_mesh_new(shape_path2, simplify=False, normalize=True)

# mesh2, _, _ = create_sphere_from_mesh(V, F)
# mesh1 = normalize_mesh(mesh1, 5)
# mesh2 = normalize_mesh(mesh2, 5)
mesh1.compute_vertex_normals()
# o3d.visualization.draw([mesh1, n1])

mesh2.compute_vertex_normals()
o3d.visualization.draw([mesh2])
o3d.visualization.draw([mesh1, mesh2])
v2 = np.array(mesh2.vertices)

s_x, s_y, s_z = get_coordinates(V, 2.5)
c_x, c_y, c_z = get_coordinates(v2, 2.5)
# plot_point_cloud([(final_configuration[0], final_configuration[1], final_configuration[2]), (c_x, c_y, c_z)])
print(canvas_vertices_transitions[0][0] == c_x, canvas_vertices_transitions[0][1] == c_y, canvas_vertices_transitions[0][2] == c_z)

print(canvas_vertices_transitions.shape)

print(s_x.min(), s_x.max(), '--', s_y.min(), s_y.max(), '--', s_z.min(), s_z.max())
print(c_x.min(), c_x.max(), '--', c_y.min(), c_y.max(), '--', c_z.min(), c_z.max())
# c_x = normalize_vector(c_x)
# c_y = normalize_vector(c_y)
# c_z = normalize_vector(c_z)

vis = Visualizer()
vis.create_window()
vis.add_geometry(mesh2)

prev_x = None
prev_y = None
prev_z = None

count_x = 0
count_y = 0
count_z = 0
print('plotting!')
prev_cx = None
n_different = 0
for i in range(len(transition_matrix)):
    c_x = canvas_vertices_transitions[i][0]
    c_y = canvas_vertices_transitions[i][1]
    c_z = canvas_vertices_transitions[i][2]
    # print(np.count_nonzero(np.array(c_x == prev_cx)))
    if i>1:
        if np.count_nonzero(np.array(c_x == prev_cx)) != 249:
            mask = np.array(masks_transitions[i-1], dtype=np.int16)
            n_different+=1
    prev_cx = c_x.copy()
    if i % 200 == 0:
        # print(c_x.min(), c_x.max(), '--', c_y.min(), c_y.max(), '--', c_z.min(), c_z.max())
        mesh2.vertices = Vector3dVector(np.vstack((c_x, c_y, c_z)).T)
        mesh2.compute_vertex_normals()
        vis.update_geometry(mesh2)
        vis.poll_events()
        vis.update_renderer()


mesh2.vertices = Vector3dVector(np.vstack((c_x, c_y, c_z)).T)
mesh2.compute_vertex_normals()
vis.update_geometry(mesh2)
vis.poll_events()
vis.update_renderer()

mesh2.compute_vertex_normals()
o3d.visualization.draw([mesh2])

final_x = final_configuration[0]
final_y = final_configuration[1]
final_z = final_configuration[2]

print(n_different, '--', len(transition_matrix))
exit(4)

for i in range(len(transition_matrix)):
    new_c_x, new_c_y, new_c_z = transition_matrix[i]

    # if i > 0 and not (i%16) == 0:
    #     if np.all(prev_x != new_c_x):
    #         print('x')
    #         count_x += 1
    #     if np.all(prev_y != new_c_y):
    #         print('y')
    #         count_y += 1
    #     if np.all(prev_z != new_c_z):
    #         print('z')
    #         count_z += 1
    prev_x = new_c_x
    prev_y = new_c_y
    prev_z = new_c_z
    # print(f'x: {new_c_x} - y: {new_c_y} - z: {new_c_y}')
    mask = np.array(masks_transitions[i][1]).astype(int)
    # print(i, mask)
    c_x[mask] = new_c_x
    c_y[mask] = new_c_y
    c_z[mask] = new_c_z
    # vert = np.asarray(mesh2.vertices)
    # print(vert[mask])
    if i % 200 == 0:
        print(c_x.min(), c_x.max(), '--', c_y.min(), c_y.max(), '--', c_z.min(), c_z.max())
        mesh2.vertices = Vector3dVector(np.vstack((c_x, c_y, c_z)).T)
        mesh2.compute_vertex_normals()
        vis.update_geometry(mesh2)
        vis.poll_events()
        vis.update_renderer()

final_x = final_configuration[0]
final_y = final_configuration[1]
final_z = final_configuration[2]

print(c_x == final_x, c_y == final_y, c_z == final_z)
print(len(final_x), len(c_x))

mesh2.vertices = Vector3dVector(np.vstack((c_x, c_y, c_z)).T)
mesh2.compute_vertex_normals()
vis.update_geometry(mesh2)
vis.poll_events()
vis.update_renderer()

mesh2.compute_vertex_normals()
o3d.visualization.draw([mesh1,mesh2])


#
# config_33 = transitions['33_config']
# config_66 = transitions['66_config']
# config_final = transitions['final_configuration']
#
# f_x = config_final[0]
# print(f_x[0] -V[0])
# print(f_x[1] -V[1])
# f_y = config_final[1]
# f_z = config_final[2]
# mesh2.vertices = Vector3dVector(np.vstack((f_x, f_y, f_z)).T)
# mesh2.compute_vertex_normals()
# o3d.visualization.draw([mesh1,mesh2])