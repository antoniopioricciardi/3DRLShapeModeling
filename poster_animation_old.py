import os
import pickle
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from open3d.cpu.pybind.utility import Vector3dVector, Vector3iVector
import time
from read_shape import load_mesh, generate_new_mesh
from open3d.visualization import Visualizer

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


shape_name= 'tr_reg_000_rem.ply'# '25.ply' # 'tr_reg_007'
transition_path = os.path.join(RESULTS_PATH, shape_name)
shape_path = os.path.join('shapes', shape_name)
print(shape_name)
# open a file, where you stored the pickled data
file = open(os.path.join(transition_path,'transitions'), 'rb')

# read information from file
transitions = pickle.load(file)
file.close()

transition_matrix = transitions['transition_matrix']
masks_transitions = transitions['vertex_masks']
config_33 = transitions['33_config']
config_66 = transitions['66_config']
final_configuration = transitions['final_configuration']
print('mamm')

if os.path.exists(os.path.join(transition_path, 'shape_pickle')):
    file = open(os.path.join(transition_path, 'shape_pickle'), 'rb')
    shape_values = pickle.load(file)
    file.close()
else:
    ''' serialize shape to avoid reloading later '''
    shape_values = from_shape(shape_path=shape_path)
    file = open(os.path.join(transition_path, 'shape_pickle'), 'wb')
    pickle.dump(shape_values, file)
    file.close()

s_x, s_y, s_z, c_x, c_y, c_z, triangles = shape_values  # from_shape()
Mesh = generate_new_mesh(np.vstack((s_x, s_y, s_z)).T, source_shape_path=shape_path)
Mesh.compute_vertex_normals()
o3d.visualization.draw([Mesh])

Mesh_init = generate_new_mesh(np.vstack((c_x, c_y, c_z)).T, source_shape_path=shape_path)
Mesh_init.compute_vertex_normals()
o3d.visualization.draw([Mesh_init])


mesh2 = o3d.geometry.TriangleMesh()
mesh2.vertices = Vector3dVector(np.vstack((c_x, c_y, c_z)).T)
mesh2.triangles = Vector3iVector(triangles)
mesh2.compute_vertex_normals()
vis = Visualizer()
vis.create_window()
vis.add_geometry(mesh2)
print('plotting!')
for i in range(len(transition_matrix)):
    new_c_x, new_c_y, new_c_z = transition_matrix[i]
    mask = np.array(masks_transitions[i][1]).astype(int)
    c_x[mask] = new_c_x
    c_y[mask] = new_c_y
    c_z[mask] = new_c_z
    # vert = np.asarray(mesh2.vertices)
    # print(vert[mask])
    if i % 200 == 0:
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