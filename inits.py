import math  
import numpy as np

from read_img import retrieve_image_points
from read_shape import load_mesh, create_sphere_from_mesh, load_mesh_new
from sorted_contour import retrieve_sorted_img
from misc_utils.shapes.operations import normalize_mesh

def zero_init(num_points=4):
    x = np.zeros(num_points)
    y = np.zeros(num_points)
    return x, y


def rand_init2d(num_points=4, center_x=0, center_y=0, span_x=0.5, span_y=0.5, spread=5):
    x = np.zeros(num_points) + center_x + np.random.uniform(-span_x,span_x, num_points)
    y = np.zeros(num_points) + center_y + np.random.uniform(-span_y,span_y, num_points)
    return x*spread, y*spread


def rand_init3d(num_points=3, center_x=0, center_y=0, center_z=0, span_x=1, span_y=1, span_z=1, spread=5):
    x = np.zeros(num_points) + center_x + np.random.uniform(-span_x,span_x, num_points)
    y = np.zeros(num_points) + center_y + np.random.uniform(-span_y,span_y, num_points)
    z = np.zeros(num_points) + center_z + np.random.uniform(-span_z,span_z, num_points)
    tri = np.arange(num_points)
    tri = tri.reshape(1, len(tri))
    return x*spread, y*spread, z*spread


def sphere(num_points=50, center_x=0, center_y=0, center_z=0, span_x=0.5, span_y=0.5, span_z=0.5, spread=5):
    #TODO: just use open3d code
    '''
    (https://stackoverflow.com/questions/61048426/python-generating-3d-sphere-in-numpy)
        create sphere with center (cx, cy, cz) and radius r
    '''
    phi = np.linspace(0, 2 * np.pi, 2 * num_points)
    theta = np.linspace(0, np.pi, num_points)

    theta, phi = np.meshgrid(theta, phi)

    r = spread
    r_xy = r * np.sin(theta)
    x = center_x + np.cos(phi) * r_xy
    y = center_y + np.sin(phi) * r_xy
    z = center_z + r * np.cos(theta)

    return np.stack([x, y, z])


def circle(num_points=4, spread=5):
    s_x = np.zeros(num_points)
    s_y = np.zeros(num_points)
    for i in range(num_points):
        angle = (i / num_points) * (math.pi * 2)
        s_x[i] = round(math.sin(angle), 2) * spread
        s_y[i] = round(math.cos(angle), 2) * spread
    return s_x, s_y


def from_img(num_points=10, spread=5, img_path='./images/seal.png'):
    # s_x, s_y = retrieve_image_points(img_path=img_path, num_points=num_points)
    s_x, s_y = retrieve_sorted_img(img_path=img_path, num_points=num_points, rotate=True)
    return s_x * spread, s_y * spread


def from_shape(spread=5, num_triangles=0, shape_path='./shapes/25.ply', simplify=False, normalize=False):
    # TODO: if pickle version exists, do not load again
    Mesh, V, F = load_mesh_new(shape_path, simplify)
    if normalize:
        Mesh = normalize_mesh(Mesh)
        V = np.array(Mesh.vertices)
    # target_vert, canvas_vert, tri_list = load_mesh(shape_path, simplify)
    x = V[:,0]
    y = V[:,1]
    z = V[:,2]
    return x * spread, y * spread, z * spread, F


def load_shape_old(spread=5, num_triangles=0, shape_path='./shapes/25.ply'):
    # TODO: if pickle version exists, do not load again
    target_vert, canvas_vert, tri_list = load_mesh(shape_path)
    t_x = target_vert[:,0]
    t_y = target_vert[:,1]
    t_z = target_vert[:,2]
    c_x = canvas_vert[:,0]
    c_y = canvas_vert[:,1]
    c_z = canvas_vert[:,2]
    return t_x * spread, t_y * spread, t_z * spread, c_x * spread, c_y * spread, c_z * spread, tri_list


def load_shape(shape_path='./shapes/25.ply', simplify=False, normalize=False):
    mesh, V, F = load_mesh_new(shape_path, simplify, normalize)
    V = np.array(mesh.vertices)
    #adj_matrix, dist_matrix
    return mesh, V, F # , adjacency


def sphere_from_mesh(V, F, spread=5):
    mesh, canvas_vert, canvas_tri = create_sphere_from_mesh(V, F)
    return mesh, canvas_vert, canvas_tri


def get_coordinates(vert, spread):
    x = vert[:,0]
    y = vert[:,1]
    z = vert[:,2]
    return x * spread, y * spread, z * spread


from misc_utils.shapes.operations import compute_adjacency_matrix_igl, compute_triangle_triangle_adjacency_matrix_igl
target_mesh, target_vert, target_tri = load_shape('shapes/tr_reg_000_rem.ply', simplify=False,
                                                       normalize=True)

def vertex_mask_from_triangle_adjacency(tri_idx, adj_tri, n_hops, n_required_vertices):
    adj_set = set()
    # adj_set.add(tri_idx)  # should order items increasingly
    adj_dict = dict()
    hop_list = [tri_idx]
    adj_dict[0] = hop_list

    for hop in range(n_hops):
        hop_list = []
        for adj in adj_dict[hop]:
            hop_list = list(adj_tri[adj])
            if not adj_dict.get(hop+1):
                adj_dict[hop+1] = []
            adj_dict[hop+1] += hop_list

    for lst in adj_dict.values():
        for el in lst:
            adj_set.add(el)

    vertex_mask = target_tri[np.array(list(set(adj_set)))]
    vertex_mask = list(set(np.reshape(vertex_mask, vertex_mask.shape[0] * vertex_mask.shape[1])))
    while len(vertex_mask) < n_required_vertices:
        vertex_mask.append(vertex_mask[-1])
    vertex_mask = np.array(vertex_mask)
    return vertex_mask

a = compute_adjacency_matrix_igl(target_tri)

adj_tri, adj_edge = compute_triangle_triangle_adjacency_matrix_igl(target_tri)
n_hops = 1
tri_idx = 0
for i in range(len(target_tri)):
    lungh = vertex_mask_from_triangle_adjacency(i, adj_tri, 2, 12)
    print(lungh)