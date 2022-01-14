import igl
import numpy as np
import open3d as o3d
from scipy.spatial.distance import cdist
from scipy.sparse import csc_matrix
from scipy import sparse
# import igl
from scipy.sparse.csgraph import dijkstra
from scipy.io import loadmat
from scipy.sparse.linalg import spsolve
import scipy as sp
from ..normalization import normalize_vector

def compute_distance_matrix(v, a):
    """
    given a list of vertices, a list of triangles and an adjacency matrix, compute and return the distance matrix d
    for the vertices, where in d[i,j] there is the distance between vertex i and vertex j
    a: an adjacency matrix where a[i,j]=1 if i is adjacent to j
    """
    # a = igl.adjacency_matrix(f)
    dist = cdist(v, v)
    values = dist[np.nonzero(a)]
    matrix = sparse.coo_matrix((values, np.nonzero(a)), shape=(v.shape[0], v.shape[0]))
    d = dijkstra(matrix, directed=False)
    return d


def compute_adjacency_matrix(adj_list):
    adj_matrix = np.zeros((len(adj_list), len(adj_list)))
    i = 0
    for el in adj_list:
        adlist = np.array(list(el)).astype(int)
        adj_matrix[i][adlist] = 1
        i += 1
    return adj_matrix


def compute_triangle_triangle_adjacency_matrix_igl(f):
    adj_tri, adj_edge = igl.triangle_triangle_adjacency(f)
    return adj_tri, adj_edge


def compute_adjacency_matrix_igl(f):
    a = igl.adjacency_matrix(f)
    return a


def mean_curvature_flow(v, f, num_iterations=10):
    """
    v: vertices
    f: triangles
    """
    l = igl.cotmatrix(v, f)

    n = igl.per_vertex_normals(v, f) * 0.5 + 0.5
    c = np.linalg.norm(n, axis=1)

    vs = [v]
    cs = [c]
    for i in range(num_iterations):
        print(f'{i}/{num_iterations}')
        m = igl.massmatrix(v, f, igl.MASSMATRIX_TYPE_BARYCENTRIC)

        s = (m - 0.001 * l)
        b = m.dot(v)
        v = spsolve(s, m.dot(v))
        n = igl.per_vertex_normals(v, f) * 0.5 + 0.5
        c = np.linalg.norm(n, axis=1)
        vs.append(v)
        cs.append(c)
    return v


def normalize_mesh(mesh, spread=1):
    v = np.array(mesh.vertices)
    min_val = v.min(axis=0).min()
    max_val = v.max(axis=0).max()
    x = normalize_vector(v[:, 0], min_val, max_val) * spread
    y = normalize_vector(v[:, 1], min_val, max_val) * spread
    z = normalize_vector(v[:, 2], min_val, max_val) * spread

    v = np.vstack((x, y, z)).T
    mesh = o3d.geometry.TriangleMesh(vertices=o3d.utility.Vector3dVector(v),
                                         triangles=o3d.utility.Vector3iVector(mesh.triangles))
    return mesh
