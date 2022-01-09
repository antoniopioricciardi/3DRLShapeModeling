from misc_utils.shapes.io import *
from sklearn.neighbors import NearestNeighbors
import numpy as np
from misc_utils.shapes.io import *
from io import *
from misc_utils.plotting.plot3d import *


def get_vertices_mapping(v_rem, v):
    """
    Get a mapping from v_rem to v
    Returns an index for each element of v
    """
    # indices specifies the neighbor to each index i. E.g. indices[0] returns the index j of the closest neighbor to the element 0.
    # when n_neighbors=1
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(v_rem)
    distances, indices = nbrs.kneighbors(v)
    return distances, indices




""" QUICK CODE TO LOAD SHAPES, FIND MAPPING AND SAVING REMESHED SHAPES (Uncomment save_mesh if you want to save to file)"""
if False:
    sphere_rem = load_mesh('../../shapes/sphere_rem.ply')
    # these two have the same triangulation, otherwise they wouldn't have a correspondence
    sphere = load_mesh('../../shapes/sphere.ply')
    human = load_mesh('../../shapes/tr_reg_000.ply')
    v_rem = np.array(sphere_rem.vertices)
    f_rem = np.array(sphere_rem.triangles)
    v = np.array(sphere.vertices)
    f = np.array(sphere.triangles)

    # distances[i] -> j contains the distance of j from i.
    distances, indices = get_vertices_mapping(v, v_rem)
    # print(len(v_rem))  # 1002
    # print(len(indices))  # 6890
    indices = indices.squeeze()

    # CODE TO FIND mapping of points from original shapes to the remeshed one
    # distances, indices = get_vertices_mapping(v_rem, v)
    #
    # landmarks = np.array([412, 2445, 3219, 6617, 5907])
    # print(indices[landmarks].squeeze())
    # v_new = v[indices]
    # print(len(v_new))
    # mesh = generate_new_mesh(v_new, f_rem)  # need to use "original" remeshed triangulations, not the mapped ones!
    # mesh.compute_vertex_normals()
    # o3d.visualization.draw([mesh])

    human_shape = load_mesh('../../shapes/tr_reg_000.ply')
    hum_v = np.array(human_shape.vertices)
    hum_v_new = hum_v[indices]
    mesh = generate_new_mesh(hum_v_new, f_rem)  # need to use "original" remeshed triangulations, not the mapped ones!
    mesh.compute_vertex_normals()
    o3d.visualization.draw([mesh])
    # plot_point_cloud((x,y,z), ([x[neighborhood]], [y[neighborhood]], [z[neighborhood]]), spread=spread)
    landmarks = np.array([538, 255, 182, 713, 615])
    landmark_v = hum_v_new[landmarks]
    plot_point_cloud((hum_v_new[:,0], hum_v_new[:,1], hum_v_new[:,2]), (landmark_v[:,0], landmark_v[:,1], landmark_v[:,2]),
                     (v_rem[:,0], v_rem[:,1], v_rem[:,2]))

    # save_mesh(mesh, '../../shapes/tr_reg_000_rem.ply')
    print(len(hum_v_new))



