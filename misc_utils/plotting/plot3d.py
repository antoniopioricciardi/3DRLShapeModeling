import io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def plot3(fig):
    with io.BytesIO() as buff:
        fig.savefig(buff, format='raw')
        buff.seek(0)
        data = np.frombuffer(buff.getvalue(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    im = data.reshape((int(h), int(w), -1))
    return im


def plot_point_cloud(coordinates, fig=None, spreads=None, marker_sizes=None, colors=None):  # arguments are triplets of xyz coords
    """
    :param coordinates: a list of triples [(x0,y0,z0), (x1,y1,z1)...] one for each plot we want to show
    :param fig:
    :param spread:
    :param marker_sizes:
    :param colors:
    :return:
    """
    if spreads is not None:
        assert len(coordinates) == len(spreads)
    if marker_sizes is not None:
        assert len(coordinates) == len(marker_sizes)
    if colors is not None:
        assert len(coordinates) == len(colors)

    colors = list(mcolors.BASE_COLORS) if colors is None else colors  # return keys of {'b': (0, 0, 1), 'g': (0, 0.5, 0), 'r': (1, 0, 0), 'c': (0, 0.75, 0.75), 'm': (0.75, 0, 0.75), 'y': (0.75, 0.75, 0), 'k': (0, 0, 0), 'w': (1, 1, 1)}
    # these give many more colors
    # mcolors.TABLEAU_COLORS
    # mcolors.CSS4_COLORS [13:]  0 is black, up to 13 it becomes white, then colors start
    # TODO: options to show, save to file etc.
    new_fig = True if fig is None else False
    if new_fig:
        fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # ax.axis('off')

    marker_sizes = marker_sizes if (marker_sizes is not None) else [10]*len(coordinates)
    for idx, coord in enumerate(coordinates):
        x, y, z = coord
        marker = marker_sizes[idx]
        ax.scatter(x, y, z, s=marker, c=colors[idx])
        print(spreads)
        if spreads is not None:
            spread = spreads[idx]
            if spread is not None:
                ax.set_xlim([-spread - 1, spread + 1])
                ax.set_ylim([-spread - 1, spread + 1])
                ax.set_zlim([-spread - 1, spread + 1])

    plt.show()  # used to show the agent progressing
    # plt.pause(0.001)
    # plt.ioff()
    # image = plot3(fig)
    # plt.savefig("mygraph.png")

    ''' If we created a new fig, we can close it'''
    if new_fig:
        plt.close()





def plot_point_cloud_old(*arguments, fig=None, spread=0, marker_size=0, colors=None):  # arguments are triplets of xyz coords
    # TODO: dizionario con tutti gli argomenti necessari per OGNI plot (vertici, colore, dimensione ecc) così più facile da gestire
    print(colors)
    colors = list(mcolors.BASE_COLORS) if colors is None else colors  # return keys of {'b': (0, 0, 1), 'g': (0, 0.5, 0), 'r': (1, 0, 0), 'c': (0, 0.75, 0.75), 'm': (0.75, 0, 0.75), 'y': (0.75, 0.75, 0), 'k': (0, 0, 0), 'w': (1, 1, 1)}

    # these give much more colors
    # mcolors.TABLEAU_COLORS
    # mcolors.CSS4_COLORS [13:]  0 is black, up to 13 it becomes white, then colors start
    # TODO: options to show, save to file etc.
    new_fig = True if fig is None else False
    if new_fig:
        fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.axis('off')

    marker_size = marker_size if (marker_size != 0) else 10

    for idx, coords in enumerate(arguments):
        x, y, z = coords
        num_vertices = len(x)
        marker = [marker_size] * num_vertices
        print(colors)
        ax.scatter(x, y, z, s=marker, color=colors[idx])
    if spread != 0:
        ax.set_xlim([-spread - 1, spread + 1])
        ax.set_ylim([-spread - 1, spread + 1])
        ax.set_zlim([-spread - 1, spread + 1])

    plt.show()  # used to show the agent progressing
    # plt.pause(0.001)
    # plt.ioff()
    # image = plot3(fig)
    ''' If we created a new fig, close it'''
    if new_fig:
        plt.close()


def prova(*arguments, altro=False):  # need to be explicit to use "altro". Good.
    print(arguments)
    print(len(arguments))
    if altro:
        print('yep')
    return

# prova('ciao', 'come', 'stai', altro=True)
#

# from read_shape import load_mesh, generate_new_mesh
# from inits import rand_init3d, sphere, from_shape
# import open3d
#
# # x,y,z = sphere(10)
# # _,_,_, x, y, z,tri = from_shape(shape_path='../../shapes/25.ply')
# x, y, z,_,_,_,tri = from_shape(shape_path='../../shapes/25.ply')
# sx,sy,sz, _ = rand_init3d()
# vert_idx = 128
# vertex_idx = tri[vert_idx][0]
# # neighborhood = set()
# # neighborhood.add(tri[vert_idx][1])
# # neighborhood.add(tri[vert_idx][2])
#
# fig = plt.figure()
# plot_point_cloud_new([(x,y,z), (sx,sy,sz)], fig=fig, colors=['blue','red'])



# plot_point_cloud((x,y,z), fig=fig, colors='blue')
# plot_point_cloud(([x[neighborhood]], [y[neighborhood]], [z[neighborhood]]), fig=fig, marker_size=100, colors='red')

# spread = 5
# # num_vertices = len(x)
# # marker = [10] * num_vertices
# # ax = fig.add_subplot(111, projection='3d')
# # ax.axis('off')
# # ax.scatter(x, y, z, s=marker, color='blue')
# # marker = [60] * len(neighborhood)
# # ax.scatter(x[neighborhood], y[neighborhood], z[neighborhood], s=marker, color='red')
# # ax.set_xlim([-spread - 1, spread + 1])
# # ax.set_ylim([-spread - 1, spread + 1])
# # ax.set_zlim([-spread - 1, spread + 1])
# # plt.show()
# #
# # plt.close()
#
# m = generate_new_mesh_from_file(np.vstack((x, y, z)).T, source_shape_path='../../env_vertices/shapes/25.ply')
# m.compute_vertex_normals()
# v = np.asarray(m.vertices)
# f = np.asarray(m.triangles)
# n = np.asarray(m.vertex_normals)
#
# smoothness = 0.8       # 0 < smoothness < 1  higher values mean smoother bump
# quantity = 0.3  # higher values mean higher bump
# m.compute_adjacency_list()
#
# adj_list = m.adjacency_list  # contains, at index i, the list of triangles neighbors to triangle i
#
# adj_matrix = np.zeros((len(v), len(v)), dtype=int)
# i = 0
# for el in adj_list:
#     adlist = np.array(list(el))
#     adj_matrix[i][adlist] = 1
#     i += 1
#
#
# D = compute_distance_matrix(v, adj_matrix)
#
# vert_idx = 128
# dist_row = D[vert_idx]
# # dist = dist_row[dist_row < 1.5]# [dist_row < 1.5]
# sorted_dist = np.sort(dist_row)
# sorted_idx = np.argsort(dist_row)
#
# selection = sorted_idx[:10]
#
#
# # Original by Rick, if we want to work with the entire shape
# # funz = quantity * np.exp(-(1/smoothness**4) * (D[vert_idx,:])**2)
# # v_new = v + np.tile(funz,(3,1)).T * n
# if False:
#     funz = quantity * np.exp(-(1/smoothness**4) * (dist)**2)
#     print(funz.shape)
#     print(np.tile(funz,(3,1)).shape, np.tile(funz,(3,1)).T.shape)
#     v_new = v
#     v_new[dist_row < 1.5] += np.tile(funz,(3,1)).T * np.array([-1,0,-1])# (n[dist_row<1] + np.array([1,-1,-1]))
#
#     print(v.shape)
#     print(v[:,0].shape)
#     new_x, new_y, new_z = gaussian_bump3d(v_new[:,0], v_new[:,1], v_new[:,2], -1, 0, -1, dist, 1, 1, smoothness)
#     v_new = v
#     v_new[0] += new_x
#     v_new[1] += new_y
#     v_new[2] += new_z
#
# print(v.shape)
# print(v[:,0].shape)
# new_x, new_y, new_z = gaussian_bump3d(v[:,0], v[:,1], v[:,2], -0.5, 0, -0.3, dist_row, 5, 1, smoothness)
# # v_new = v
# v_new = np.dstack((new_x, new_y, new_z))  # put every vector in a column, so if x y z are all n dim vectors, we will have a nx3 matrix
# v_new = v_new.squeeze()  # remove first dimension from (1,n,3)
# print(v_new.shape)
# print('mammit')
# v[selection] = v_new[selection]
# print(v_new.shape)
# m2 = o3d.geometry.TriangleMesh(vertices = o3d.utility.Vector3dVector(v), triangles = o3d.utility.Vector3iVector(f))
# m2.compute_vertex_normals()
# o3d.visualization.draw([m2])
#
#
#
#
#
# exit(55)
#
#
# print(vert_adj)
# neighborhood = set(vert_adj)
# print(neighborhood)
# level_neighborhood = neighborhood.copy()
# new_level_neighborhood = []
# # for vert in level_neighborhood:
# #     adjacencies = adjacency_list[vert]
# #     for adj in adjacencies:
# #         neighborhood.add(adj)
#     # new_level_neighborhood.append(adjacency_list[vert]) check
# neighborhood = np.array(list(neighborhood))
# # neighborhood.add(vert_idx)
#
# # plot_point_cloud((x,y,z), ([x[neighborhood]], [y[neighborhood]], [z[neighborhood]]), spread=spread)
# num_vertices = len(x)
# marker = [10] * num_vertices
# ax1 = fig.add_subplot(211, projection='3d')
# ax1.axis('off')
# ax1.scatter(x, y, z, s=marker, color='blue')
# # neighborhood = np.zeros(len(vert_adj)+1, dtype=int)
# # neighborhood[0] = vert_idx
# # neighborhood[1:] = np.array(list(vert_adj))
# print(neighborhood)
# marker = [60] * len(neighborhood)
#
# ax1.scatter(x[neighborhood], y[neighborhood], z[neighborhood], s=marker, color='red')
# ax1.scatter(x[vert_idx], y[vert_idx], z[vert_idx], s=60, color='green')
#
# ax1.set_xlim([-spread - 1, spread + 1])
# ax1.set_ylim([-spread - 1, spread + 1])
# ax1.set_zlim([-spread - 1, spread + 1])
#
# marker = [10] * num_vertices
#
# mu, sigma = 0.,0.3
# x_neigh, y_neigh, z_neigh = move_points_gaussian3d(0, x[neighborhood],y[neighborhood],z[neighborhood], x_movement=-0.5, y_movement=-0.5, z_movement=-0.5, max_spread=5, mu=mu, sigma=sigma)
# x[neighborhood] = x_neigh
# y[neighborhood] = y_neigh
# z[neighborhood] = z_neigh
#
# ax2 = fig.add_subplot(212, projection='3d')
# ax2.axis('off')
# ax2.scatter(x, y, z, s=marker, color='blue')
# print(neighborhood)
# # TODO: set entries might not be sorted, this is an issue (in fact vert_idx does not return the correct element)
# ax2.scatter(x[neighborhood], y[neighborhood], z[neighborhood], s=[60] * len(neighborhood), color='red')
# # ax2.scatter(x[neighborhood[0]], y[neighborhood[0]], z[neighborhood[0]], s=60, color='green')
# ax2.scatter(x[vert_idx], y[vert_idx], z[vert_idx], s=60, color='green')
#
# ax2.set_xlim([-spread - 1, spread + 1])
# ax2.set_ylim([-spread - 1, spread + 1])
# ax2.set_zlim([-spread - 1, spread + 1])
#
# plt.show()
#
# plt.close()
#
# exit(5)
#
# marker = [10] * num_vertices
# # TODO: FIX NEIGHBORHOOD
# mu, sigma = 0.,0.5
# x_neigh, y_neigh, z_neigh = move_points_gaussian3d(0, x[neighborhood],y[neighborhood],z[neighborhood], x_movement=1, y_movement=1, z_movement=1, max_spread=5, mu=mu, sigma=sigma)
# x[neighborhood] = x_neigh
# y[neighborhood] = y_neigh
# z[neighborhood] = z_neigh
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.axis('off')
# ax.scatter(x, y, z, s=marker, color='blue')
# ax.set_xlim([-spread - 1, spread + 1])
# ax.set_ylim([-spread - 1, spread + 1])
# ax.set_zlim([-spread - 1, spread + 1])
# plt.show()
# plt.close()
# # TODO: controlla slides di CG su vert adj