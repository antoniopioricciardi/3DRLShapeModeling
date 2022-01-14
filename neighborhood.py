import numpy as np


def neighborhood_from_index(num_points, num_neighbors_per_side):
    neigh_dict = dict()
    point_list = list(range(num_points))
    for i in range(num_points):
        lower_bound = i - num_neighbors_per_side
        upper_bound = i + num_neighbors_per_side
        if lower_bound < 0:
            #TODO: pop(idx of i) instead of remove(i) should be faster
            temp_list = point_list[lower_bound:] + point_list[:upper_bound+1]
            temp_list.remove(i)
            neigh_dict[i] = temp_list  # simple lists concatenate with "+"
        elif upper_bound > num_points-1:
            temp_list = point_list[lower_bound:] + point_list[:(upper_bound+1)%num_points]
            temp_list.remove(i)
            neigh_dict[i] = temp_list
        else:
            neigh_dict[i] = point_list[lower_bound:i] + point_list[i+1:upper_bound+1]  # +1 because of slicing rules, we do not want "i" in our list and want to take upper_bound+1
    return neigh_dict


def no_neighbors():
    # TODO: complete (or not?) this function
    return None

# def neighborhood_from_radius(pts_x, pts_y, radius):

def triangle_neighborhood(triangles, triangle_idx, point_idx):
    """
    Returns a mask for selecting a 3-vertex coordinates from c_x,c_y,c_z and t_x,t_y,t_z
    :param triangles:
    :param triangle_idx:
    :param point_idx:
    :return:
    """
    triangle_vertices = triangles[triangle_idx]  # this gives three vertex-indices
    tri_vert_list = list(triangle_vertices)
    # set the object vertex in first position (by removing it from its previous position)
    el = tri_vert_list.pop(point_idx)
    vertex_mask = np.array([el] + tri_vert_list)
    return vertex_mask


def sorted_distance_neighborhood(distance_matrix):
    sorting_idx = np.argsort(distance_matrix)
    sorted_matrix = np.sort(distance_matrix)
    return sorted_matrix, sorting_idx


def get_neighborhood_mask(num_points, central_point):
    neighborhood = np.arange(num_points)
    neighborhood = np.delete(neighborhood, central_point)
    return neighborhood
