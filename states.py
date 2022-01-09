import numpy as np 


def signed_dist(c_x, c_y, s_x, s_y):
    state_x = s_x - c_x
    state_y = s_y - c_y
    state = np.concatenate((state_x, state_y))
    return state


def signed_dist_3D(c_x, c_y, c_z, s_x, s_y, s_z):
    state_x = s_x - c_x
    state_y = s_y - c_y
    state_z = s_z - c_z
    state = np.concatenate((state_x, state_y, state_z))
    return state


def signed_dist_3D_and_dist_from_center(c_x, c_y, c_z, s_x, s_y, s_z):
    state_x = s_x - c_x
    state_y = s_y - c_y
    state_z = s_z - c_z
    center_x = c_x[0]
    center_y = c_y[0]
    center_z = c_z[0]
    distance_vector = np.sqrt((center_x-c_x)**2 + (center_y - c_y)**2 + (center_z - c_z)**2)
    state = np.concatenate((state_x, state_y, state_z, distance_vector))
    return state


def single_point_signed_dist(s_x, s_y, c_x, c_y, point_idx):
    state_x = s_x - c_x
    state_y = s_y - c_y
    # state = np.concatenate((state_x[point_idx], state_y[point_idx]))
    state = np.array([state_x[point_idx], state_y[point_idx]])  # create an array with two scalars
    return state


def neighborhood_signed_dist(s_x, s_y, c_x, c_y, pt_neighs):
    # TODO: check this works
    state_x = s_x - c_x
    state_y = s_y - c_y
    state = (np.concatenate((state_x[(pt_neighs)], state_y[(pt_neighs)]))).reshape(-1)  # concatenate two vectors
    return state
