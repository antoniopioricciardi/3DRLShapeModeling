import numpy as np
from misc_utils.normalization import normalize_vector
from misc_utils.math.distributions import *


def normal_distribution(x,mu,sigma):
    return ( 2.*np.pi*sigma**2.)**-.5 * np.exp( -.5 * (x-mu)**2. / sigma**2. )


def find_neighbors(num_points, num_neigh_points):
    neigh_dict = dict()
    for i in range(num_points):
        neighs = []
        if i == 0:
            neighs.append(num_points-1)
        if i > 0:
            neighs.append(i - 1)
        # else append(-1)  # -1 come "valore non trovato"?
        if i < num_points - 1:
            neighs.append(i + 1)
        if i == num_points - 1:
            neighs.append(0)
        neigh_dict[i] = neighs
    return neigh_dict


def move_neighbors(point_idx, neighbors, point_x_movement, point_y_movement, canvas_x_coords, canvas_y_coords, pts_max_spread):
    neighbors_idx = neighbors[int(point_idx)]

    return move_point((neighbors_idx), point_x_movement, point_y_movement, canvas_x_coords, canvas_y_coords, pts_max_spread)


def move_neighbors3d(point_idx, neighbors, point_x_movement, point_y_movement, point_z_movement, canvas_x_coords,
                     canvas_y_coords, canvas_z_coords, pts_max_spread):
    neighbors_idx = neighbors

    return move_point3d(neighbors_idx, point_x_movement, point_y_movement, point_z_movement, canvas_x_coords,
                        canvas_y_coords, canvas_z_coords, pts_max_spread)


def move_point(point_idx, x_movement, y_movement, canvas_x_coords, canvas_y_coords, pts_max_spread):
    # move points according to agent's output
    canvas_x_coords[point_idx] += x_movement
    canvas_y_coords[point_idx] += y_movement

    # check the points do not exceed the canvas bounds
    canvas_x_coords[canvas_x_coords > pts_max_spread] = pts_max_spread
    canvas_x_coords[canvas_x_coords < -pts_max_spread] = -pts_max_spread
    canvas_y_coords[canvas_y_coords > pts_max_spread] = pts_max_spread
    canvas_y_coords[canvas_y_coords < -pts_max_spread] = -pts_max_spread
    return canvas_x_coords, canvas_y_coords


def check_bounds2d(x, y, max_spread):
    # TODO: 2d and 3d could be joined
    x[x > max_spread] = max_spread
    x[x < -max_spread] = -max_spread
    y[y > max_spread] = max_spread
    y[y < -max_spread] = -max_spread
    return x, y


def check_bounds3d(x, y, z, max_spread):
    x[x > max_spread] = max_spread
    x[x < -max_spread] = -max_spread
    y[y > max_spread] = max_spread
    y[y < -max_spread] = -max_spread
    z[z > max_spread] = max_spread
    z[z < -max_spread] = -max_spread
    return x, y, z


def move_point3d(point_idx, x_movement, y_movement, z_movement, canvas_x_coords, canvas_y_coords, canvas_z_coords, pts_max_spread):
    # move points according to agent's output
    canvas_x_coords[point_idx] += x_movement
    canvas_y_coords[point_idx] += y_movement
    canvas_z_coords[point_idx] += z_movement

    return check_bounds3d(canvas_x_coords, canvas_y_coords, canvas_z_coords, pts_max_spread)
    # # check the points do not exceed the canvas bounds
    # canvas_x_coords[canvas_x_coords > pts_max_spread] = pts_max_spread
    # canvas_x_coords[canvas_x_coords < -pts_max_spread] = -pts_max_spread
    # canvas_y_coords[canvas_y_coords > pts_max_spread] = pts_max_spread
    # canvas_y_coords[canvas_y_coords < -pts_max_spread] = -pts_max_spread
    # canvas_z_coords[canvas_z_coords > pts_max_spread] = pts_max_spread
    # canvas_z_coords[canvas_z_coords < -pts_max_spread] = -pts_max_spread
    #
    # return canvas_x_coords, canvas_y_coords, canvas_z_coords


def move_points_gaussian2d(center_idx, x, y, x_movement, y_movement, max_spread, mu, sigma):
    """
    move a set of points according to a gaussian distribution computed over the distance values of vertices
    from a given point.
    It is better not to give the entire shape as input, this might be slow
    """

    xy = np.vstack((x, y)).T  # create coordinate pairs
    center = xy[center_idx]
    # dist = np.linalg.norm(center-xy, axis=1, ord=1)
    distances = (center[0] - x) + (center[1] - y)

    distribution = normal(distances, mu, sigma)
    distribution = normalize_vector(distribution)  # normalize distr to be in [0,1]

    # scale distribution by the movement factor
    x_movement *= distribution
    y_movement *= distribution

    x += x_movement
    y += y_movement

    x, y = check_bounds2d(x, y, max_spread)
    return x, y

import matplotlib.pyplot as plt


def move_points_gaussian3d(center_idx, x, y, z, x_movement, y_movement, z_movement, max_spread, mu, sigma):
    """
    move a set of points according to a gaussian distribution computed over the distance values of vertices
    from a given point.
    It is better not to give the entire shape as input, this might be slow
    center_id: the center of the gaussian curve, the point that the agent wants to move
    x: x coordinates, numpy array
    y: y coordinates, numpy array
    z: z coordinates, numpy array
    x_movement: the maximum amount of movement to perform along x coordinates (where the gaussian is at its max value)
    y_movement: the maximum amount of movement to perform along y coordinates (where the gaussian is at its max value)
    z_movement: the maximum amount of movement to perform along z coordinates (where the gaussian is at its max value)
    """

    xyz = np.vstack((x, y, z)).T  # create coordinate pairs
    center = xyz[center_idx]
    # dist = np.linalg.norm(center-xy, axis=1, ord=1)
    # TODO: We need positive distances, i think.
    distances = (center[0] - x) + (center[1] - y) + (center[2] - z)
    print(distances)
    distribution = normal(distances, mu, sigma)
    distribution = normalize_vector(distribution)  # normalize distr to be in [0,1]

    # fig = plt.figure()
    # plt.scatter(distances, distribution, s=[80] * len(x), c='blue')
    # plt.show()
    # plt.close()

    # scale distribution by the movement factor
    x_movement *= distribution
    y_movement *= distribution
    z_movement *= distribution

    x += x_movement
    y += y_movement
    z += z_movement
    # x, y, z = check_bounds3d(x, y, z, max_spread)

    return check_bounds3d(x, y, z, max_spread)


def move_points_gaussian3dalt(center_idx, x, y, z, x_movement, y_movement, z_movement, max_spread, mu, sigma):
    """
    move a set of points according to a gaussian distribution computed over the distance values of vertices
    from a given point.
    It is better not to give the entire shape as input, this might be slow
    center_id: the center of the gaussian curve, the point that the agent wants to move
    x: x coordinates, numpy array
    y: y coordinates, numpy array
    z: z coordinates, numpy array
    x_movement: the maximum amount of movement to perform along x coordinates (where the gaussian is at its max value)
    y_movement: the maximum amount of movement to perform along y coordinates (where the gaussian is at its max value)
    z_movement: the maximum amount of movement to perform along z coordinates (where the gaussian is at its max value)
    """

    xyz = np.vstack((x, y, z)).T  # create coordinate pairs
    center = xyz[center_idx]
    # dist = np.linalg.norm(center-xy, axis=1, ord=1)
    # TODO: We need positive distances, i think.
    distances_x = (center[0] - x)
    distances_y = (center[1] - y)
    distances_z = (center[2] - z)
    distribution_x = normal(distances_x, mu, sigma)
    distribution_x = normalize_vector(distribution_x)  # normalize distr to be in [0,1]
    distribution_y = normal(distances_y, mu, sigma)
    distribution_y = normalize_vector(distribution_y)  # normalize distr to be in [0,1]
    distribution_z = normal(distances_z, mu, sigma)
    distribution_z = normalize_vector(distribution_z)  # normalize distr to be in [0,1]

    # fig = plt.figure()
    # plt.scatter(distances_x, distribution_x, s=[80] * len(x), c='blue')
    # plt.show()
    # plt.close()

    # scale distribution by the movement factor
    x_movement *= distribution_x
    y_movement *= distribution_y
    z_movement *= distribution_z

    x += x_movement
    y += y_movement
    z += z_movement
    # x, y, z = check_bounds3d(x, y, z, max_spread)

    return check_bounds3d(x, y, z, max_spread)


def gaussian_bump3d(x, y, z, x_movement, y_movement, z_movement, distance_vector, max_spread, mu, sigma):
    """
    Create a 3d bump that follows a gaussian distribution, based on the order provided by distance_vector
    :param x:
    :param y:
    :param z:
    :param x_movement:
    :param y_movement:
    :param z_movement:
    :param distance_vector:
    :param max_spread:
    :param mu:
    :param sigma:
    :return:
    """
    funz = mu * np.exp(-(1 / sigma ** 4) * (distance_vector) ** 2)
    x += funz * x_movement
    y += funz * y_movement
    z += funz * z_movement
    return check_bounds3d(x, y, z, max_spread)


def gaussian_bump2d(x, y, x_movement, y_movement, distance_vector, max_spread, mu, sigma):
    """
    Create a 3d bump that follows a gaussian distribution, based on the order provided by distance_vector
    :param x:
    :param y:
    :param z:
    :param x_movement:
    :param y_movement:
    :param z_movement:
    :param distance_vector:
    :param max_spread:
    :param mu:
    :param sigma:
    :return:
    """
    funz = mu * np.exp(-(1 / sigma ** 4) * (distance_vector) ** 2)
    x += funz * x_movement
    y += funz * y_movement
    return check_bounds2d(x, y, max_spread)

def normalize_reward(x, normaliztion_value):
    return x / normaliztion_value



