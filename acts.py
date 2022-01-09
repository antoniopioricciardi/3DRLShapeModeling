from envs.utils import *


def linear(actions, c_x, c_y, spread):
    num_points = len(actions)//2
    actions_x = actions[:num_points]
    actions_y = actions[num_points:]

    # move points according to agent's output
    c_x += actions_x
    c_y += actions_y

    c_x[c_x > spread] = spread
    c_x[c_x < -spread] = -spread
    c_y[c_y > spread] = spread
    c_y[c_y < -spread] = -spread

    return c_x, c_y


def single_linear(actions, c_x, c_y, spread, point_idx):
    """
    Move a single point
    :param actions:
    :param c_x:
    :param c_y:
    :param spread:
    :param point_idx:
    :return:
    """
    return move_point(point_idx, actions[0], actions[1], c_x, c_y, spread)


def neighborhood_linear(actions, c_x, c_y, spread, point_idx, neighbors, neighbors_movement_scale):
    """
    Move the neighborhood of a point
    :param actions:
    :param c_x:
    :param c_y:
    :param spread:
    :param point_idx:
    :param neighbors:
    :param neighbors_movement_scale:
    :return:
    """
    c_x, c_y = move_point(point_idx, actions[0], actions[1], c_x, c_y, spread)
    c_x, c_y = move_neighbors(point_idx, neighbors, actions[0]*neighbors_movement_scale, actions[1]*neighbors_movement_scale, c_x, c_y, spread)
    return c_x, c_y


def single_linear3d(actions, c_x, c_y, c_z, spread, point_idx):
    """
    Move a single point
    :param actions:
    :param c_x:
    :param c_y:
    :param c_z:
    :param spread:
    :param point_idx:
    :return:
    """
    return move_point3d(point_idx, actions[0], actions[1], actions[2], c_x, c_y, c_z, spread)


def neighborhood_linear3d(actions, c_x, c_y, c_z, spread, point_idx, neighbors, neighbors_movement_scale):
    """
    Move the neighborhood of a point
    :param actions:
    :param c_x:
    :param c_y:
    :param c_z:
    :param spread:
    :param point_idx:
    :param neighbors:
    :param neighbors_movement_scale:
    :return:
    """
    neighbors = [1,2]
    c_x, c_y, c_z = move_point3d(point_idx, actions[0], actions[1], actions[2], c_x, c_y, c_z, spread)
    c_x, c_y, c_z = move_neighbors3d(point_idx, neighbors,
                                     actions[0]*neighbors_movement_scale, actions[1]*neighbors_movement_scale, actions[2]*neighbors_movement_scale,
                                     c_x, c_y, c_z, spread)
    return c_x, c_y, c_z


def neighborhood_linear3d_anylength(actions, c_x, c_y, c_z, spread=5, point_idx=0, neighbors = [], neighbors_movement_scale = 1):
    """
    Move the neighborhood of a point working with neighborhood of any length
    :param actions:
    :param c_x:
    :param c_y:
    :param c_z:
    :param spread:
    :param point_idx:
    :param neighbors:
    :param neighbors_movement_scale:
    :return:
    """
    c_x, c_y, c_z = move_point3d(point_idx, actions[0], actions[1], actions[2], c_x, c_y, c_z, spread)
    c_x, c_y, c_z = move_neighbors3d(point_idx, neighbors,
                                     actions[0]*neighbors_movement_scale, actions[1]*neighbors_movement_scale, actions[2]*neighbors_movement_scale,
                                     c_x, c_y, c_z, spread)
    return c_x, c_y, c_z


def gaussian2d(actions, c_x, c_y, spread, point_idx, neighbors=[], neighbors_movement_scale=1, sigma=1):
    center_x = c_x[point_idx]
    center_y = c_y[point_idx]
    neighbors_idx = np.concatenate(([point_idx], neighbors[int(point_idx)]))
    distance_vector = np.sqrt((center_x - c_x[neighbors_idx])**2 + (center_y - c_y[neighbors_idx])**2)
    x, y = gaussian_bump2d(c_x[neighbors_idx], c_y[neighbors_idx], actions[0], actions[1], distance_vector, spread, neighbors_movement_scale, sigma)
    # move_points_gaussian2d()
    return x, y


def gaussian3d(actions, c_x, c_y, c_z, spread, point_idx=0, neighbors=[], neighbors_movement_scale=1, sigma=1):
    # neighbors_movement_scale = mu
    # TODO: Controlla che il bump dipenda dalla posizione dei vertici E dall'ordine dato in distance_vector
    # move_points_gaussian3d()
    center_x = c_x[0]
    center_y = c_y[0]
    center_z = c_z[0]
    distance_vector = np.sqrt((center_x-c_x)**2 + (center_y - c_y)**2 + (center_z - c_z)**2)
    x, y, z = gaussian_bump3d(c_x, c_y, c_z, actions[0], actions[1], actions[2], distance_vector, spread, neighbors_movement_scale, sigma)
    return x, y, z
