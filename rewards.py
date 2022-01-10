import numpy as np


# TODO: funzione.
def normalize_reward(x, normaliztion_value):
    return x / normaliztion_value


def dummy(t_x, t_y, t_z, c_x, c_y, c_z):

    x_diff = np.sum(abs(t_x - c_x))
    y_diff = np.sum(abs(t_y - c_y))
    z_diff = np.sum(abs(t_z - c_z))
    assert (np.sum(x_diff).size == 1 and np.sum(y_diff).size == 1 and np.sum(z_diff).size == 1)

    reward = - (x_diff + y_diff + z_diff) ** 2
    # reward = reward * 10
    return reward


def dummy2d(t_x, t_y, c_x, c_y):

    x_diff = np.sum(abs(t_x - c_x))
    y_diff = np.sum(abs(t_y - c_y))
    assert (np.sum(x_diff).size == 1 and np.sum(y_diff).size == 1)

    reward = - (x_diff + y_diff) ** 2
    # reward = reward * 10
    return reward
# sx = np.array([5, 5, 5])
# sy = np.array([5, 5, 5])
# sz = np.array([5, 5, 5])
#
# cx = np.array([-5, -5, -5])
# cy = np.array([-5, -5, -5])
# cz = np.array([-5, -5, -5])
#
# rew = dummy(sx, sy, sz, cx, cy, cz)
# print(rew/((5 * 2 * 3)))
