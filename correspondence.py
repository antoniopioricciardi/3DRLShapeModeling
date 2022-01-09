import numpy as np
from read_img import retrieve_image_points

def chamfer_distance_numpy(array1, array2):
    global distance_matrix
    batch_size, num_point, num_features = array1.shape
    dist = 0

    # TODO: Implement min of min distance between d(arr1,arr2) and d(arr2,arr1) (now it's one way only)
    for i in range(batch_size):
        point_idx_1, correspondence_1, dist_1, dist_matrix = array2samples_distance(array1[i], array2[i])
        # need to invert correspondence_2 and point_idx_2 because we are interested in moving points from array_1
        correspondence_2, point_idx_2, dist_2, _ = array2samples_distance(array2[i], array1[i])

        distance_matrix = dist_matrix
        if dist_1 > dist_2:
            return point_idx_1, correspondence_1
        else:
            return point_idx_2, correspondence_2
        # av_dist1 = array2samples_distance(array1[i], array2[i])
        # av_dist2 = array2samples_distance(array2[i], array1[i])
        # dist = dist + (av_dist1+av_dist2)/batch_size
    # return point_idx, correspondence


def array2samples_distance(array1, array2):
    """
    arguments:
        array1: the array, size: (num_point, num_feature)
        array2: the samples, size: (num_point, num_feature)
    returns:
        distances: each entry is the distance from a sample to array1
    """
    # global dist_matrix
    num_point, num_features = array1.shape
    # tile constructs an array by repeating A the number of times given by reps.
    # (num_point, 1) creates a matrix that repeats the array for num_point times, once per row
    expanded_array1 = np.tile(array1, (num_point, 1))  # duplicate the whole array into a matrix, num_point times
    # repeat each element num_point times, in single-column matrix
    expanded_array2 = np.reshape(
            np.tile(np.expand_dims(array2, 1),
                    (1, num_point, 1)),
            (-1, num_features))
    distances = np.linalg.norm(expanded_array1-expanded_array2, axis=1)
    distance_matrix = np.reshape(distances, (num_point, num_point))
    # dist_matrix = distances
    # distances is a matrix of distances between points in array1 (rows) and array2 (columns)
    min_row_el = np.argmin(distance_matrix, axis=1)  # get the idx of the smallest element for each row
    # use range and indices to obtain the smallest distance for each row
    per_point_smallest_dist = distance_matrix[range(len(distance_matrix)), min_row_el]
    point_idx = np.argmax(per_point_smallest_dist)  # finally, obtain array1 point matched to the max min value
    correspondence = min_row_el[point_idx]  # and subsequently get the matched array2 point (correspondence)
    # plt.scatter(s_x, s_y, marker='+')
    # plt.scatter(c_x, c_y, color='red')
    # plt.scatter(s_x[correspondence], s_y[correspondence], color='green')
    # plt.scatter(c_x[point_idx], c_y[point_idx], color='orange')
    # plt.show()  # used to show a single still
    # color_grad = correspondence
    # plt.scatter(s_x, s_y, c=color_grad, marker='+')

    return point_idx, correspondence, per_point_smallest_dist[point_idx], distance_matrix


def chamfer(s_x, s_y, c_x, c_y):
    # TODO: da rinominare
    source = np.reshape(np.vstack((s_x,s_y)).T,(1,-1,2))
    canvas = np.reshape(np.vstack((c_x,c_y)).T,(1,-1,2))
    # reward = - chamfer_distance_numpy(source, canvas)
    point_idx, correspondence = chamfer_distance_numpy(source, canvas)
    return point_idx, correspondence


def get_chamfer_correspondence(idx):
    # TODO: da rinominare
    """

    :param idx: an index or a list of indices to recover correspondence for
    :return:
    """
    global distance_matrix
    return np.argmin(distance_matrix[idx], axis=1)

'''METHODS TO RETURN CORRESPONDENCES INDICES'''

pt_neighs = None
point_idx = None
neigh_correspondence = None
correspondence = None

def chamfer_correspondence(step_n, steps_per_round, s_x, s_y, c_x, c_y, neighbors):
    global pt_neighs
    global point_idx
    global neigh_correspondence
    global correspondence

    if step_n % (steps_per_round + 1) == 0:
        # find new correspondence to return the new state set
        point_idx, correspondence = chamfer(s_x, s_y, c_x, c_y)
        pt_neighs = neighbors[point_idx].copy()
        neigh_correspondence = get_chamfer_correspondence(pt_neighs)

    canvas_points_mask = np.insert(pt_neighs, len(pt_neighs) // 2, point_idx)
    source_points_mask = np.insert(neigh_correspondence, len(neigh_correspondence) // 2,
                                   correspondence)
    return canvas_points_mask, source_points_mask, correspondence

def sequential_correspondence(step_n, steps_per_round, point_idx, num_points, neighbors):
    if step_n % (steps_per_round + 1) == 0:
        if not step_n == 0:
            point_idx += 1
            if point_idx == num_points:
                point_idx = 0

    pt_neighs = neighbors[point_idx].copy()
    pt_neighs.insert(len(neighbors[point_idx]) // 2, point_idx)
    canvas_points_mask = pt_neighs
    source_points_mask = pt_neighs
    return point_idx, canvas_points_mask, source_points_mask, point_idx


# import math
# import matplotlib.pyplot as plt
# n_pts = 10
# point_dist = 1 / n_pts
# spread = 4
# c_x = np.zeros(n_pts)
# c_y = np.zeros(n_pts)
#
# for i in range(n_pts):
#     angle = (i * point_dist) * (math.pi * 2)
#     # angle = random.random()*self.num_points * (math.pi * 2)
#
#     # self.source_x_coords[i] = random.randint(-self.pts_max_spread//2, self.pts_max_spread//2)
#     # self.source_y_coords[i] = random.randint(-self.pts_max_spread//2, self.pts_max_spread//2)
#     c_x[i] = round(math.sin(angle), 2) * spread
#     c_y[i] = round(math.cos(angle), 2) * spread
#     # self.source_x_coords[i] = round(math.sin(angle), 2) * self.pts_max_spread/2
#     # self.source_y_coords[i] = round(math.cos(angle), 2) * self.pts_max_spread/2

# s_x, s_y = retrieve_image_points('../images/dolphin/*', n_pts)
# s_x *= 4
# s_y *= 4
# i=0
# distance_matrix = None
#
# # print(chamfer_correspondence(s_x, s_y, c_x, c_y))
# print(chamfer_correspondence(s_x, s_y, c_x, c_y))
# print('dm', distance_matrix)
# print('-.-.-.-.-.-')
# print(distance_matrix[[1,2]])
# print(np.argmin(distance_matrix[[1,2]], axis=1))
#
# # color_grad = np.linspace(0, 1, len(canvas_x_coords))
# pt_idx, correspondence = chamfer_correspondence(s_x, s_y, c_x, c_y)
# color_grad = correspondence
# plt.scatter(s_x, s_y, c=color_grad, marker='+')
# plt.scatter(c_x, c_y, color='red')
# # plt.scatter(s_x[correspondence], s_y[correspondence], color='green')
# # plt.scatter(c_x[point_idx], c_y[point_idx], color='orange')
# plt.show()  # used to show a single still