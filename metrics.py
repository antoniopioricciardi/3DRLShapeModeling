import math
import numpy as np
from scipy.spatial import ConvexHull

def compute_metrics(s_x, s_y, c_x, c_y):
    source = np.reshape(np.vstack((s_x, s_y)).T, (1, -1, 2))
    canvas = np.reshape(np.vstack((c_x, c_y)).T, (1, -1, 2))
    if len(source.shape) > 2:
        source = np.squeeze(source, axis=0)
    if len(canvas.shape) > 2:
        canvas = np.squeeze(canvas, axis=0)
    # TODO: get_distances_eucl ritorna una matrice, che ci facciamo?
    D_S_e = get_distances_eucl(source, source)
    D_C_e = get_distances_eucl(canvas, canvas)
    abs_dist = np.linalg.norm(D_S_e - D_C_e)

    source_centroid = get_centroid(s_x, s_y)
    canvas_centroid = get_centroid(c_x, c_y)

    convex_hull_area_source = calc_area(source)
    convex_hull_area_canvas = calc_area(canvas)
    return abs_dist, source_centroid, canvas_centroid, convex_hull_area_source, convex_hull_area_canvas


# todo: fai in modo che array1 sia quello da cui calcoliamo le distanze
def get_distances_eucl(array1, array2):
    num_point, num_features = array1.shape
    # tile constructs an array by repeating A the number of times given by reps.
    # (num_point, 1) creates a matrix that repeats the array for num_point times, once per row
    expanded_array1 = np.tile(array1, (num_point, 1))  # duplicate the whole array into a matrix, num_point times
    # repeat each element num_point times, in single-column matrix
    expanded_array2 = np.reshape(
        np.tile(np.expand_dims(array2, 1),
                (1, num_point, 1)),
        (-1, num_features))
    distances = np.linalg.norm(expanded_array1 - expanded_array2, axis=1)

    distance_matrix = np.reshape(distances, (num_point, num_point))
    return distance_matrix


def get_centroid(x_coords, y_coords):
    num_coords = len(x_coords)
    centr_x = np.sum(x_coords)/num_coords
    centr_y = np.sum(y_coords)/num_coords
    return centr_x, centr_y


def calc_area(points):
    # QJ Pp will joggle the input points slightly so you get to 3D.
    qh = ConvexHull(points, qhull_options='QJ Pp')  # if the function does not work
    # qh = ConvexHull(points)
    return qh.area


# TODO: decidi se spostare le inizializzazioni di shape da init a un file misc_utils
def get_polygon(num_points=4, spread=5):
    s_x = np.zeros(num_points)
    s_y = np.zeros(num_points)
    for i in range(num_points):
        angle = (i / num_points) * (math.pi * 2)
        s_x[i] = round(math.sin(angle), 2) * spread
        s_y[i] = round(math.cos(angle), 2) * spread
    return s_x, s_y