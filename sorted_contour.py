import numpy as np
import glob
import cv2 as cv
import matplotlib.pyplot as plt
from PIL import Image
import networkx as nx
from scipy.spatial import distance

import math


def rotate_points(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy


def obtain_line(x):
    G_x = x[1:, :] - x[:-1, :]
    G_x = G_x[:-1, 1:-1]
    G_x[G_x == 0] = 0
    G_x[G_x != 0] = 1

    G_y = x[:, 1:] - x[:, :-1]
    G_y = G_y[1:-1, 1:]
    G_y[G_y == 0] = 0
    G_y[G_y != 0] = 1

    contour = G_x + G_y
    line_f = np.vstack(np.where(contour)).T / G_y.shape[0]

    # line = line_f[pick,:].reshape(1,-1,2)
    return line_f


def retrieve_sorted_img(img_path='./images/seal.png', num_points=0, rotate=False):
    # Choose Image
    # im_frame = Image.open('./images/seal.png')
    im_frame = Image.open(img_path)

    dists = []
    imgs = []

    np_frame = np.array(im_frame)
    flag = np_frame[0, 0]  # get the "empty" value
    # if values in the image are different from this empty value, then 0
    np_frame[np_frame != flag] = np.int32(0)
    # 255 otherwise
    np_frame[np_frame == flag] = np.int32(255)
    # If the image is RGB, converting to BW
    if len(np_frame.shape) == 3:
        np_frame = cv.cvtColor(np_frame, cv.COLOR_BGR2GRAY)

    # Distance Transform to compute Signed Distance Function
    dist = cv.distanceTransform(np_frame, cv.DIST_L2, 5) - cv.distanceTransform(255 - np_frame, cv.DIST_L2, 5)
    dists.append(dist)
    imgs.append(np_frame)

    imgs = np.dstack(imgs)
    x = imgs[:, :, 0].astype(np.float32)

    # Line from image
    line_f = obtain_line(x)

    line_f = line_f[np.random.choice(line_f.shape[0]//1, line_f.shape[0]//2)]
    d=distance.cdist(line_f,line_f)

    idx = 0
    adj = np.zeros((line_f.shape[0],line_f.shape[0]))
    flagged = np.ones((line_f.shape[0]))
    flagged[idx]=0
    while True:
        np.sum(flagged)
        valids = np.where(flagged)
        vec = d[idx,:]
        vec[idx] = np.inf
        new_idx = np.argmin(vec[valids])
        val = np.min(vec[valids])
        new_idx = valids[0][new_idx]
        if val < 0.4:
            adj[idx, new_idx] = adj[new_idx, idx] = 1
            idx = new_idx
        flagged[new_idx]=0
        if not(np.any(flagged)):
            break

    G=nx.from_numpy_matrix(adj)

    # nx.draw(G, line_f, node_size = 0.2)
    # plt.show()

    p = nx.shortest_path(G,0,idx)
    sorted = line_f[p]
    sorted_x = sorted[:,0]
    sorted_y = sorted[:,1]
    new_max = 1
    new_min = -1
    old_max_x = np.max(sorted_x)
    old_min_x = np.min(sorted_x)
    sorted_x = (((sorted_x - old_min_x)*(new_max - new_min))/(old_max_x - old_min_x)) + new_min
    old_max_y = np.max(sorted_y)
    old_min_y = np.min(sorted_y)
    sorted_y = (((sorted_y - old_min_y)*(new_max - new_min))/(old_max_y - old_min_y)) + new_min
    # sorted = np.vstack((sorted_x, sorted_y)).T


    x_coords = []
    y_coords = []

    if num_points == 0:
        num_points = len(sorted_x)  # len(line_f)  # this sometimes has different number of points from sorted_x,y
    mod = len(line_f) // num_points
    for idx, el in enumerate(line_f):
        if (idx+1) % mod == 0:
            x_coords.append(sorted_x[idx])
            y_coords.append(sorted_y[idx])
            if len(x_coords) == num_points:
                break
    x_coords = np.array(x_coords)
    y_coords = np.array(y_coords)
    if rotate:
        x_coords, y_coords = rotate_points((0, 0), (x_coords, y_coords), math.radians(-90))

    # coords = np.arange(0, num_points)
    # coords = (coords + 100) % num_points

    return x_coords, y_coords


# x_coords, y_coords = retrieve_sorted_img(img_path='./images/snapchat.png', rotate=True)
# plt.scatter(x_coords,y_coords, 16,np.arange(0,len(x_coords)))
# plt.show()
