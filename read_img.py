import numpy as np
import glob
import cv2 as cv
import matplotlib.pyplot as plt
from PIL import Image


# Computing the gradient in X and Y direction to detect contours
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


# n_points=0 retrieves all points
def retrieve_image_points(img_path='../images/*.png', num_points=0):
    # List of images (NOTE: All should share the same dimension)
    lista = glob.glob(img_path, recursive=True)
    dists = []
    imgs = []
    # Change the second value to iterate on len(lista)
    for i in np.arange(0, 1):
        # i
        img = lista[i]
        im_frame = Image.open(img)
        np_frame = np.array(im_frame)
        flag = np_frame[0, 0]
        np_frame[np_frame != flag] = np.int32(0)
        np_frame[np_frame == flag] = np.int32(255)

        # If the image is RGB, converting to BW
        if len(np_frame.shape) == 3:
            np_frame = cv.cvtColor(np_frame, cv.COLOR_BGR2GRAY)

        # Distance Transform to compute Signed Distance Function
        dist = cv.distanceTransform(np_frame, cv.DIST_L2, 5) - cv.distanceTransform(255 - np_frame, cv.DIST_L2, 5)
        dists.append(dist)
        imgs.append(np_frame)

    imgs = np.dstack(imgs)
    np.random.seed(0)

    n_contours_p = 220

    x = imgs[:, :, 0].astype(np.float32)

    # Line from image
    line_f = obtain_line(x)

    # Plot
    # plt.scatter(line_f[:, 0], line_f[:, 1], color=[0, 0, 1], s=16)
    # plt.show()

    x_coords = []
    y_coords = []

    if num_points == 0:
        num_points = len(line_f)
    mod = len(line_f) // num_points
    for idx, el in enumerate(line_f):
        if (idx+1) % mod == 0:
            x_coords.append(el[0])
            y_coords.append(el[1])
            if len(x_coords) == num_points:
                break

    # center in [-1,1]
    x_coords = np.array(x_coords)
    y_coords = np.array(y_coords)

    x_coords = x_coords - np.mean(x_coords)
    y_coords = y_coords - np.mean(y_coords)

    # plt.scatter(x_coords*5, y_coords*5, color=[0, 0, 1], s=16)
    # plt.show()
    return x_coords, y_coords


def find_correspondence(src_pts, canvas_pts, mode='shortest_path'):
    correspondence_list = np.zeros(len(src_pts), dtype=int)  # won't contain repetitions
    temp_src_pts = src_pts.copy()
    if mode=='shortest_path':
        # TODO: Avoid using for loop
        for idx, el in enumerate(canvas_pts):
            # distances contains the l2 distances from the i-th point in canvas_pts to all other points in src_pts
            # distances = np.sqrt(np.sum((el - src_pts)**2, 1))
            distances = np.linalg.norm(el - temp_src_pts, axis=1)
            # closest_idx = np.argmax(distances)
            sorted_dist = np.argsort(distances)
            closest_idx = -1
            for i in sorted_dist:
                if i in correspondence_list:
                    continue
                else:
                    closest_idx = i
                    break
            correspondence_list[idx] = closest_idx
            # delete from temp_src_pts, the i-th element in dimension 0. We are removing an entire row (the i-th point coords)
            # temp_src_pts = np.delete(temp_src_pts, closest_idx, 0)

    return src_pts[correspondence_list]


# a,b = retrieve_image_points(img_path='../images/dolphin/*.png', num_points=150)