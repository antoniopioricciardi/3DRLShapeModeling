import os
import gym
import torch
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

import wandb
import imageio
from env3d_triangles_testing import CanvasModelingTest
import poly
import os

# TODO: need to be able to use other correspondences and distances oher that the chamfer. Create a top-layer function to use in env2d.

# TODO: clean, or fix outputs and models paths from the folders created during each run

from neighborhood import *
from inits import *
from misc_utils.shapes.operations import compute_triangle_triangle_adjacency_matrix_igl


def test(cfg, model, wandb_logger, model_name, save_animation_gif, res_path):
    env = CanvasModelingTest(cfg)
    neighborhood_size = cfg.neighborhood_size
    images = []
    # env.store_transition()

    steps_list = []
    dists_1 = []
    dists_2 = []
    reward_list = []
    reward_list.append(0)
    area_diff = 0
    # for i in range(3000):
    done = False
    test_score = 0
    i = 0
    tqdm_interval_update = 500  # if we update tqdm too often it will strongly slow down our code

    import time
    obs = env.reset(neighborhood_size)
    if save_animation_gif:
        img = env.render(mode='rgb_array')    # print(env.l2_distances, 'obs:', obs)
    # TODO: break quando errore minimo raggiunto
    start = time.time()
    with tqdm(total=cfg.max_steps) as pbar:
        while not done:
            i += 1
            if i % tqdm_interval_update == 0:
                pbar.update(tqdm_interval_update)
            # env.render()
            steps_list.append(i)
            # dists_1.append(env.l2_distances[0])
            # dists_2.append(env.l2_distances[1])
            action, _states = model.predict(obs, deterministic=True)
            obs, rewards, done, info = env.step(action)
            env.store_transition()
            # vid.capture_frame()
            if cfg.is_testing and env.total_step_n % 500 == 0:
                area_diff = abs(env.convex_hull_area_source - env.convex_hull_area_canvas)
                abs_dist = env.abs_dist
                source_centroid = env.source_centroid
                canvas_centroid = env.canvas_centroid
                centroid_x_diff = source_centroid[0] - canvas_centroid[0]
                centroid_y_diff = source_centroid[1] - canvas_centroid[1]
                centroid_z_diff = source_centroid[2] - canvas_centroid[2]
                # area_diff = abs(env.get_attr("convex_hull_area_source")[0] - env.get_attr("convex_hull_area_canvas")[0])
                # abs_dist = env.get_attr("abs_dist")[0]
                # source_centroid = env.get_attr("source_centroid")[0]
                # canvas_centroid = env.get_attr("canvas_centroid")[0]
                # centroid_x_diff = source_centroid[0] - canvas_centroid[0]
                # centroid_y_diff = source_centroid[1] - canvas_centroid[1]

                wandb.log({"area difference": area_diff})
                wandb.log({"absolute distance between shapes": abs_dist})
                wandb.log({"centroid x difference": centroid_x_diff})
                wandb.log({"centroid y difference": centroid_y_diff})
                wandb.log({"centroid z difference": centroid_z_diff})


            # wandb.log({"source centroid": source_centroid})
            # wandb.log({"canvas centroid": canvas_centroid})


            #TODO: need all(done) if we are working with >1 env

            # done = all(done)
            reward_list.append(rewards)
            if save_animation_gif:
                if (i % 150) == 0:
                    images.append(img)
                    img = env.render(mode='rgb_array')
            test_score += rewards

            if info['sweep_completed']:
                obs = env.advance_neighborhood(neighborhood_size)

    end = time.time() - start
    # print(end)
    if not os.path.exists(res_path):
        os.mkdir(res_path)
    res_path = os.path.join(res_path, 'transitions')
    if not os.path.exists(res_path):
        os.mkdir(res_path)
    env.save_transitions(os.path.join(res_path, model_name))

    # shutdown the logger
    if wandb_logger is not None:
        wandb_logger.finish()

    if save_animation_gif:
        imageio.mimsave(model_name + '-animation.gif', [np.array(img) for i, img in enumerate(images) if i % 2 == 0], fps=29)

    # Close Environment
    # env.close()

    return area_diff, abs_dist, centroid_x_diff, centroid_y_diff
