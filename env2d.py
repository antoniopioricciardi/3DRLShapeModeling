import gym
import math
import numpy as np
import matplotlib 
matplotlib.use('agg')
import matplotlib.pyplot as plt
import random

from gym import spaces
from metrics import compute_metrics
#from envs.utils import *

import inits, states, acts, rewards, poly, shapes
from collections import OrderedDict
from typing import Any, Dict, Optional, Union
from read_img import find_correspondence
from correspondence import chamfer_correspondence, sequential_correspondence
import io
from omegaconf import DictConfig, OmegaConf
import hydra


def plot3(fig):
    with io.BytesIO() as buff:
        fig.savefig(buff, format='raw')
        buff.seek(0)
        data = np.frombuffer(buff.getvalue(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    im = data.reshape((int(h), int(w), -1))
    return im

def normalize_reward(x, normaliztion_value):
    return x / normaliztion_value


#TODO: What if we give it positive reward where the lowest pos rew is given when distance is maximum
class CanvasModeling(gym.Env):
    metadata = {
        "render.modes": ["human", "rgb_array", "state_pixels"],
    }

    def __init__(self, cfg):
        super(CanvasModeling, self).__init__()
        self.num_points = cfg.num_points
        self.spread = cfg.spread
        self.max_steps = cfg.max_steps
        self.step_n = 0
        self.total_step_n = 0
        self.done = False
        self.f_inits_t = lambda: hydra.utils.instantiate(cfg.inits_t[list(cfg.inits_t.keys())[0]],
                                                         num_points=self.num_points)
        self.f_inits_c = lambda: hydra.utils.instantiate(cfg.inits_c[list(cfg.inits_c.keys())[0]],
                                                         num_points=self.num_points)

        self.f_act = lambda actions, c_x, c_y: hydra.utils.instantiate(
            cfg.actions[list(cfg.actions.keys())[0]],
            actions=actions, c_x=c_x, c_y=c_y,
            spread=self.spread)

        self.f_reward = lambda c_x, c_y, t_x, t_y: hydra.utils.instantiate(
            cfg.rewards[list(cfg.rewards.keys())[0]],
            c_x=c_x, c_y=c_y, t_x=t_x, t_y=t_y)

        # TODO: normalize obs space to be in -1,1 (hence, what before was in -10/10 will be -1/1)
        self.observation_space = spaces.Box(
            low=-2 * self.spread,
            high=2 * self.spread,
            shape=(self.num_points * 2,),  # x and y coordinates for each point in target - canvas pts
            dtype=np.float32
        )

        # consider using (+-spread) as a bound
        self.action_space = spaces.Box(
            low=-1.,
            high=1.,
            shape=(self.num_points * 2,),
            dtype=np.float32,
        )

    def reset(self):
        self.step_n = 0
        self.done = False
        self.t_x, self.t_y = self.f_inits_t()
        self.c_x, self.c_y = self.f_inits_c()

        self.state = np.float32(np.concatenate((self.t_x-self.c_x, self.t_y-self.c_y)))

        self.abs_dist, self.source_centroid, self.canvas_centroid, self.convex_hull_area_source, self.convex_hull_area_canvas = compute_metrics(
            self.t_x, self.t_y, self.c_x, self.c_y)
        return self.state

    def step(self, actions):
        self.c_x, self.c_y = self.f_act(actions, self.c_x, self.c_y)
        reward = self.f_reward(c_x=self.c_x, c_y=self.c_y,
                               t_x=self.t_x, t_y=self.t_y)
        reward = normalize_reward(reward, (self.spread * 2 * self.num_points) ** 2)  # a scalar

        self.state = np.float32(np.concatenate((self.t_x-self.c_x, self.t_y-self.c_y)))
        self.step_n += 1
        self.total_step_n += 1

        if self.total_step_n % 500 == 0:
            self.abs_dist, self.source_centroid, self.canvas_centroid, self.convex_hull_area_source, self.convex_hull_area_canvas = compute_metrics(
                self.t_x, self.t_y, self.c_x, self.c_y)

        info = {"is_success": False}
        if self.step_n == self.max_steps:
            self.done = True
        return self.state, reward, self.done, info

    def render(self, mode='human', close=False):
        marker_size = [70] * self.num_points
        fig = plt.figure()
        ax = fig.add_axes([0., 0., 1., 1.])
        ax.axis('off')

        ax.scatter(self.t_x, self.t_y, color='blue', s=marker_size)
        ax.scatter(self.c_x, self.c_y, color='red')
        x_vals = [self.c_x, self.t_x]
        y_vals = [self.c_y, self.t_y]
        ax.plot(x_vals, y_vals)
        ax.set_xlim([-self.spread*2, self.spread*2])
        ax.set_ylim([-self.spread*2, self.spread*2])

        ''' this commented code is to show the agent progressing in real time instead of only logging to wandb '''
        #plt.draw()
        # plt.show()
        # plt.pause(0.001)
        ''' until here '''
        plt.ioff()
        image = plot3(fig)
        plt.close()
        return image




    # def render(self, mode='human', close=False):
    #     marker_size = [70] * self.num_points
    #     fig = plt.figure()
    #     ax = fig.add_axes([0., 0., 1., 1.])
    #     ax.axis('off')
    #
    #     plt.scatter(self.s_x, self.s_y, s=marker_size)
    #     plt.scatter(self.c_x, self.c_y, color='red')
    #     # color_grad = np.linspace(0, 1, len(self.canvas_x_coords))
    #
    #     # plt.scatter(self.source_x_coords, self.source_y_coords, s=marker_size, c=color_grad, marker='+')#, edgecolors='blue')
    #     # plt.scatter(self.source_x_coords[self.pt_neighs], self.source_y_coords[self.pt_neighs], color='green')
    #     # plt.scatter(self.canvas_x_coords, self.canvas_y_coords, c=color_grad)# , edgecolors='red')
    #     x_vals = [self.c_x[self.point_idx], self.s_x[self.correspondence]]
    #     y_vals = [self.c_y[self.point_idx], self.s_y[self.correspondence]]
    #     plt.plot(x_vals, y_vals)
    #     # plt.scatter(self.canvas_x_coords[self.pt_neighs], self.canvas_y_coords[self.pt_neighs], color='orange')
    #     plt.xlim([-self.spread - 2, self.spread + 2])
    #     plt.ylim([-self.spread - 2, self.spread + 2])
    #
    #     # plt.draw()  # used to show the agent progressing
    #     # plt.pause(0.001)
    #     plt.ioff()
    #     image = plot3(fig)
    #     plt.close()
    #     return image

