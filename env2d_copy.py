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

    def __init__(self, cfg, simulated_initialization=False):
        super(CanvasModeling, self).__init__()
        self.num_points = cfg.num_points
        self.num_moving_points = cfg.neighborhood_size  # cfg.num_moving_points
        self.steps_per_round = cfg.steps_per_round
        self.spread = cfg.spread
        self.correspondence_type = cfg.correspondence
        self.neighbors_movement_scale = cfg.neighbors_movement_scale

        # self.neighborhood_type = cfg.neighborhood_type
        self.num_neighbors_per_side = cfg.neighborhood_size//2  # cfg.num_neighbors_per_side

        self.simulated_initialization = simulated_initialization

        self.f_inits_neighborhood = lambda: hydra.utils.instantiate(cfg.neighborhood[list(cfg.neighborhood.keys())[0]], num_points=self.num_points, num_neighbors_per_side=self.num_neighbors_per_side)
        self.f_inits_s = lambda : hydra.utils.instantiate(cfg.inits_s[list(cfg.inits_s.keys())[0]], num_points=self.num_points)
        self.f_inits_c = lambda : hydra.utils.instantiate(cfg.inits_c[list(cfg.inits_c.keys())[0]], num_points=self.num_points)

        # self.f_correspondence = lambda: hydra.utils.instantiate(cfg.correspondence[list(cfg.correspondence.keys())[0]], )

        self.s_x, self.s_y = self.f_inits_s() #inits.zero_init(self.num_points)
        self.c_x, self.c_y = self.f_inits_c() #inits.zero_init(self.num_points)
        self.neighbors = self.f_inits_neighborhood()

        self.action_type = list(cfg.actions.keys())[0]

        if self.neighbors is None:
            self.f_act = lambda actions, c_x, c_y: hydra.utils.instantiate(cfg.actions[list(cfg.actions.keys())[0]],
                                                                actions=actions, c_x = c_x, c_y = c_y, spread=self.spread)
        else:
            self.point_idx = 0
            if self.action_type == 'gaussian3d':
                self.f_act = lambda actions, c_x, c_y, point_idx, neighbors: hydra.utils.instantiate(
                    cfg.actions[list(cfg.actions.keys())[0]],
                    actions=actions, c_x=c_x, c_y=c_y,
                    spread=self.spread, point_idx=point_idx,
                    neighbors=neighbors,
                    neighbors_movement_scale=self.neighbors_movement_scale,
                    sigma=cfg.sigma)
            else:
                self.f_act = lambda actions, c_x, c_y, point_idx, neighbors: hydra.utils.instantiate(cfg.actions[list(cfg.actions.keys())[0]],
                                                                               actions=actions, c_x=c_x, c_y=c_y,
                                                                               spread=self.spread, point_idx=point_idx,
                                                                               neighbors=neighbors,
                                                                                neighbors_movement_scale=self.neighbors_movement_scale)

        self.f_reward = lambda c_x, c_y, s_x, s_y: hydra.utils.instantiate(cfg.rewards[list(cfg.rewards.keys())[0]],
                                                             c_x = c_x, c_y = c_y, s_x = s_x, s_y=s_y)

        # self.observation_space = spaces.Dict(
        #     {
        #         "observation": spaces.Box(
        #         low=-2*self.spread,
        #         high=2*self.spread,
        #         shape=(self.num_moving_points*2,),
        #         dtype=np.float32,
        #         ),
        #     }
        # )

        self.observation_space = spaces.Box(
            low=-2 * self.spread,
            high=2 * self.spread,
            shape=(self.num_moving_points * 2,),
            dtype=np.float32
        )
        # consider using (+-spread) as a bound
        self.action_space = spaces.Box(
            low=-1.,
            high=1.,
            shape=(2,),
            dtype=np.float32,
        )

        self.spread = self.spread
        self.l2_distances = 0

        self.step_n = 0
        self.total_step_n = 0
        self.max_steps = cfg.max_steps
        self.point_idx = 0

        self.abs_dist = None
        self.source_centroid = None
        self.canvas_centroid = None
        self.convex_hull_area_source = None
        self.convex_hull_area_canvas = None

        if self.simulated_initialization:
            self.fake_env = CanvasModeling(cfg, simulated_initialization=False)

        self.done = False
        self.do_render = False

    def reset(self):
        self.step_n = 0
        self.point_idx = 0  # for choosing the point to move

        if self.simulated_initialization:
            self.fake_env.reset()
            self.s_x = self.fake_env.c_x.copy()
            self.s_y = self.fake_env.c_y.copy()
            for i in range(75):
                x_acts = round(np.random.uniform(-1, 1), 3)
                y_acts = round(np.random.uniform(-1, 1), 3)
                self.fake_env.step([x_acts, y_acts])
            self.c_x = self.fake_env.c_x.copy()
            self.c_y = self.fake_env.c_y.copy()
        else:
            self.s_x, self.s_y = self.f_inits_s()

            self.c_x, self.c_y = self.f_inits_c()
        if self.correspondence_type == 'chamfer':
            source_points_mask, canvas_points_mask, self.correspondence = chamfer_correspondence(self.step_n, self.steps_per_round, self.s_x,
                                                                                self.s_y,
                                                                                self.c_x, self.c_y, self.neighbors)
        if self.correspondence_type == 'sequential':
            self.point_idx, source_points_mask, canvas_points_mask, self.correspondence = sequential_correspondence(self.step_n, self.steps_per_round, self.point_idx,
                                                                        self.num_points, self.neighbors)

        self.state = states.signed_dist(self.c_x[canvas_points_mask], self.c_y[canvas_points_mask],
                                        self.s_x[source_points_mask], self.s_y[source_points_mask])

        # self.state = states.signed_dist(self.c_x, self.c_y, self.s_x, self.s_y)
        self.l2_distances = np.sqrt((self.s_x - self.c_x) ** 2 + (
            self.s_y - self.c_y) ** 2)
        # self.abs_dist, self.source_centroid, self.canvas_centroid, self.convex_hull_area_source, self.convex_hull_area_canvas = compute_metrics(self.s_x, self.s_y, self.c_x, self.c_y)
        return np.float32(self.state)  # self._get_obs()

    def step(self, actions):
        self.step_n += 1
        self.total_step_n += 1

        # LOGGING DETERMINISTIC BEHAVIOUR
        # if self.total_step_n > 1000:
        #     print('now_print')
        #     print(f'c_x: {self.c_x} - c_y: {self.c_y} || s_x: {self.s_x} - s_y: {self.s_y}')
        #     print(f'actions: {actions}')
        #
        # if self.total_step_n > 1100:
        #     print('exiting')
        #     exit(3)
        if self.neighbors is None:
            self.c_x, self.c_y = self.f_act(actions, self.c_x, self.c_y)
        else:
            self.c_x, self.c_y = self.f_act(actions, self.c_x, self.c_y, point_idx=self.point_idx, neighbors=self.neighbors)

        reward = self.f_reward(c_x=self.c_x, c_y=self.c_y, s_x = self.s_x, s_y = self.s_y )
        reward = normalize_reward(reward, (self.spread * 2 * self.num_points)**2)  # a scalar

        if self.correspondence_type == 'chamfer':
            source_points_mask, canvas_points_mask, self.correspondence = chamfer_correspondence(self.step_n,
                                                                                                 self.steps_per_round,
                                                                                                 self.s_x,
                                                                                                 self.s_y,
                                                                                                 self.c_x, self.c_y,
                                                                                                 self.neighbors)
        if self.correspondence_type == 'sequential':
            self.point_idx, source_points_mask, canvas_points_mask, self.correspondence = sequential_correspondence(self.step_n,
                                                                                                    self.steps_per_round,
                                                                                                    self.point_idx,
                                                                                                    self.num_points,
                                                                                                    self.neighbors)

        self.state = states.signed_dist(self.c_x[canvas_points_mask], self.c_y[canvas_points_mask],
                                        self.s_x[source_points_mask], self.s_y[source_points_mask])

        self.done = False
        if self.step_n == self.max_steps:
            self.done = True
        info = {"is_success": False}
        self.l2_distances = np.sqrt((self.s_x - self.c_x) ** 2 + (
            self.s_y - self.c_y) ** 2)

        self.abs_dist, self.source_centroid, self.canvas_centroid, self.convex_hull_area_source, self.convex_hull_area_canvas = compute_metrics(self.s_x, self.s_y, self.c_x, self.c_y)
        # if np.sum(self.l2_distances < 0.1) == self.num_points:
        #     done = True
        #     info = {"is_success": True}
        # obs = self._get_obs()
        return np.float32(self.state), reward, self.done, info

    def get_distances(self):
        return self.l2_distances

    def render(self, mode='human', close=False):
        marker_size = [70] * self.num_points
        fig = plt.figure()
        ax = fig.add_axes([0., 0., 1., 1.])
        ax.axis('off')

        plt.scatter(self.s_x, self.s_y, s=marker_size)
        plt.scatter(self.c_x, self.c_y, color='red')
        # color_grad = np.linspace(0, 1, len(self.canvas_x_coords))

        # plt.scatter(self.source_x_coords, self.source_y_coords, s=marker_size, c=color_grad, marker='+')#, edgecolors='blue')
        # plt.scatter(self.source_x_coords[self.pt_neighs], self.source_y_coords[self.pt_neighs], color='green')
        # plt.scatter(self.canvas_x_coords, self.canvas_y_coords, c=color_grad)# , edgecolors='red')
        x_vals = [self.c_x[self.point_idx], self.s_x[self.correspondence]]
        y_vals = [self.c_y[self.point_idx], self.s_y[self.correspondence]]
        plt.plot(x_vals, y_vals)
        # plt.scatter(self.canvas_x_coords[self.pt_neighs], self.canvas_y_coords[self.pt_neighs], color='orange')
        plt.xlim([-self.spread - 2, self.spread + 2])
        plt.ylim([-self.spread - 2, self.spread + 2])

        # plt.draw()  # used to show the agent progressing
        # plt.pause(0.001)
        plt.ioff()
        image = plot3(fig)
        plt.close()
        return image


    # original render function
    # def render(self, mode='human', close=False):
    #     marker_size = [60] * self.num_points
    #     fig = plt.figure()
    #     ax = fig.add_axes([0.,0.,1.,1.])
    #     ax.axis('off')
    #
    #     plt.scatter(self.s_x, self.s_y, s=marker_size)
    #     plt.scatter(self.c_x, self.c_y, color='red')
    #     plt.xlim([-self.spread - 2, self.spread + 2])
    #     plt.ylim([-self.spread - 2, self.spread + 2])
    #
    #     #plt.draw()  # used to show the agent progressing
    #     #plt.pause(0.001)
    #     plt.ioff()
    #     image = plot3(fig)
    #     plt.close()
    #     return image

    def _get_obs(self) -> Dict[str, Union[int, np.ndarray]]:
        return OrderedDict(
             [
                 ("observation", self.state.copy()),
             ]
        )





