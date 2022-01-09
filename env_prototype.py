import gym
import math
import numpy as np
import matplotlib
# matplotlib.use('agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random

from gym import spaces
#from envs.misc_utils import *
from neighborhood import *
from reset_helper import *

import inits,states, acts, rewards, poly, shapes
from collections import OrderedDict
from typing import Any, Dict, Optional, Union
from metrics3d import compute_metrics3d
import io
from omegaconf import DictConfig, OmegaConf
import hydra
import pickle

from inits import from_shape
from misc_utils.shapes.operations import *
from inits import *
from acts import *
from rewards import *
from misc_utils.shapes.remeshing import *


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

def normalize_vertices_3d(vertices, spread):
    x = normalize_vector(vertices[:, 0]) * spread
    y = normalize_vector(vertices[:, 1]) * spread
    z = normalize_vector(vertices[:, 2]) * spread
    return np.vstack((x, y, z)).T


class CanvasModeling(gym.Env):
    metadata = {
        "render.modes": ["human", "rgb_array", "state_pixels"],
    }

    def __init__(self, cfg):
        super(CanvasModeling, self).__init__()
        """ is_testing must be True if we are testing. This is used to avoid setting the env to done,
        hence to avoid resetting the env. Moreover, the entire shape will be plotted instead of a subsample of vertices """
        self.is_testing = cfg.is_testing

        """ Mesh loading """
        if self.is_testing:
            self.mesh_target, self.vert_target, self.triangles = load_mesh_new('./shapes/tr_reg_000_low_poly.ply', simplify=False)
            self.mesh_canvas, self.vert_canvas, _ = load_mesh_new('./shapes/smpl_base_neutro_low_poly.ply', simplify=False)
        else:
            self.mesh_target, self.vert_target, self.triangles = load_mesh_new('../../../shapes/tr_reg_000_low_poly.ply', simplify=False)
            self.mesh_canvas, self.vert_canvas, _ = load_mesh_new('../../../shapes/smpl_base_neutro_low_poly.ply', simplify=False)
        self.vert_target = normalize_vertices_3d(self.vert_target, cfg.spread)
        self.vert_canvas = normalize_vertices_3d(self.vert_canvas, cfg.spread)
        # self.triangles = self.triangles[:3]
        # self.vert_target = self.vert_target[:6]
        # self.vert_canvas = self.vert_canvas[:6]
        # self.mesh_canvas, self.vert_canvas, _ = load_mesh_new('../../../shapes/sphere_rem.ply', simplify=False)

        """ Hyperparameters """
        self.steps_per_vertex = cfg.steps_per_vertex
        self.max_steps = cfg.max_steps
        self.neighborhood_size = cfg.neighborhood_size
        self.spread = cfg.spread
        self.mu = cfg.mu
        self.sigma = cfg.sigma
        self.vertex_neighborhood_size = cfg.neighborhood_size # 10
        self.total_step_n = 0

        """ Used for logging metrics """
        self.l2_distances = 0
        self.abs_dist = None
        self.source_centroid = None
        self.canvas_centroid = None
        self.convex_hull_area_source = None
        self.convex_hull_area_canvas = None

        self.done = False
        self.do_render = False

        self.vertex_idx = -1
        self.context_idx = 0

        " landmarks to choose the vertices to work with "
        self.landmarks = np.arange(len(self.vert_canvas))
        # self.landmarks = np.arange(6)
        self.observation_space = spaces.Box(
            low=-5 * self.spread,
            high=5 * self.spread,
            shape=(self.vertex_neighborhood_size * 3,),  # neighborhood, central vertex, 3 coord each  (if 6, because we use actual vertices (canvas and target and not distances) (a-b is 1 value, a AND b are 2)
            # shape=((6,)),
            dtype=np.float32
        )

        self.action_space = spaces.Box(
            low=-1.,
            high=1.,
            shape=(3,),
            dtype=np.float32,
        )

        self.sweeps_per_context = cfg.sweeps_per_context
        self.context_sweep_n = 0

        """ Code to visualize vertices when the env is created. Can be safely deleted/moved (but might be useful to
        do some quick plotting). This is NOT env render """
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.axis('off')
        #
        # ax.scatter(self.vert_target[:,0], self.vert_target[:,1], self.vert_target[:,2], color='red')
        # ax.scatter(self.vert_canvas[:,0], self.vert_canvas[:,1], self.vert_canvas[:,2], color='green')
        # # plt.draw()  # used to show the agent progressing
        # plt.show()
        # plt.pause(0.001)
        # # plt.ioff()
        # # image = plot3(fig)
        # plt.close()
        # exit(11)

    def get_selector(self, central_vertex, sorting_idx, vertex_neighborhood_size, num_vertices):
        ''' we will have to loop over these items and use them as central vertex during a whole episode '''
        env_context = sorting_idx[central_vertex][:vertex_neighborhood_size]
        # env_context = dist_idx.copy()  # might rename dist_idx to env_context directly..
        selector = np.zeros(num_vertices, dtype=bool)
        '''
        set all the elements specified in env_context to True, so that we can later use selector to select
        items from other matrices (if we directly used dist_idx we would have gotten a smaller vector,
        containing only needed items, sorted according to dist_idx (AND THINGS WOULD NOT WORK)
        '''
        selector[env_context] = True
        return selector, env_context

    def get_selector_row_sorting(self, central_vertex, sorting_idx,vertex_neighborhood_size, num_vertices):
        ''' we will have to loop over these items and use them as central vertex during a whole episode '''
        env_context = sorting_idx[:vertex_neighborhood_size]
        selector = np.zeros(num_vertices, dtype=bool)
        '''
        set all the elements specified in env_context to True, so that we can later use selector to select
        items from other matrices (if we directly used dist_idx we would have gotten a smaller vector,
        containing only needed items, sorted according to dist_idx (AND THINGS WOULD NOT WORK)
        '''
        selector[env_context] = True
        return selector, env_context

    def reset(self):
        self.step_n = 0
        self.vertex_idx += 1
        if self.vertex_idx == len(self.landmarks):
            self.vertex_idx = 0
            self.context_sweep_n += 1
        # reset c and t values to those of the original mesh
        self.t_x = self.vert_target[:, 0]
        self.t_y = self.vert_target[:, 1]
        self.t_z = self.vert_target[:, 2]
        self.c_x = self.vert_canvas[:, 0]
        self.c_y = self.vert_canvas[:, 1]
        self.c_z = self.vert_canvas[:, 2]
        self.num_vertices = len(self.c_x)  # for plotting
        # TODO: this could go into init
        self.adj_matrix = compute_adjacency_matrix_igl(self.triangles)
        self.dist_matrix = compute_distance_matrix(self.vert_canvas, self.adj_matrix)

        # self.sorting_idx = np.argsort(self.dist_matrix[self.vertex_idx])
        self.sorting_idx = np.argsort(self.dist_matrix)
        sorted_matrix = np.sort(self.dist_matrix)

        self.central_vertex = self.landmarks[self.vertex_idx]
        self.selector, self.env_context = self.get_selector(self.central_vertex, self.sorting_idx, self.vertex_neighborhood_size, len(self.c_x))
        # print(len(self.c_x[self.selector]), len(self.c_x[self.env_context]))
        # print(self.c_x[self.env_context], self.c_x[self.central_vertex])
        # exit(5)
        # self.selector, self.env_context = self.get_selector_row_sorting(self.central_vertex, self.sorting_idx,
        #                                                     self.vertex_neighborhood_size, len(self.c_x))

        self.distance_from_vertex = self.dist_matrix[self.central_vertex][self.selector]

        # use selector to pick the collection of vertices closer to the central vertex
        context_c_x = self.c_x[self.env_context]
        context_c_y = self.c_y[self.env_context]
        context_c_z = self.c_z[self.env_context]
        context_t_x = self.t_x[self.env_context]
        context_t_y = self.t_y[self.env_context]
        context_t_z = self.t_z[self.env_context]

        # context_t_x, context_t_y, context_t_z, _ = rand_init3d(num_triangles=1, num_points=3, center_x=0, center_y=0, center_z=0, span_x=0.5, span_y=0.5, span_z=0.5, spread=5)
        # context_c_x, context_c_y, context_c_z, _ = rand_init3d(num_triangles=1, num_points=3, center_x=0, center_y=0, center_z=0, span_x=0.5, span_y=0.5, span_z=0.5, spread=5)

        # self.state = np.concatenate((context_c_x, context_c_y, context_c_z, context_t_x, context_t_y, context_t_z))
        self.state = states.signed_dist_3D(context_c_x, context_c_y, context_c_z, context_t_x, context_t_y, context_t_z)


        return np.float32(self.state)

    def step(self, actions):
        """ a little debug print to make sure everything loops as it should """
        # print(f"step_n: {self.step_n} -- total_step_n: {self.total_step_n} --- vertex_idx: {self.vertex_idx} --- context_idx: {self.context_idx} --- context len: {len(self.env_context)} --- context_sweep_n: {self.context_sweep_n}")
        # print(np.where(self.selector))
        """ increase counters. total_step_n is used for saving metrics """
        self.step_n += 1
        self.total_step_n += 1
        # update vertex positions according to actions
        # self.c_x_new, self.c_y_new, self.c_z_new = gaussian3d(self.c_x[self.selector], self.c_y[self.selector], self.c_z[self.selector], actions, self.distance_from_vertex, 2, 0.2, 0.3)
        '''THIS MUST ONLY BE USED WHEN USING SCALED NEIGHBORHOOD MOVEMENT '''
        neighbors = np.arange(self.vertex_neighborhood_size-1)
        neighbors[self.context_idx:] += 1
        spread=self.spread
        neighbors_movement_scale=0.5
        self.c_x_new, self.c_y_new, self.c_z_new = neighborhood_linear3d_anylength(self.c_x[self.selector], self.c_y[self.selector], self.c_z[self.selector],
                                        actions, self.context_idx, neighbors, neighbors_movement_scale, spread)
        ''' UP TO HERE '''


        self.c_x[self.selector] = self.c_x_new
        self.c_y[self.selector] = self.c_y_new
        self.c_z[self.selector] = self.c_z_new

        reward = dummy(c_x=self.c_x[self.env_context], c_y=self.c_y[self.env_context],
                       c_z=self.c_z[self.env_context],
                       s_x=self.t_x[self.env_context], s_y=self.t_y[self.env_context],
                       s_z=self.t_z[self.env_context])

        # *2 is because we have canvas and target, 3 is for x, y and z coordinates.
        reward = normalize_reward(reward, (self.spread * 2 * 3) ** 2)  # a scalar

        """ once the env performed steps_per_vertex steps, move to the next element
        in the context vector, recompute the distance matrix,"""
        if self.step_n % self.steps_per_vertex == 0:
            v_canv = np.vstack((self.c_x, self.c_y, self.c_z)).T
            # recompute the distance matrix using current vertices positions
            self.dist_matrix = compute_distance_matrix(v_canv, self.adj_matrix)
            self.sorting_idx = np.argsort(self.dist_matrix)
            # self.sorting_idx = np.argsort(self.dist_matrix[self.context_idx])

            ''' pick the central vertex as the next element in the context vector '''
            self.central_vertex = self.env_context[self.context_idx]
            ''' recompute selector '''
            # self.selector, self.env_context = self.get_selector(self.central_vertex, self.sorting_idx,
            #                                                     self.vertex_neighborhood_size,
            #                                                     len(self.c_x))
            # self.selector, self.env_context = self.get_selector_row_sorting(self.central_vertex, self.sorting_idx,
            #                                                     self.vertex_neighborhood_size, len(self.c_x))
            ''' get the distance vector for the central vertex'''
            self.distance_from_vertex = self.dist_matrix[self.central_vertex][self.selector]
            self.context_idx += 1

            ''' if we reached the end of context vector, reset the idx and update the context_sweep_n '''
            if self.context_idx == (len(self.env_context)):
                # start again sweeping this context and increase the number of sweeps for THIS context.
                self.context_idx = 0
                self.context_sweep_n += 1

        ''' if we are testing and have reached the desired number of repeating sweeps per a context '''
        if self.is_testing and (self.context_sweep_n == self.sweeps_per_context):
            self.context_sweep_n = 0
            self.dist_matrix = compute_distance_matrix(self.vert_canvas, self.adj_matrix)

            self.sorting_idx = np.argsort(self.dist_matrix)
            # sorted_matrix = np.sort(self.dist_matrix)

            ''' take central vertex from landmarks list '''
            self.central_vertex = self.landmarks[self.vertex_idx]
            # self.selector, self.env_context = self.get_selector_row_sorting(self.central_vertex, self.sorting_idx, self.vertex_neighborhood_size,
            #                                   len(self.c_x))
            self.selector, self.env_context = self.get_selector(self.central_vertex, self.sorting_idx,
                                                                self.vertex_neighborhood_size,
                                                                len(self.c_x))
            self.distance_from_vertex = self.dist_matrix[self.central_vertex][self.selector]

            # context_c_x = self.c_x[self.selector]
            # context_c_y = self.c_y[self.selector]
            # context_c_z = self.c_z[self.selector]
            # context_t_x = self.t_x[self.selector]
            # context_t_y = self.t_y[self.selector]
            # context_t_z = self.t_z[self.selector]
            # self.state = np.concatenate((context_c_x, context_c_y, context_c_z, context_t_x, context_t_y, context_t_z))

            self.vertex_idx += 1
            if self.vertex_idx == len(self.landmarks - 1):
                self.vertex_idx = 0
                self.context_sweep_n += 1

        context_c_x = self.c_x[self.env_context]
        context_c_y = self.c_y[self.env_context]
        context_c_z = self.c_z[self.env_context]
        context_t_x = self.t_x[self.env_context]
        context_t_y = self.t_y[self.env_context]
        context_t_z = self.t_z[self.env_context]
        # self.state = np.concatenate((self.c_x[self.selector], self.c_y[self.selector], self.c_z[self.selector],
        #                              self.t_x[self.selector], self.t_y[self.selector], self.t_z[self.selector]))

        self.state = states.signed_dist_3D(context_c_x, context_c_y, context_c_z, context_t_x, context_t_y, context_t_z)

        # TODO: this is a test, we might need to recompute distances every step, to allow neighborhood to move
        # v_canv = np.vstack((self.c_x, self.c_y, self.c_z)).T
        # self.dist_matrix = compute_distance_matrix(v_canv, self.adj_matrix)
        # # self.dist_matrix = compute_distance_matrix(v_canv, self.adj_matrix[self.context_idx])
        # self.sorting_idx = np.argsort(self.dist_matrix)
        # # sorted_matrix = np.sort(self.dist_matrix)
        #
        # self.central_vertex = self.landmarks[self.vertex_idx]
        # print(self.selector.shape)
        # self.selector, self.env_context = self.get_selector(self.central_vertex, self.sorting_idx,
        #                                                     self.vertex_neighborhood_size,
        #                                                     len(self.c_x))
        # self.distance_from_vertex = self.dist_matrix[self.central_vertex][self.selector]

        """ END OF TODO """

        done = False
        if self.step_n == self.max_steps:
            done = True
            info = {"is_success": False}  # we are not using info, but stable baselines requires it to be returned.
            return np.float32(self.state), reward, done, info

        info = {"is_success": False}

        if self.total_step_n % 500 == 0:
            self.abs_dist, self.source_centroid, self.canvas_centroid, self.convex_hull_area_source, self.convex_hull_area_canvas = compute_metrics3d(
                self.t_x, self.t_y, self.t_z, self.c_x, self.c_y, self.c_z)
        return np.float32(self.state), reward, done, info

    def get_distances(self):
        return self.l2_distances

    def render(self, mode='human', close=False):
        marker_size = [60] # * len(self.c_x[self.selector])  # self.num_vertices
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.axis('off')
        # if testing, visualize the entire shape
        if self.is_testing:
            ax.scatter(self.t_x, self.t_y, self.t_z,# s=marker_size,
                       color='blue')
            ax.scatter(self.c_x, self.c_y, self.c_z, color='red')
        else:
            ax.scatter(self.t_x[self.selector], self.t_y[self.selector], self.t_z[self.selector], color='blue')# s=marker_size,)
            ax.scatter(self.c_x[self.selector], self.c_y[self.selector], self.c_z[self.selector], color='red')
        ax.scatter(self.c_x[self.central_vertex], self.c_y[self.central_vertex], self.c_z[self.central_vertex],
                   color='green', s=60)
        ax.scatter(self.c_x[self.central_vertex], self.c_y[self.central_vertex], self.c_z[self.central_vertex],
                   color='orange')
            # ax.scatter(self.c_x, self.c_y, self.c_z, color='red')
        ax.set_xlim([-self.spread - 1, self.spread + 1])
        ax.set_ylim([-self.spread - 1, self.spread + 1])
        ax.set_zlim([-self.spread - 1, self.spread + 1])

        # plt.draw()  # used to show the agent progressing
        # plt.show()
        # plt.pause(0.001)
        plt.ioff()
        image = plot3(fig)
        plt.close()
        return image
