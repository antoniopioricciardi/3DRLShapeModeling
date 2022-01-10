import gym
# matplotlib.use('agg')
import matplotlib.pyplot as plt

from gym import spaces
from neighborhood import *

import states
from collections import OrderedDict
from typing import Any, Dict, Optional, Union
import io
import hydra
import pickle


# TODO: ESEGUI TEST CON RANDOM INIT E LOGGA SEQUENZA DI AZIONI E COORDINATE

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

class CanvasModeling(gym.Env):
    metadata = {
        "render.modes": ["human", "rgb_array", "state_pixels"],
    }

    def __init__(self, cfg):
        super(CanvasModeling, self).__init__()
        self.init_type = list(cfg.inits_s.keys())[0]
        self.num_triangles = cfg.num_triangles
        self.num_moving_points = 3
        self.neighborhood_size = 3
        self.steps_per_vertex = cfg.steps_per_vertex
        self.spread = cfg.spread
        self.neighbors_movement_scale = cfg.neighbors_movement_scale

        self.f_inits_s = lambda : hydra.utils.instantiate(cfg.inits_s[list(cfg.inits_s.keys())[0]], num_triangles=self.num_triangles)
        self.f_inits_c = lambda : hydra.utils.instantiate(cfg.inits_c[list(cfg.inits_c.keys())[0]], num_triangles=self.num_triangles)
        self.f_inits_neighborhood = lambda: hydra.utils.instantiate(cfg.neighborhood[list(cfg.neighborhood.keys())[0]], triangles=self.triangles)

        # TODO: this should become a dictionary that, given a triangle index, returns the 3-vertex neighborhood.
        self.neighbors = np.arange(1,self.neighborhood_size)
        self.f_act = lambda actions, c_x, c_y, c_z, point_idx, neighbors : hydra.utils.instantiate(cfg.actions[list(cfg.actions.keys())[0]],
                                                            actions=actions, c_x=c_x, c_y=c_y, c_z=c_z,
                                                            spread=self.spread, point_idx=point_idx,
                                                             neighbors=neighbors, neighbors_movement_scale=self.neighbors_movement_scale)

        self.f_reward = lambda c_x, c_y, c_z, s_x, s_y, s_z : hydra.utils.instantiate(cfg.rewards[list(cfg.rewards.keys())[0]],
                                                             c_x = c_x, c_y = c_y, s_x = s_x, s_y=s_y, 
                                                             c_z = c_z, s_z=s_z)

        self.observation_space = spaces.Box(
            low=-2 * self.spread,
            high=2 * self.spread,
            shape=(self.num_moving_points * 3,),
            dtype=np.float32
        )

        # consider using (+-spread) as a bound
        self.action_space = spaces.Box(
            low=-1.,
            high=1.,
            shape=(3,),
            dtype=np.float32,
        )

        self.spread = self.spread
        self.l2_distances = 0

        self.step_n = 0
        self.total_step_n = 0
        self.max_steps = cfg.max_steps
        self.vertex_idx = 0

        self.abs_dist = None
        self.source_centroid = None
        self.canvas_centroid = None
        self.convex_hull_area_source = None
        self.convex_hull_area_canvas = None

        self.done = False
        self.do_render = False

        self.triangle_idx = 0
        self.vertex_idx = 0

        ''' if we want to save transitions, initialize transition matrices. '''
        self.save_trans = cfg.save_transitions
        if self.save_trans:
            # for each step, save t_x,t_y,t_z,c_x,c_y,c_z, and the new triangle vertex positions (3)
            self.transition_matrix = np.zeros((cfg.max_steps, 3, 3))
            # for each step, save, for target_vertex_mask and canvas_vertex_mask, the three vertex points
            self.vertex_idx_transitions = np.zeros((cfg.max_steps, 2, 3))

    def reset(self):
        self.step_n = 0
        self.triangle_idx = 0  # for choosing the triangle to move (useful when we are modelling a shape)
        self.vertex_idx = 0  # for choosing the vertex to move

        # TODO: repeatedly calling init functions is okay with shapes in testing because we only call reset once.
        # TODO: Otherwise this could be a big slowdown factor
        if self.init_type == 'from_shape':
            self.s_x, self.s_y, self.s_z, self.c_x, self.c_y, self.c_z, self.triangles = self.f_inits_s()
            # self.c_x, self.c_y, self.c_z, _, _, _, _ = from_shape(shape_path='./shapes/tr_reg_000.ply')
        else:
            self.s_x, self.s_y, self.s_z, self.triangles = self.f_inits_s()  # inits.zero_init(self.num_points)
            self.c_x, self.c_y, self.c_z, self.triangles = self.f_inits_c()  # inits.zero_init(self.num_points)

        self.num_vertices = len(self.s_x)
        self.marker_size = [60] * self.num_vertices

        self.num_triangles = len(self.triangles)

        # the focused vertex is ALWAYS at pos 0!
        self.canvas_vertex_mask = triangle_neighborhood(self.triangles, self.triangle_idx, self.vertex_idx)  # gives the index of 3 vertices
        self.target_vertex_mask = triangle_neighborhood(self.triangles, self.triangle_idx, self.vertex_idx)
        self.state = states.signed_dist_3D(self.c_x[self.canvas_vertex_mask], self.c_y[self.canvas_vertex_mask], self.c_z[self.canvas_vertex_mask],
                                        self.s_x[self.target_vertex_mask], self.s_y[self.target_vertex_mask], self.s_z[self.target_vertex_mask])

        # self.l2_distances = np.sqrt((selfs_x - self.c_x) ** 2 +
        #                             (self.s_y - self.c_y) ** 2 +
        #                             (self.s_z - self.c_z) ** 2)
        #self.abs_dist, self.source_centroid, self.canvas_centroid, self.convex_hull_area_source, self.convex_hull_area_canvas = compute_metrics3d(self.s_x, self.s_y, self.s_z, self.c_x, self.c_y, self.c_z)

        ''' if we want to store transitions '''
        if self.save_trans:
            # points configuration at 33% of the computation
            self.config_33 = np.zeros((3, len(self.c_x)))
            self.config_66 = np.zeros((3, len(self.c_x)))

        return np.float32(self.state)  # self._get_obs()

    def step(self, actions):
        # Now c_x,c_y,c_z (same with s_) are restricted by the triangle context. Instead of passing the whole vertex-set
        # we only gives the vertices of a single triangle. This should make things faster.
        self.step_n += 1
        self.total_step_n += 1
        new_c_x, new_c_y, new_c_z = self.f_act(actions, self.c_x[self.canvas_vertex_mask], self.c_y[self.canvas_vertex_mask], self.c_z[self.canvas_vertex_mask],
                                                  point_idx=0, neighbors=self.canvas_vertex_mask[1:])

        self.c_x[self.canvas_vertex_mask] = new_c_x
        self.c_y[self.canvas_vertex_mask] = new_c_y
        self.c_z[self.canvas_vertex_mask] = new_c_z

        reward = self.f_reward(c_x=self.c_x[self.canvas_vertex_mask], c_y=self.c_y[self.canvas_vertex_mask], c_z=self.c_z[self.canvas_vertex_mask],
                               s_x=self.s_x[self.target_vertex_mask], s_y=self.s_y[self.target_vertex_mask], s_z=self.s_z[self.target_vertex_mask])
        # TODO: 3 or num_vertices? Computing rewards on an entire shape could be expensive. Going with 3 for now
        # *2 is because we have canvas and target, 3 is for x, y and z coordinates.
        reward = normalize_reward(reward, (self.spread * 2 * 3) ** 2)  # a scalar

        """ START WORKING FOR NEXT STATE """
        if self.step_n % (self.steps_per_vertex + 1) == 0:
            self.vertex_idx+=1
            if self.vertex_idx == 3:
                self.vertex_idx = 0
                self.triangle_idx += 1
                if self.triangle_idx == self.num_triangles:
                    self.triangle_idx = 0

        self.old_canvas_vertex_mask = self.canvas_vertex_mask.copy()
        self.old_target_vertex_mask = self.target_vertex_mask.copy()
        self.canvas_vertex_mask = triangle_neighborhood(self.triangles, self.triangle_idx, self.vertex_idx)  # gives the index of 3 vertices
        self.target_vertex_mask = triangle_neighborhood(self.triangles, self.triangle_idx, self.vertex_idx)

        self.state = states.signed_dist_3D(self.c_x[self.canvas_vertex_mask], self.c_y[self.canvas_vertex_mask],
                                           self.c_z[self.canvas_vertex_mask],
                                           self.s_x[self.target_vertex_mask], self.s_y[self.target_vertex_mask],
                                           self.s_z[self.target_vertex_mask])
        self.done = False
        if self.step_n == self.max_steps:
            self.done = True

        info = {"is_success": False}
        # self.l2_distances = np.sqrt((self.s_x - self.c_x) ** 2 + (
        #     self.s_y - self.c_y) ** 2+ (
        #     self.s_z - self.c_z) ** 2)

        # TODO: This works in testing but not in training. Uncomment and put "if is_testing" condition.
        # TODO: Otherwise make it work in training, too.
        # if self.total_step_n % 500 == 0:
        #     self.abs_dist, self.source_centroid, self.canvas_centroid, self.convex_hull_area_source, self.convex_hull_area_canvas = compute_metrics3d(self.s_x, self.s_y, self.s_z, self.c_x, self.c_y, self.c_z)

        ''' This can be used if we want to introduce a stopping condition'''
        # if np.sum(self.l2_distances < 1) == self.num_points:
        #     done = True
        #     info = {"is_success": True}
        # obs = self._get_obs()

        return np.float32(self.state), reward, self.done, info

    def get_distances(self):
        return self.l2_distances

    def render(self, mode='human', close=False):
        # marker_size = [60] * self.num_vertices
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.axis('off')

        ax.scatter(self.s_x[0], self.s_y[0], self.s_z[0], color='green')# , s=marker_size, color='blue')
        ax.scatter(self.s_x[1:], self.s_y[1:], self.s_z[1:], color='blue')# , s=marker_size, color='yellow')
        ax.scatter(self.c_x[0], self.c_y[0], self.c_z[0], color='orange')
        ax.scatter(self.c_x[1:], self.c_y[1:], self.c_z[1:], color='red')

        ax.set_xlim([-self.spread - 1, self.spread + 1])
        ax.set_ylim([-self.spread - 1, self.spread + 1])
        ax.set_zlim([-self.spread - 1, self.spread + 1])

        ''' this commented code is to show the agent progressing in real time instead of only logging to wandb '''
        #plt.draw()
        # plt.show()
        # plt.pause(0.001)
        ''' until here '''
        plt.ioff()
        image = plot3(fig)
        plt.close()
        return image

    def _get_obs(self) -> Dict[str, Union[int, np.ndarray]]:
        return OrderedDict(
             [
                 ("observation", self.state.copy().astype(np.float32)),
             ]
        )

    def store_transition(self):
        # print(np.concatenate((self.s_x[self.canvas_vertex_mask],
        #                                                       self.s_y[self.canvas_vertex_mask],
        #                                                       self.s_z[self.canvas_vertex_mask],
        #                                                       self.c_x[self.canvas_vertex_mask],
        #                                                       self.c_y[self.canvas_vertex_mask],
        #                                                       self.c_z[self.canvas_vertex_mask])).reshape(6,3))
        self.transition_matrix[self.step_n-1] = np.concatenate((
                                                              self.c_x[self.old_canvas_vertex_mask],
                                                              self.c_y[self.old_canvas_vertex_mask],
                                                              self.c_z[self.old_canvas_vertex_mask])).reshape(3,3)

        self.vertex_idx_transitions[self.step_n-1] = np.concatenate((self.old_target_vertex_mask,
                                                                     self.old_canvas_vertex_mask)).reshape((2,3))

        if self.total_step_n == 66000:
            self.config_33[0] = self.c_x
            self.config_33[1] = self.c_y
            self.config_33[2] = self.c_z

        if self.total_step_n == 132000:
            self.config_66[0] = self.c_x
            self.config_66[1] = self.c_y
            self.config_66[2] = self.c_z


    def save_transitions(self, filename):
        # open a file, where you ant to store the data
        file = open(filename, 'wb')
        #
        # # dump information to that file
        # pickle.dump(self.transition_matrix, file)
        #
        # # close the file
        # file.close()
        #
        # file = open('vertex_masks', 'wb')
        #
        # # dump information to that file
        # pickle.dump(self.vertex_idx_transitions, file)
        #
        # # close the file
        # file.close()

        final_configuration = np.zeros((3, len(self.c_x)))
        final_configuration[0] = self.c_x
        final_configuration[1] = self.c_y
        final_configuration[2] = self.c_z
        # file = open('final_configuration', 'wb')
        # pickle.dump(final_configuration, file)

        transitions = {'transition_matrix': self.transition_matrix, 'vertex_masks': self.vertex_idx_transitions,
                       '33_config': self.config_33, '66_config': self.config_66, 'final_configuration': final_configuration}
        pickle.dump(transitions, file)
        file.close()





