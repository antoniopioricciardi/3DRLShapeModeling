import gym
# matplotlib.use('agg')
import matplotlib.pyplot as plt

from gym import spaces
from neighborhood import *
from metrics3d import *
import states
from collections import OrderedDict
from typing import Any, Dict, Optional, Union
import io
import hydra
import pickle

from inits import *
from misc_utils.normalization import normalize_vector

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
        self.action_type = list(cfg.actions.keys())[0]
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

        # TODO: in alternativa inserisci sigma anche nelle azioni linear, come percentuale di movimento del punto centrale
        if self.action_type == 'gaussian3d':
            self.f_act = lambda actions, c_x, c_y, c_z, point_idx, neighbors: hydra.utils.instantiate(
                cfg.actions[list(cfg.actions.keys())[0]],
                actions=actions, c_x=c_x, c_y=c_y, c_z=c_z,
                spread=self.spread, point_idx=point_idx,
                neighbors=neighbors, neighbors_movement_scale=self.neighbors_movement_scale, sigma=cfg.sigma)
        else:
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
        self.total_timesteps = cfg.total_timesteps
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
        self.is_testing = cfg.is_testing
        if self.save_trans:
            # for each step, save t_x,t_y,t_z,c_x,c_y,c_z, and the new triangle vertex positions (3)
            self.canvas_transition_matrix = np.zeros((cfg.max_steps+1, 3, 3))
            # for each step, save, for target_vertex_mask and canvas_vertex_mask, the three vertex points
            self.vertex_masks_transitions = np.zeros((cfg.max_steps+1, 2, 3))
            self.actions_transitions = np.zeros((cfg.max_steps+1, 3))
            self.states_transitions = np.zeros((cfg.max_steps+1, self.num_moving_points*3))
            self.actions_to_save = np.zeros(3)

    def reset(self):
        self.step_n = 0
        self.triangle_idx = 0  # for choosing the triangle to move (useful when we are modelling a shape)
        self.vertex_idx = 0  # for choosing the vertex to move

        # TODO: repeatedly calling init functions is okay with shapes in testing because we only call reset once.
        # TODO: Otherwise this could be a big slowdown factor

        if self.init_type == 'load_shape_new':
            # target_mesh, target_vert, self.triangles = load_shape('shapes/tr_reg_000_rem.ply', simplify=False, normalize=True)
            target_mesh, target_vert, self.triangles = load_shape('shapes/tr_reg_000_rem.ply', simplify=False, normalize=True)
            self.s_x, self.s_y, self.s_z = get_coordinates(target_vert, self.spread)
            # canv_mesh, canvas_vert, self.canvas_tri = sphere_from_mesh(target_vert, self.triangles)
            canv_mesh, canvas_vert, self.canvas_tri = load_shape('shapes/smpl_base_neutro_rem.ply', simplify=False, normalize=True)
            self.c_x, self.c_y, self.c_z = get_coordinates(canvas_vert, self.spread)

            # print(self.s_x.min(), self.s_x.max(), '--', self.s_y.min(), self.s_y.max(), '--', self.s_z.min(), self.s_z.max())
            # print(self.c_x.min(), self.c_x.max(), '--', self.c_y.min(), self.c_y.max(), '--', self.c_z.min(), self.c_z.max())
            # self.s_x, self.s_y, self.s_z, self.triangles = self.f_inits_s()
            # self.c_x, self.c_y, self.c_z, _ = from_shape(shape_path='./shapes/smpl_base_neutro_rem.ply', normalize=True)
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
        if self.is_testing:
            self.abs_dist, self.source_centroid, self.canvas_centroid, self.convex_hull_area_source, self.convex_hull_area_canvas = compute_metrics3d(self.s_x, self.s_y, self.s_z, self.c_x, self.c_y, self.c_z)

        ''' if we want to store transitions '''
        if self.save_trans:
            # points configuration at 33% of the computation
            self.config_33 = np.zeros((3, len(self.c_x)))
            self.config_66 = np.zeros((3, len(self.c_x)))
            self.target_vertices = np.vstack((self.s_x, self.s_y, self.s_z))
            self.canvas_vertices_transitions = np.zeros((self.max_steps+1, 3, len(self.c_x)))
            # self.target_vertices_transitions = np.zeros((self.max_steps+1, 3, len(self.s_x)))
            # self.target_mesh = target_mesh
            # self.canvas_mesh = canv_mesh

        return np.float32(self.state)  # self._get_obs()

    def step(self, actions):
        self.actions_to_save = actions
        # Now c_x,c_y,c_z (same with s_) are restricted by the triangle context. Instead of passing the whole vertex-set
        # we only gives the vertices of a single triangle. This should make things faster.
        self.step_n += 1
        self.total_step_n += 1
        # new_c_x, new_c_y, new_c_z = self.f_act(actions, self.c_x[self.canvas_vertex_mask], self.c_y[self.canvas_vertex_mask], self.c_z[self.canvas_vertex_mask],
        #                                        point_idx=self.canvas_vertex_mask[0], neighbors=self.canvas_vertex_mask[1:])#, neighbors=self.canvas_vertex_mask[1:])

        new_c_x, new_c_y, new_c_z = self.f_act(actions, self.c_x[self.canvas_vertex_mask],
                                               self.c_y[self.canvas_vertex_mask], self.c_z[self.canvas_vertex_mask],
                                               point_idx=0, neighbors=[1,2])  # , neighbors=self.canvas_vertex_mask[1:])
        # new_c_x, new_c_y, new_c_z = self.f_act(actions, self.c_x[self.canvas_vertex_mask],
        #                                        self.c_y[self.canvas_vertex_mask], self.c_z[self.canvas_vertex_mask],
        #                                        point_idx=self.vertex_idx, neighbors=self.canvas_vertex_mask[1:])  # , neighbors=self.canvas_vertex_mask[1:])

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

        # TODO: Delete old_...
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
        # TODO: Otherwise make it to work in training, too.
        if self.is_testing and self.total_step_n % 500 == 0:
            self.abs_dist, self.source_centroid, self.canvas_centroid, self.convex_hull_area_source, self.convex_hull_area_canvas = compute_metrics3d(self.s_x, self.s_y, self.s_z, self.c_x, self.c_y, self.c_z)

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
        # ax.axis('off')

        if self.is_testing:
            ax.scatter(self.s_x, self.s_y,
                       self.s_z, color='blue')  # , s=marker_size, color='blue')
            ax.scatter(self.s_x[self.canvas_vertex_mask], self.s_y[self.canvas_vertex_mask],
                       self.s_z[self.canvas_vertex_mask], color='green')  # , s=marker_size, color='yellow')
            ax.scatter(self.c_x, self.c_y,
                       self.c_z, color='red')
            ax.scatter(self.c_x[self.canvas_vertex_mask], self.c_y[self.canvas_vertex_mask],
                       self.c_z[self.canvas_vertex_mask], color='red')
        else:
            ax.scatter(self.s_x[self.canvas_vertex_mask][0], self.s_y[self.canvas_vertex_mask][0], self.s_z[self.canvas_vertex_mask][0], color='green')# , s=marker_size, color='blue')
            ax.scatter(self.s_x[self.canvas_vertex_mask][1:], self.s_y[self.canvas_vertex_mask][1:], self.s_z[self.canvas_vertex_mask][1:], color='blue')# , s=marker_size, color='yellow')
            ax.scatter(self.c_x[self.canvas_vertex_mask][0], self.c_y[self.canvas_vertex_mask][0], self.c_z[self.canvas_vertex_mask][0], color='orange')
            ax.scatter(self.c_x[self.canvas_vertex_mask][1:], self.c_y[self.canvas_vertex_mask][1:], self.c_z[self.canvas_vertex_mask][1:], color='red')

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
        # contains the the status of the whole set of vertices at that timestep
        self.canvas_vertices_transitions[self.step_n][0] = self.c_x
        self.canvas_vertices_transitions[self.step_n][1] = self.c_y
        self.canvas_vertices_transitions[self.step_n][2] = self.c_z
        # contains the set of vertices modified during current step
        self.canvas_transition_matrix[self.step_n] = np.concatenate((
                                                              self.c_x[self.canvas_vertex_mask],
                                                              self.c_y[self.canvas_vertex_mask],
                                                              self.c_z[self.canvas_vertex_mask])).reshape(3,3)

        self.vertex_masks_transitions[self.step_n] = np.concatenate((self.target_vertex_mask,
                                                                     self.canvas_vertex_mask)).reshape((2,3))
        self.actions_transitions[self.step_n] = self.actions_to_save
        self.states_transitions[self.step_n] = self.state
        # save point configuration as 33 and 66 % of the execution
        if self.total_step_n == (self.total_timesteps//0.33):
            self.config_33[0] = self.c_x
            self.config_33[1] = self.c_y
            self.config_33[2] = self.c_z

        if self.total_step_n == (self.total_timesteps//0.66):
            self.config_66[0] = self.c_x
            self.config_66[1] = self.c_y
            self.config_66[2] = self.c_z

    def save_transitions(self, filename):
        # open a file, where you ant to store the data
        file = open(filename, 'wb')

        final_configuration = np.zeros((3, len(self.c_x)))
        final_configuration[0] = self.c_x
        final_configuration[1] = self.c_y
        final_configuration[2] = self.c_z

        transitions = {'target_vertices': self.target_vertices, 'target_triangles': self.triangles, 'canvas_triangles': self.canvas_tri,
                       'canvas_vertices_transitions': self.canvas_vertices_transitions,
            'states_transitions': self.states_transitions, 'actions_transitions': self.actions_transitions,
            'canvas_transition_matrix': self.canvas_transition_matrix, 'vertex_masks': self.vertex_masks_transitions,
                       '33_config': self.config_33, '66_config': self.config_66, 'final_configuration': final_configuration}
        pickle.dump(transitions, file)
        file.close()



