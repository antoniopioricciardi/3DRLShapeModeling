import gym
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('agg')

from gym import spaces
from neighborhood import *
from metrics3d import compute_metrics3d
from inits import *
import io
import hydra
import pickle
from misc_utils.shapes.operations import compute_triangle_triangle_adjacency_matrix_igl

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


class CanvasModelingTest:
    metadata = {
        "render.modes": ["human", "rgb_array", "state_pixels"],
    }
    def __init__(self, cfg):
        self.num_points = cfg.num_points
        self.neighborhood_size = cfg.neighborhood_size
        self.steps_per_vertex = cfg.steps_per_vertex
        self.spread = cfg.spread
        self.neighbors_movement_scale = cfg.neighbors_movement_scale
        self.max_steps = cfg.max_steps
        self.is_testing = cfg.is_testing
        self.n_hops = 0

        self.step_n = 0
        self.total_step_n = 0
        self.vertex_idx = 0
        self.neighborhood_mask = np.arange(self.num_points)
        self.done = False
        self.marker_size = 70

        self.f_inits_t = lambda: hydra.utils.instantiate(cfg.inits_t[list(cfg.inits_t.keys())[0]],
                                                         num_points=self.num_points)
        self.f_inits_c = lambda: hydra.utils.instantiate(cfg.inits_c[list(cfg.inits_c.keys())[0]],
                                                         num_points=self.num_points)

        self.f_act = lambda actions, c_x, c_y, c_z, point_idx, neighbors: hydra.utils.instantiate(
            cfg.actions[list(cfg.actions.keys())[0]],
            actions=actions, c_x=c_x, c_y=c_y, c_z=c_z,
            spread=self.spread, point_idx=point_idx,
            neighbors=neighbors, neighbors_movement_scale=self.neighbors_movement_scale)

        self.f_reward = lambda c_x, c_y, c_z, t_x, t_y, t_z: hydra.utils.instantiate(
            cfg.rewards[list(cfg.rewards.keys())[0]],
            c_x=c_x, c_y=c_y, c_z=c_z, t_x=t_x, t_y=t_y, t_z=t_z)

        # TODO: normalize obs space to be in -1,1 (hence, what before was in -10/10 will be -1/1)
        self.observation_space = spaces.Box(
            low=-2 * self.spread,
            high=2 * self.spread,
            shape=(self.num_points * 3,),  # x and y coordinates for each point in target - canvas pts
            dtype=np.float32
        )

        # consider using (+-spread) as a bound
        self.action_space = spaces.Box(
            low=-1.,
            high=1.,
            shape=(3,),
            dtype=np.float32,
        )

        ''' if we want to save transitions, initialize transition matrices. '''
        self.save_trans = cfg.save_transitions
        self.is_testing = cfg.is_testing
        if self.save_trans:
            # for each step, save t_x,t_y,t_z,c_x,c_y,c_z, and the new triangle vertex positions (3)
            self.canvas_transition_matrix = np.zeros((cfg.max_steps+1, 3), dtype=object) # , 3))
            # for each step, save, for target_vertex_mask and canvas_vertex_mask, the three vertex points
            self.actions_transitions = np.zeros((cfg.max_steps+1, 3))
            self.states_transitions = np.zeros(cfg.max_steps+1, dtype=object) # self.num_moving_points*3))
            self.actions_to_save = np.zeros(3)

        self.target_mesh, self.target_vert, self.target_tri = load_shape('shapes/tr_reg_000_rem.ply', simplify=False,
                                                          normalize=True)
        self.shape_t_x, self.shape_t_y, self.shape_t_z = get_coordinates(self.target_vert, self.spread)
        # canv_mesh, canvas_vert, self.canvas_tri = sphere_from_mesh(target_vert, self.triangles)
        self.canvas_mesh, self.canvas_vert, self.canvas_tri = load_shape('shapes/smpl_base_neutro_rem.ply', simplify=False,
                                                          normalize=True)
        self.shape_c_x, self.shape_c_y, self.shape_c_z = get_coordinates(self.canvas_vert, self.spread)
        self.adj_tri, self.adj_edge = compute_triangle_triangle_adjacency_matrix_igl(self.canvas_tri)

        self.triangle_idx = 0

    def reset(self, neighborhood_size):
        self.n_rolls = 0
        self.step_n = 0
        self.vertex_idx = 0  # for choosing the vertex to move
        # self.triangle_idx = 0
        self.done = False
        # self.t_x, self.t_y, self.t_z = self.f_inits_t()
        # self.c_x, self.c_y, self.c_z = self.f_inits_c()
        self.neighborhood_size = neighborhood_size
        if neighborhood_size == 3:
            self.n_hops = 0
        if neighborhood_size == 6:
            self.n_hops = 1
        if neighborhood_size == 12:
            self.n_hops = 2
        # canvas_vertex_mask = triangle_neighborhood(self.canvas_tri, self.triangle_idx, self.vertex_idx)
        self.vertex_mask = vertex_mask_from_triangle_adjacency(self.canvas_tri, self.triangle_idx, self.adj_tri, self.n_hops,
                                                                self.neighborhood_size)

        self.neighborhood_mask = np.arange(self.num_points)
        self.old_neighborhood_mask = self.neighborhood_mask.copy()

        self.t_x = self.shape_t_x[self.vertex_mask]
        self.t_y = self.shape_t_y[self.vertex_mask]
        self.t_z = self.shape_t_z[self.vertex_mask]
        self.c_x = self.shape_c_x[self.vertex_mask]
        self.c_y = self.shape_c_y[self.vertex_mask]
        self.c_z = self.shape_c_z[self.vertex_mask]

        self.state = np.float32(np.concatenate((self.t_x-self.c_x, self.t_y-self.c_y, self.t_z-self.c_z)))

        if self.is_testing:
            self.abs_dist, self.source_centroid, self.canvas_centroid, self.convex_hull_area_source, self.convex_hull_area_canvas = compute_metrics3d(
                self.shape_t_x, self.shape_t_y, self.shape_t_z, self.shape_c_x, self.shape_c_y, self.shape_c_z)
        self.triangle_idx += 1
        if self.triangle_idx == len(self.canvas_tri):
            self.triangle_idx = 0

        ''' if we want to store transitions '''
        if self.save_trans:
            # points configuration at 33% of the computation
            self.config_33 = np.zeros((3, len(self.c_x)))
            self.config_66 = np.zeros((3, len(self.c_x)))
            self.target_vertices = np.vstack((self.shape_t_x, self.shape_t_y, self.shape_t_z))
            self.neighborhood_mask_transitions = np.zeros(self.max_steps+1, dtype=object)
            self.vertex_masks_transitions = np.zeros(self.max_steps+1, dtype=object)
            self.canvas_vertices_transitions = np.zeros((self.max_steps + 1, 3), dtype=object)# self.neighborhood_size))
        return self.state

    def advance_neighborhood(self, neighborhood_size):
        """
        Similar to reset, but avoid resetting some values breaking the simulation.
        Might find a prettier way of doing this without too much code reusing
        """
        self.n_rolls = 0
        self.vertex_idx = 0  # for choosing the vertex to move
        # self.triangle_idx = 0
        # self.t_x, self.t_y, self.t_z = self.f_inits_t()
        # self.c_x, self.c_y, self.c_z = self.f_inits_c()
        self.neighborhood_size = neighborhood_size
        if neighborhood_size == 3:
            self.n_hops = 0
        if neighborhood_size == 6:
            self.n_hops = 1
        if neighborhood_size == 12:
            self.n_hops = 2
        # canvas_vertex_mask = triangle_neighborhood(self.canvas_tri, self.triangle_idx, self.vertex_idx)
        self.vertex_mask = vertex_mask_from_triangle_adjacency(self.canvas_tri, self.triangle_idx, self.adj_tri, self.n_hops,
                                                                self.neighborhood_size)
        self.neighborhood_mask = np.arange(self.num_points)
        self.old_neighborhood_mask = self.neighborhood_mask.copy()

        self.t_x = self.shape_t_x[self.vertex_mask]
        self.t_y = self.shape_t_y[self.vertex_mask]
        self.t_z = self.shape_t_z[self.vertex_mask]
        self.c_x = self.shape_c_x[self.vertex_mask]
        self.c_y = self.shape_c_y[self.vertex_mask]
        self.c_z = self.shape_c_z[self.vertex_mask]

        self.state = np.float32(np.concatenate((self.t_x-self.c_x, self.t_y-self.c_y, self.t_z-self.c_z)))

        self.triangle_idx += 1
        if self.triangle_idx == len(self.canvas_tri):
            self.triangle_idx = 0

        return self.state

    def step(self, actions):
        self.step_n += 1
        self.total_step_n += 1
        sweep_completed = False

        # print(self.c_x)
        # print(self.c_x, self.neighborhood_mask, actions[0])

        new_c_x, new_c_y, new_c_z = self.f_act(actions, self.c_x.copy(), self.c_y.copy(), self.c_z.copy(),
                                               point_idx=self.neighborhood_mask[0], neighbors=self.neighborhood_mask[1:])
        # print(self.c_x, self.neighborhood_mask, actions[0], new_c_x)
        self.c_x = new_c_x
        self.c_y = new_c_y
        self.c_z = new_c_z

        reward = self.f_reward(c_x=self.c_x, c_y=self.c_y, c_z=self.c_z,
                               t_x=self.t_x, t_y=self.t_y, t_z=self.t_z)
        # reward = normalize_reward(reward, (self.spread * 2 * self.num_points) ** 2)  # a scalar
        if self.step_n % (self.steps_per_vertex + 1) == 0:
            self.old_neighborhood_mask = self.neighborhood_mask.copy()
            self.neighborhood_mask = np.roll(self.neighborhood_mask, -1)
            self.n_rolls += 1
            if self.n_rolls == len(self.neighborhood_mask)*5:
                sweep_completed = True
            """ part below unused for now """
            self.vertex_idx += 1
            if self.vertex_idx == self.num_points:
                self.vertex_idx = 0

        # self.canvas_vertex_mask = triangle_neighborhood(self.triangles, self.triangle_idx, self.vertex_idx)  # gives the index of 3 vertices
        # self.target_vertex_mask = triangle_neighborhood(self.triangles, self.triangle_idx, self.vertex_idx)

        self.state = np.float32(np.concatenate((self.t_x[self.neighborhood_mask]-self.c_x[self.neighborhood_mask],
                                                self.t_y[self.neighborhood_mask]-self.c_y[self.neighborhood_mask],
                                                self.t_z[self.neighborhood_mask]-self.c_z[self.neighborhood_mask])))

        self.shape_c_x[self.vertex_mask] = self.c_x
        self.shape_c_y[self.vertex_mask] = self.c_y
        self.shape_c_z[self.vertex_mask] = self.c_z
        if self.is_testing and self.total_step_n % 500 == 0:
            self.abs_dist, self.source_centroid, self.canvas_centroid, self.convex_hull_area_source, self.convex_hull_area_canvas = compute_metrics3d(
                self.shape_t_x, self.shape_t_y, self.shape_t_z, self.shape_c_x, self.shape_c_y, self.shape_c_z)

        info = {"is_success": False, "sweep_completed": sweep_completed}
        if self.total_step_n == self.max_steps:
            self.done = True
        return self.state, reward, self.done, info

    def render(self, mode='human', close=False):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # ax.axis('off')

        ax.scatter(self.shape_t_x, self.shape_t_y, self.shape_t_z, color='blue', s=self.marker_size)
        ax.scatter(self.shape_t_x[self.vertex_mask], self.shape_t_y[self.vertex_mask], self.shape_t_z[self.vertex_mask],
                   color='green', s=self.marker_size)
        ax.scatter(self.shape_c_x, self.shape_c_y, self.shape_c_z, color='orange')
        ax.scatter(self.shape_c_x[self.vertex_mask], self.shape_c_y[self.vertex_mask], self.shape_c_z[self.vertex_mask],
                   color='red')

        ax.set_xlim([-self.spread - 1, self.spread + 1])
        ax.set_ylim([-self.spread - 1, self.spread + 1])
        ax.set_zlim([-self.spread - 1, self.spread + 1])

        ''' this commented code is for showing the agent progressing in real time instead of only logging to wandb '''
        #plt.draw()
        # plt.show()
        # plt.pause(0.001)
        ''' until here '''
        plt.ioff()
        image = plot3(fig)
        plt.close()
        return image

    def store_transition(self):
        self.neighborhood_mask_transitions[self.step_n] = self.neighborhood_mask
        # contains the the status of the whole set of vertices at that timestep
        self.canvas_vertices_transitions[self.step_n][0] = self.c_x
        self.canvas_vertices_transitions[self.step_n][1] = self.c_y
        self.canvas_vertices_transitions[self.step_n][2] = self.c_z
        # contains the set of vertices modified during current step
        self.canvas_transition_matrix[self.step_n] = (self.c_x, self.c_y, self.c_z)
        # np.concatenate((
        #                                                       self.c_x[self.canvas_vertex_mask],
        #                                                       self.c_y[self.canvas_vertex_mask],
        #                                                       self.c_z[self.canvas_vertex_mask])).reshape(3,3)

        self.vertex_masks_transitions[self.step_n] = self.vertex_mask
        # self.vertex_masks_transitions[self.step_n] = np.concatenate((self.target_vertex_mask,
        #                                                              self.canvas_vertex_mask)).reshape((2,3))
        self.actions_transitions[self.step_n] = self.actions_to_save
        self.states_transitions[self.step_n] = self.state
        # save point configuration as 33 and 66 % of the execution
        if self.total_step_n == (self.max_steps//0.33):
            self.config_33[0] = self.c_x
            self.config_33[1] = self.c_y
            self.config_33[2] = self.c_z

        if self.total_step_n == (self.max_steps//0.66):
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

        transitions = {'target_vertices': self.target_vertices, 'target_triangles': self.target_tri, 'canvas_triangles': self.canvas_tri,
                       'canvas_vertices_transitions': self.canvas_vertices_transitions,
            'states_transitions': self.states_transitions, 'actions_transitions': self.actions_transitions,
            'canvas_transition_matrix': self.canvas_transition_matrix, 'vertex_masks': self.vertex_masks_transitions,
           'neighborhood_mask_transitions': self.neighborhood_mask_transitions, '33_config': self.config_33,
           '66_config': self.config_66, 'final_configuration': final_configuration}
        pickle.dump(transitions, file)
        file.close()
