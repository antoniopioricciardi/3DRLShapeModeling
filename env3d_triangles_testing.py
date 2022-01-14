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


obs = reset


while not done
    a = modeli.predict(obs)
    obs = envi.step(a)



class CanvasModelingTest():
    metadata = {
        "render.modes": ["human", "rgb_array", "state_pixels"],
    }
    def __init__(self, cfg):
        super(CanvasModelingTest, self).__init__()
        self.num_points = cfg.num_points
        self.neighborhood_size = cfg.neighborhood_size
        self.steps_per_vertex = cfg.steps_per_vertex
        self.spread = cfg.spread
        self.neighbors_movement_scale = cfg.neighbors_movement_scale
        self.max_steps = cfg.max_steps
        self.is_testing = cfg.is_testing

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
            spread=self.spread,point_idx=point_idx,
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

        self.triangle_idx = 0

        # target_mesh, target_vert, self.triangles = load_shape('shapes/tr_reg_000_rem.ply', simplify=False, normalize=True)
        target_mesh, target_vert, self.target_tri = load_shape('shapes/tr_reg_000_rem.ply', simplify=False,
                                                              normalize=True)
        self.t_x, self.t_y, self.t_z = get_coordinates(target_vert, self.spread)
        # canv_mesh, canvas_vert, self.canvas_tri = sphere_from_mesh(target_vert, self.triangles)
        canvas_mesh, canvas_vert, self.canvas_tri = load_shape('shapes/smpl_base_neutro_rem.ply', simplify=False,
                                                             normalize=True)
        self.c_x, self.c_y, self.c_z = get_coordinates(canvas_vert, self.spread)

        # print(self.s_x.min(), self.s_x.max(), '--', self.s_y.min(), self.s_y.max(), '--', self.s_z.min(), self.s_z.max())
        # print(self.c_x.min(), self.c_x.max(), '--', self.c_y.min(), self.c_y.max(), '--', self.c_z.min(), self.c_z.max())
        # self.s_x, self.s_y, self.s_z, self.triangles = self.f_inits_s()
        # self.c_x, self.c_y, self.c_z, _ = from_shape(shape_path='./shapes/smpl_base_neutro_rem.ply', normalize=True)

    def reset(self, neighborhood_size):
        self.neighborhood_size = neighborhood_size
        self.step_n = 0
        self.vertex_idx = 0  # for choosing the vertex to move
        self.triangle_idx = 0
        self.done = False
        # self.t_x, self.t_y, self.t_z = self.f_inits_t()
        # self.c_x, self.c_y, self.c_z = self.f_inits_c()

        canvas_vertex_mask = triangle_neighborhood(self.canvas_tri, self.triangle_idx, self.vertex_idx)
        self.neighborhood_mask = np.zeros(self.neighborhood_size)

        self.old_neighborhood_mask = self.neighborhood_mask.copy()

        self.state = np.float32(np.concatenate((self.t_x-self.c_x, self.t_y-self.c_y, self.t_z-self.c_z)))

        if self.is_testing:
            self.abs_dist, self.source_centroid, self.canvas_centroid, self.convex_hull_area_source, self.convex_hull_area_canvas = compute_metrics3d(
                self.t_x, self.t_y, self.t_z, self.c_x, self.c_y, self.c_z)

        return self.state

    def step(self, actions):
        self.step_n += 1
        # print(self.c_x)
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
            """ part below unused for now """
            self.vertex_idx += 1
            if self.vertex_idx == self.num_points:
                self.vertex_idx = 0

        # self.canvas_vertex_mask = triangle_neighborhood(self.triangles, self.triangle_idx, self.vertex_idx)  # gives the index of 3 vertices
        # self.target_vertex_mask = triangle_neighborhood(self.triangles, self.triangle_idx, self.vertex_idx)

        self.state = np.float32(np.concatenate((self.t_x[self.neighborhood_mask]-self.c_x[self.neighborhood_mask],
                                                self.t_y[self.neighborhood_mask]-self.c_y[self.neighborhood_mask],
                                                self.t_z[self.neighborhood_mask]-self.c_z[self.neighborhood_mask])))

        if self.is_testing and self.total_step_n % 500 == 0:
            self.abs_dist, self.source_centroid, self.canvas_centroid, self.convex_hull_area_source, self.convex_hull_area_canvas = compute_metrics3d(
                self.t_x, self.t_y, self.t_z, self.c_x, self.c_y, self.c_z)

        sweep_completed = False
        if ...:
            sweep_completed = True
        info = {"is_success": False, "sweep_completed": sweep_completed}
        if self.step_n == self.max_steps:
            self.done = True
        return self.state, reward, self.done, info

    def render(self, mode='human', close=False):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # ax.axis('off')

        ax.scatter(self.t_x[self.old_neighborhood_mask][0], self.t_y[self.old_neighborhood_mask][0], self.t_z[self.old_neighborhood_mask][0], color='green', s=self.marker_size)
        ax.scatter(self.t_x[self.old_neighborhood_mask][1:], self.t_y[self.old_neighborhood_mask][1:], self.t_z[self.old_neighborhood_mask][1:], color='blue', s=self.marker_size)
        ax.scatter(self.c_x[self.old_neighborhood_mask][0], self.c_y[self.old_neighborhood_mask][0], self.c_z[self.old_neighborhood_mask][0], color='orange')
        ax.scatter(self.c_x[self.old_neighborhood_mask][1:], self.c_y[self.old_neighborhood_mask][1:], self.c_z[self.old_neighborhood_mask][1:], color='red')

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
