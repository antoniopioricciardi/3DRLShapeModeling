import numpy as np

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from wandb.integration.sb3 import WandbCallback
import gym

class WandbTrainCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """
    def __init__(self, verbose=0):
        super(WandbTrainCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        # TODO: this gets called after reset is called, this is a problem.
        is_reset = self.training_env.get_attr("done")[0]

        if self.training_env.get_attr("total_step_n")[0] % 500 == 0:
            # print(self.training_env.get_attr("l2_distances")[0])  # [0] because get_attr returns an array with values
            # Log scalar value (here a random variable)
            # self.logger.record('l2_distances', self.training_env.get_attr("l2_distances")[0])
            self.logger.record('convex hull area diff', abs(self.training_env.get_attr("convex_hull_area_source")[0] - self.training_env.get_attr("convex_hull_area_canvas")[0]))
            self.logger.record('absolute_dist', self.training_env.get_attr("abs_dist")[0])

            self.logger.record('centroid x difference', self.training_env.get_attr("source_centroid")[0][0] - self.training_env.get_attr("canvas_centroid")[0][0])
            self.logger.record('centroid y difference', self.training_env.get_attr("source_centroid")[0][1] - self.training_env.get_attr("canvas_centroid")[0][1])

        # self.logger.record('source centroid', self.training_env.get_attr("source_centroid")[0])
        # self.logger.record('canvas centroid', self.training_env.get_attr("canvas_centroid")[0])
        return True



class WandbTestCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """
    def __init__(self, verbose=0):
        super(WandbTestCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        # print(self.training_env.get_attr("l2_distances")[0])  # [0] because get_attr returns an array with values
        # Log scalar value (here a random variable)
        # self.logger.record('l2_distances', self.training_env.get_attr("l2_distances")[0])
        self.logger.record('convex hull area diff', abs(self.training_env.get_attr("convex_hull_area_source")[0] - self.training_env.get_attr("convex_hull_area_canvas")[0]))
        self.logger.record('absolute_dist', self.training_env.get_attr("abs_dist")[0])
        self.logger.record('source centroid', self.training_env.get_attr("source_centroid")[0])
        self.logger.record('canvas centroid', self.training_env.get_attr("canvas_centroid")[0])
        return True
