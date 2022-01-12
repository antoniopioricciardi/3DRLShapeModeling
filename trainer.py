import os
import gym
import torch
import numpy as np
import matplotlib.pyplot as plt


from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed

from stable_baselines3 import HerReplayBuffer, SAC, DDPG, PPO
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy

from stable_baselines3.common.logger import configure

from wandb.integration.sb3 import WandbCallback
from callbackutils import WandbTrainCallback
import wandb
from stable_baselines3.common.envs import BitFlippingEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecVideoRecorder, VecMonitor
#from env3d_triangles import CanvasModeling
from env3d_triangles import CanvasModeling
from stable_baselines3.common.env_checker import check_env

from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
import hydra 
import random
from hydra.utils import get_original_cwd, to_absolute_path

def train(model, timesteps):
    # TODO: Sposta, ma prima review della callback esistente
    wandb_callback = WandbTrainCallback()
    # START TRAINING
    model.learn(total_timesteps=timesteps, log_interval=2)#, callback=wandb_callback)