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


def train(cfg, env, model, wandb_logger, learning_rate, model_class, timesteps, model_name, max_steps):

    # # Wrapper for multi-environment
    # def make_env(cfg):
    #     env = CanvasModeling(cfg)
    #     k = check_env(env, warn=True)
    #     env = Monitor(env,LOGS_PATH)  # record stats such as returns
    #     return env
    #
    # # Generate instances
    # if cfg.envs == 1:
    #     env = make_vec_env(make_env,cfg.envs, env_kwargs = {'cfg':cfg})
    #     # env = DummyVecEnv([make_env(**args_env)])
    # else:
    #     env = make_vec_env(make_env,cfg.envs, env_kwargs = {'cfg':cfg})
    #
    # # Wrapper for logging video
    # env = VecVideoRecorder(env, f"videos/{wandb_logger.id}", record_video_trigger=lambda x: x % cfg.log_video_steps == cfg.log_video_steps-1, video_length=max_steps)

    # define the model policy
    policy_kwargs = hydra.utils.instantiate(cfg.policies['model'], **cfg.policies.params)
    # policy_kwargs = dict(
    # features_extractor_class=poly.CustomPerc,
    # features_extractor_kwargs=dict(features_dim=128)
    # )
    # policy_kwargs = dict(activation_fn=torch.nn.ReLU, net_arch=dict(pi=[256, 256, 256], qf=[256, 512, 512, 512]))
    # print(policy_kwargs)
    # policy_kwargs = dict(activation_fn=torch.nn.ReLU, net_arch=dict(pi=[256, 256, 256], qf=[256, 256, 256]))
    #
    # model = model_class("MlpPolicy", env, verbose=1,policy_kwargs=policy_kwargs, tensorboard_log=f"runs/{wandb_logger.id}",
    #                     learning_rate=learning_rate, seed=cfg.seed)# , n_steps = cfg.update_step)
    # policy_kwargs['net_arch'][-1]['pi']

    # TODO: Sposta, ma prima review della callback esistente
    wandb_callback = WandbTrainCallback()
    # START TRAINING
    model.learn(total_timesteps=timesteps, log_interval=2)#, callback=wandb_callback)

    # shutdown the logger
    if wandb_logger is not None:
        wandb_logger.finish()

    # Save
    model.save(os.path.join(MODELS_PATH, model_name))
    # model.save(os.path.join(model_path, cfg.model_name))
   
    # Close Environment
    env.close()