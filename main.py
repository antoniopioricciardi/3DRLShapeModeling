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

MODELS_PATH = 'models'
VIDEOS_PATH = 'videos'
LOGS_PATH = 'logs'

if not os.path.exists(MODELS_PATH):
    os.mkdir(MODELS_PATH)
if not os.path.exists(VIDEOS_PATH):
    os.mkdir(VIDEOS_PATH)
if not os.path.exists(LOGS_PATH):
    os.mkdir(LOGS_PATH)

def run(cfg):

    # print(to_absolute_path(''))  # this gives src path even when running with hydra
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    set_random_seed(cfg.seed)
    # torch.use_deterministic_algorithms(mode=True)  # probably to slow down code, based on functions we use, might not always work
    # Settings
    num_triangles = cfg.num_triangles
    steps_per_vertex = cfg.steps_per_vertex
    # num_points = cfg.num_points
    spread = cfg.spread 
    max_steps = cfg.max_steps
    args_env = cfg

    model_class = SAC # PPO
    timesteps = cfg.total_timesteps
    learning_rate = cfg.learning_rate # 1e-2 is the best up to 4pt
    experiment_name = 'TRAIN:maxsteps_' + str(max_steps) + '-numpoints_' + str(num_triangles) + \
                 '-timesteps_' + str(timesteps) + '-lr_' + str(learning_rate)

    # model_name = 'circle_env-max_steps_' + str(max_steps) + '-numpoints_' + str(num_triangles) + \
    #              '-timesteps_' + str(timesteps) + '-lr_' + str(learning_rate)


    # TODO: Hydra seems to override the "model.save" below, hence this is useless for now.
    # model_path_list = [cfg.model_name, 'seed_'+ str(cfg.seed),
    #                    'total-timesteps' + str(timesteps),'max-steps_' + str(max_steps),
    #                    'spread_' + str(spread),
    #                    'lr_' + str(learning_rate), 'sigma_' + str(cfg.sigma), 'neighbors-movement-scale_' + str(cfg.neighbors_movement_scale)
    #                    ]

    # model_path = MODELS_PATH
    # for dir_name in model_path_list:
    #     model_path = os.path.join(model_path, dir_name)
    #     os.makedirs(model_path)

    # TODO: until here

    model_path_list = [cfg.model_name, '-seed_' + str(cfg.seed),
                       # '-totaltimesteps' + str(timesteps),'-maxsteps_' + str(max_steps),
                       '-spread_' + str(spread),
                       '-lr_' + str(learning_rate), '-sigma_' + str(cfg.sigma),
                       '-neighborsmovementscale_' + str(cfg.neighbors_movement_scale)
                       ]

    model_name = ''
    for dir_name in model_path_list:
        model_name += dir_name


    # Initialise a W&B run
    wandb_logger = wandb.init(
        name=model_name,
        project="3Dmodeling_gaussian",
        config=cfg,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=True,  # auto-upload the videos of agents playing the game
        save_code=True,  # optional
    )

    wandb_callback = WandbCallback(
        gradient_save_freq=5,
        model_save_path=f"models/{wandb_logger.id}",
        verbose=2,
    )

    # Wrapper for multi-environment
    def make_env(cfg):
        env = CanvasModeling(cfg)
        k = check_env(env, warn=True)
        env = Monitor(env,LOGS_PATH)  # record stats such as returns
        return env
    
    # Generate instances
    if cfg.envs == 1:
        env = make_vec_env(make_env,cfg.envs, env_kwargs = {'cfg':args_env})
        # env = DummyVecEnv([make_env(**args_env)])
    else:
        env = make_vec_env(make_env,cfg.envs, env_kwargs = {'cfg':args_env})
    
    # Wrapper for logging video
    env = VecVideoRecorder(env, f"videos/{wandb_logger.id}", record_video_trigger=lambda x: x % cfg.log_video_steps == cfg.log_video_steps-1, video_length=max_steps)

    # define the model policy
    policy_kwargs = hydra.utils.instantiate(cfg.policies['model'], **cfg.policies.params)
    # policy_kwargs = dict(
    # features_extractor_class=poly.CustomPerc,
    # features_extractor_kwargs=dict(features_dim=128)
    # )
    # policy_kwargs = dict(activation_fn=torch.nn.ReLU, net_arch=dict(pi=[256, 256, 256], qf=[256, 512, 512, 512]))
    # print(policy_kwargs)
    policy_kwargs = dict(activation_fn=torch.nn.ReLU, net_arch=dict(pi=[256, 256, 256], qf=[256, 256, 256]))

    model = model_class("MlpPolicy", env, verbose=1,policy_kwargs=policy_kwargs, tensorboard_log=f"runs/{wandb_logger.id}",
                        learning_rate=learning_rate, seed=cfg.seed)# , n_steps = cfg.update_step)
    # policy_kwargs['net_arch'][-1]['pi']

    # TODO: Sposta, ma prima review della callback esistente
    wandb_callback = WandbTrainCallback()
    # START TRAINING
    env.seed(cfg.seed)
    model.learn(total_timesteps=timesteps, log_interval=2)#, callback=wandb_callback)

    # shutdown the logger
    if wandb_logger is not None:
        wandb_logger.finish()

    # Save
    model.save(os.path.join(MODELS_PATH, model_name))
    # model.save(os.path.join(model_path, cfg.model_name))
   
    # Close Environment
    env.close()


@hydra.main(config_path="confs", config_name="configs.yaml")
def main(cfg: DictConfig) -> None:
    OmegaConf.resolve(cfg)
    print(OmegaConf.to_yaml(cfg,resolve=True))
    run(cfg)

if __name__ == '__main__':
    main()
