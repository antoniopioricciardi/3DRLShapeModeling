import os
import gym
import torch
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed

from stable_baselines3 import HerReplayBuffer, SAC, DDPG, PPO
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy

from stable_baselines3.common.logger import configure

from wandb.integration.sb3 import WandbCallback
from callbackutils import WandbTestCallback
import wandb
from stable_baselines3.common.envs import BitFlippingEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecVideoRecorder, VecMonitor
# from env3d_triangles import CanvasModeling
from env3d_gaussian import CanvasModeling
from stable_baselines3.common.env_checker import check_env

from omegaconf import DictConfig, OmegaConf

from rich.console import Console
from rich.table import Table

from hydra.core.hydra_config import HydraConfig
from hydra.utils import get_original_cwd, to_absolute_path

import hydra
import random
import imageio

import poly
import os

# TODO: need to be able to use other correspondences and distances oher that the chamfer. Create a top-layer function to use in env2d.

# TODO: clean, or fix outputs and models paths from the folders created during each run


def test(cfg, wandb_logger, model_class, model_name, save_animation_gif, ):
    # Wrapper for multi-environment
    def make_env(cfg):
        env = CanvasModeling(cfg)
        # k = check_env(env, warn=True)
        # env = Monitor(env,LOGS_PATH)  # record stats such as returns
        return env
    
    # Generate instances
    # if cfg.envs == 1:
    #     env = make_vec_env(make_env,cfg.envs, env_kwargs = {'cfg':args_env})
    #
    #     #env = DummyVecEnv([make_env(**args_env)])
    # else:
    #     env = make_vec_env(make_env,cfg.envs, env_kwargs = {'cfg':args_env})
    
    # Wrapper for logging video

    env = make_env(cfg)
    # env = gym.wrappers.Monitor(env, './videos/' + model_name, force=True)

    # from gym.wrappers.monitoring import video_recorder
    # vid = video_recorder.VideoRecorder(env, path="./videos/vid.mp4")
    # env = VecVideoRecorder(env, f"videos/{wandb_logger.id}", record_video_trigger=lambda x: x >= 0, video_length=max_steps)

    # env = VecVideoRecorder(env, f"videos/{wandb_logger.id}", record_video_trigger=lambda x: x % cfg.log_video_steps == cfg.log_video_steps-1, video_length=max_steps)

    # define the model policy
    # policy_kwargs = hydra.misc_utils.instantiate(cfg.policies['model'], **cfg.policies.params)
    # policy_kwargs = dict(
    # features_extractor_class=poly.CustomPerc,
    # features_extractor_kwargs=dict(features_dim=128)
    # )
    # policy_kwargs = dict(activation_fn=torch.nn.ReLU, net_arch=dict(pi=[256, 256, 256], qf=[256, 512, 512, 512]))
    policy_kwargs = dict(activation_fn=torch.nn.ReLU, net_arch=dict(pi=[256, 256, 256], qf=[256, 256, 256]), use_sde=False)

    # model = model_class("MultiInputPolicy", env, verbose=1,policy_kwargs=policy_kwargs, tensorboard_log=f"runs/{wandb_logger.id}",
    #                     learning_rate=learning_rate, n_steps = cfg.update_step)

    model = model_class.load(os.path.join(MODELS_PATH, model_name), env=env, policy_kwargs=policy_kwargs, tensorboard_log=f"runs/{wandb_logger.id}")

    # policy_kwargs['net_arch'][-1]['pi']
    # START TRAINING
    # model.learn(total_timesteps=timesteps, log_interval=2, callback=wandb_callback)

    images = []
    obs = env.reset()
    env.store_transition()

    if save_animation_gif:
        img = model.env.render(mode='rgb_array')    # print(env.l2_distances, 'obs:', obs)
    steps_list = []
    dists_1 = []
    dists_2 = []
    reward_list = []
    reward_list.append(0)
    area_diff = 0
    # for i in range(3000):
    done = False
    test_score = 0
    i = 0
    tqdm_interval_update = 500  # if we update tqdm too often it will strongly slow down our code

    import time

    start = time.time()
    with tqdm(total=cfg.max_steps) as pbar:
        while not done:
            i += 1
            if i % tqdm_interval_update == 0:
                pbar.update(tqdm_interval_update)
            # env.render()
            steps_list.append(i)
            # dists_1.append(env.l2_distances[0])
            # dists_2.append(env.l2_distances[1])
            action, _states = model.predict(obs, deterministic=True)
            obs, rewards, done, info = env.step(action)
            env.store_transition()
            # vid.capture_frame()
            if cfg.is_testing and env.total_step_n % 500 == 0:

                area_diff = abs(env.convex_hull_area_source - env.convex_hull_area_canvas)
                abs_dist = env.abs_dist
                source_centroid = env.source_centroid
                canvas_centroid = env.canvas_centroid
                centroid_x_diff = source_centroid[0] - canvas_centroid[0]
                centroid_y_diff = source_centroid[1] - canvas_centroid[1]
                centroid_z_diff = source_centroid[2] - canvas_centroid[2]
                # area_diff = abs(env.get_attr("convex_hull_area_source")[0] - env.get_attr("convex_hull_area_canvas")[0])
                # abs_dist = env.get_attr("abs_dist")[0]
                # source_centroid = env.get_attr("source_centroid")[0]
                # canvas_centroid = env.get_attr("canvas_centroid")[0]
                # centroid_x_diff = source_centroid[0] - canvas_centroid[0]
                # centroid_y_diff = source_centroid[1] - canvas_centroid[1]

                wandb.log({"area difference": area_diff})
                wandb.log({"absolute distance between shapes": abs_dist})
                wandb.log({"centroid x difference": centroid_x_diff})
                wandb.log({"centroid y difference": centroid_y_diff})
                wandb.log({"centroid z difference": centroid_z_diff})


            # wandb.log({"source centroid": source_centroid})
            # wandb.log({"canvas centroid": canvas_centroid})


            #TODO: need all(done) if we are working with >1 env

            # done = all(done)
            reward_list.append(rewards)
            if save_animation_gif:
                if (i % 150) == 0:
                    images.append(img)
                    img = model.env.render(mode='rgb_array')
            test_score += rewards

    end = time.time() - start
    # print(end)
    if not os.path.exists(res_path):
        os.mkdir(res_path)
    res_path = os.path.join(res_path, 'transitions')
    if not os.path.exists(res_path):
        os.mkdir(res_path)
    env.save_transitions(os.path.join(res_path, model_name))
    env.close()

    # TODO: crea image_path e image_name così da non richiamare continuamente cfg...
    if shape_metrics.get(shape_name) is None:
        shape_metrics[shape_name] = dict()
        shape_metrics[shape_name]['area_diff'] = [area_diff]
        shape_metrics[shape_name]['abs_dist'] = [abs_dist]
        shape_metrics[shape_name]['centr_x_difference'] = [centroid_x_diff]
        shape_metrics[shape_name]['centr_y_difference'] = [centroid_y_diff]

        avg_shape_metrics[shape_name] = dict()
        avg_shape_metrics[shape_name]['area_diff'] = area_diff
        avg_shape_metrics[shape_name]['abs_dist'] = abs_dist
        avg_shape_metrics[shape_name]['centr_x_difference'] = centroid_x_diff
        avg_shape_metrics[shape_name]['centr_y_difference'] = centroid_y_diff
    else:
        shape_metrics[shape_name]['area_diff'].append(area_diff)
        shape_metrics[shape_name]['abs_dist'].append(abs_dist)
        shape_metrics[shape_name]['centr_x_difference'].append(centroid_x_diff)
        shape_metrics[shape_name]['centr_y_difference'].append(centroid_y_diff)

        avg_shape_metrics[shape_name]['area_diff'] = np.mean(shape_metrics[shape_name]['area_diff'])
        avg_shape_metrics[shape_name]['abs_dist'] = np.mean(shape_metrics[shape_name]['abs_dist'])
        avg_shape_metrics[shape_name]['centr_x_difference'] = np.mean(shape_metrics[shape_name]['centr_x_difference'])
        avg_shape_metrics[shape_name]['centr_y_difference'] = np.mean(shape_metrics[shape_name]['centr_y_difference'])

        #TODO: rewards_list
        #TODO: final_reward

    # print(shape_metrics)

    # shutdown the logger
    if wandb_logger is not None:
        wandb_logger.finish()

    if save_animation_gif:
        imageio.mimsave('animation.gif', [np.array(img) for i, img in enumerate(images) if i % 2 == 0], fps=29)

    # Close Environment
    env.close()


# 51.53378987312317


@hydra.main(config_path="confs", config_name="configs_test.yaml")
def main(cfg: DictConfig) -> None:
    OmegaConf.resolve(cfg)
    print(OmegaConf.to_yaml(cfg,resolve=True))
    run(cfg)
    # TODO: make function
    console = Console()

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Shape", style="dim", width=12)
    for el in metrics:
        table.add_column(el, justify="center")
    for el in avg_shape_metrics.keys():
        row_values = [el] + list(avg_shape_metrics.get(el).values())
        table.add_row(*(str(r) for r in row_values))


    # TODO: make function
    results_str = "shape"
    for name in metrics:
        results_str += ',' + name
    results_str += '\n'
    for shape in avg_shape_metrics.keys():
        results_str += shape  # <- maybe cast to str() to be absolutely safe
        for metric_name in avg_shape_metrics[shape].keys():
            results_str += ',' + str(avg_shape_metrics[shape][metric_name])
        results_str += '\n'

    with open('results.csv', 'w') as f:
        f.write(results_str)

    console.print(table)

if __name__ == '__main__':
    main()
    print('\n\n###############\nFinished testing, bye!\n###############')


