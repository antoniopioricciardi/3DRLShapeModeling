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
from env2d import CanvasModeling
from stable_baselines3.common.env_checker import check_env

from omegaconf import DictConfig, OmegaConf

from rich.console import Console
from rich.table import Table

from hydra.core.hydra_config import HydraConfig
import hydra 
import random 
import imageio
import os

MODELS_PATH = 'models/2d'
VIDEOS_PATH = 'videos'
LOGS_PATH = 'logs'

metrics = ['area_diff', 'abs_dist', 'centr_x_difference', 'centr_y_difference']
avg_metrics = {'area_diff': [], 'abs_dist': [], 'centr_x_difference': [], 'centr_y_difference': []}
shape_metrics = {}
avg_shape_metrics = {}
save_animation_gif = False
# measures =
# wandb.log({"area difference": area_diff})
#             wandb.log({"absolute distance between shapes": abs_dist})
#             wandb.log({"centroid x difference": centroid_x_diff})
#             wandb.log({"centroid y difference": centroid_y_diff})


# TODO: need to be able to use other correspondences and distances oher that the chamfer. Create a top-layer function to use in env2d.

# TODO: clean, or fix outputs and models paths from the folders created during each run

def run(cfg):
    save_animation_gif = cfg.save_animation_gif
    # print(cfg.from_img.img_path)
    # TOOD: a nicer way to go up a directory?
    os.chdir('../../../')
    # TODO: prova questo sotto
    # os.chdir(hydra.utils.get_original_cwd())
    # print(os.getcwd())
    # print(hydra.utils.get_original_cwd())  # this gives src path even when running with hydra
    # print('image:', cfg.inits_s.from_img.img_path)
    img_path = cfg.inits_s.from_img.img_path
    img_name = img_path[img_path.rfind("/")+1:]
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    set_random_seed(cfg.seed)

    # Settings
    num_points = cfg.neighborhood_size
    spread = cfg.spread
    max_steps = cfg.max_steps
    args_env = cfg

    model_class = SAC # PPO
    timesteps = cfg.total_timesteps
    learning_rate = cfg.learning_rate # 1e-2 is the best up to 4pt
    experiment_name = 'TEST:maxsteps_' + str(max_steps) + '-numpoints_' + str(num_points) + \
                 '-timesteps_' + str(timesteps) + '-lr_' + str(learning_rate)

    # model_name = 'circle_env-max_steps_' + str(max_steps) + '-numpoints_' + str(num_points) + \
    #              '-timesteps_' + str(timesteps) + '-lr_' + str(learning_rate)
    # model_name = cfg.model_name
    experiment_name = "TEST-2Dmodeling_gaussian-" + str(num_points) + 'points'

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
        project=experiment_name, # "2Dmodeling_gaussian_test",
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

        #env = DummyVecEnv([make_env(**args_env)])
    else:
        env = make_vec_env(make_env,cfg.envs, env_kwargs = {'cfg':args_env})
    
    # Wrapper for logging video
    env = VecVideoRecorder(env, f"videos/{wandb_logger.id}", record_video_trigger=lambda x: x >= 0, video_length=max_steps)
    # env = VecVideoRecorder(env, f"videos/{wandb_logger.id}", record_video_trigger=lambda x: (x % cfg.log_video_steps) == cfg.log_video_steps-1, video_length=max_steps)

    # define the model policy
    # policy_kwargs = hydra.utils.instantiate(cfg.policies['model'], **cfg.policies.params)
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
    wandb_callback = WandbTestCallback()
    obs = env.reset()
    if save_animation_gif:
        img = model.env.render(mode='rgb_array')    # print(env.l2_distances, 'obs:', obs)
    # print(env.l2_distances, 'obs:', obs)
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
            total_step_n = int(env.get_attr("total_step_n")[0])
            i += 1
            if i % tqdm_interval_update == 0:
                pbar.update(tqdm_interval_update)
            # env.render()
            steps_list.append(i)
            # dists_1.append(env.l2_distances[0])
            # dists_2.append(env.l2_distances[1])
            action, _states = model.predict(obs, deterministic=True)
            obs, rewards, done, info = env.step(action)
            if total_step_n % 500 == 0:
                area_diff = abs(env.get_attr("convex_hull_area_source")[0] - env.get_attr("convex_hull_area_canvas")[0])
                abs_dist = env.get_attr("abs_dist")[0]
                source_centroid = env.get_attr("source_centroid")[0]
                canvas_centroid = env.get_attr("canvas_centroid")[0]
                centroid_x_diff = source_centroid[0] - canvas_centroid[0]
                centroid_y_diff = source_centroid[1] - canvas_centroid[1]

                wandb.log({"area difference": area_diff})
                wandb.log({"absolute distance between shapes": abs_dist})
                wandb.log({"centroid x difference": centroid_x_diff})
                wandb.log({"centroid y difference": centroid_y_diff})
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
    env.close()

    # TODO: crea image_path e image_name così da non richiamare continuamente cfg...
    if shape_metrics.get(img_name) is None:
        shape_metrics[img_name] = dict()
        shape_metrics[img_name]['area_diff'] = [area_diff]
        shape_metrics[img_name]['abs_dist'] = [abs_dist]
        shape_metrics[img_name]['centr_x_difference'] = [centroid_x_diff]
        shape_metrics[img_name]['centr_y_difference'] = [centroid_y_diff]

        avg_shape_metrics[img_name] = dict()
        avg_shape_metrics[img_name]['area_diff'] = area_diff
        avg_shape_metrics[img_name]['abs_dist'] = abs_dist
        avg_shape_metrics[img_name]['centr_x_difference'] = centroid_x_diff
        avg_shape_metrics[img_name]['centr_y_difference'] = centroid_y_diff
    else:
        shape_metrics[img_name]['area_diff'].append(area_diff)
        shape_metrics[img_name]['abs_dist'].append(abs_dist)
        shape_metrics[img_name]['centr_x_difference'].append(centroid_x_diff)
        shape_metrics[img_name]['centr_y_difference'].append(centroid_y_diff)

        avg_shape_metrics[img_name]['area_diff'] = np.mean(shape_metrics[img_name]['area_diff'])
        avg_shape_metrics[img_name]['abs_dist'] = np.mean(shape_metrics[img_name]['abs_dist'])
        avg_shape_metrics[img_name]['centr_x_difference'] = np.mean(shape_metrics[img_name]['centr_x_difference'])
        avg_shape_metrics[img_name]['centr_y_difference'] = np.mean(shape_metrics[img_name]['centr_y_difference'])

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


@hydra.main(config_path="confs", config_name="2dconfigs_test.yaml")
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


