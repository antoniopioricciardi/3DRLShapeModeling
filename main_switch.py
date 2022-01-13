import os
import torch
import hydra
import wandb
import random
import trainer
import tester
import numpy as np

from names import Filenames
from rich.table import Table
from rich.console import Console
from env3d_triangles import CanvasModeling
from omegaconf import DictConfig, OmegaConf
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import HerReplayBuffer, SAC, DDPG, PPO
from stable_baselines3.common.vec_env import VecVideoRecorder, VecMonitor


MODELS_PATH = 'models'
VIDEOS_PATH = 'videos'
LOGS_PATH = 'logs'
RESULTS_PATH = 'results'

metrics = ['area_diff', 'abs_dist', 'centr_x_difference', 'centr_y_difference']
avg_metrics = {'area_diff': [], 'abs_dist': [], 'centr_x_difference': [], 'centr_y_difference': []}
shape_metrics = {}
avg_shape_metrics = {}

if not os.path.exists(MODELS_PATH):
    os.mkdir(MODELS_PATH)
if not os.path.exists(VIDEOS_PATH):
    os.mkdir(VIDEOS_PATH)
if not os.path.exists(LOGS_PATH):
    os.mkdir(LOGS_PATH)
if not os.path.exists(RESULTS_PATH):
    os.mkdir(RESULTS_PATH)


def run(cfg):

    # print(to_absolute_path(''))  # this gives src path even when running with hydra
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    set_random_seed(cfg.seed)
    # torch.use_deterministic_algorithms(mode=True)  # probably to slow down code, based on functions we use, might not always work
    # Settings
    save_animation_gif = cfg.save_animation_gif
    num_triangles = cfg.num_triangles
    steps_per_vertex = cfg.steps_per_vertex
    num_points = cfg.num_points
    spread = cfg.spread
    max_steps = cfg.max_steps
    args_env = cfg

    model_class = SAC  # PPO
    timesteps = cfg.total_timesteps
    learning_rate = cfg.learning_rate  # 1e-2 is the best up to 4pt
    experiment_name = 'TRAIN:maxsteps_' + str(max_steps) + '-numpoints_' + str(num_triangles) + \
                      '-timesteps_' + str(timesteps) + '-lr_' + str(learning_rate)

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
    model_path_list = [cfg.model_name, '-numpoints_' + str(num_points), '-seed_' + str(cfg.seed),
                       # '-totaltimesteps' + str(timesteps),'-maxsteps_' + str(max_steps),
                       '-spread_' + str(spread),
                       '-lr_' + str(learning_rate), '-sigma_' + str(cfg.sigma),
                       '-neighborsmovementscale_' + str(cfg.neighbors_movement_scale)
                       ]

    model_name = ''
    for dir_name in model_path_list:
        model_name += dir_name
    print(os.getcwd())
    print(model_name)
    "linear_model-numpoints_3-seed_42-spread_5-lr_0.001-sigma_0.1-neighborsmovementscale_0.5"
    "linear_model-numpoints_3-seed_42-spread_5-lr_0.001-sigma_0.1-neighborsmovementscale_0.5"
    "linear_model-numpoints_3-seed_42-spread_5-lr_0.001-sigma_0.1-neighborsmovementscale_0.5"
    # Initialise a W&B run
    wandb_logger = wandb.init(
        name=model_name,
        project="TRAIN-3Dmodeling_linear",
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
    policy_kwargs = hydra.utils.instantiate(cfg.policies['model'], **cfg.policies.params)

    policy_kwargs = dict(activation_fn=torch.nn.ReLU, net_arch=dict(pi=[256, 256, 256], qf=[256, 256, 256]), use_sde=False)
    os.chdir('../../../')

    model = model_class.load(os.path.join(MODELS_PATH, model_name), policy_kwargs=policy_kwargs)
    vec_prova = np.random.random((1,9))
    num = model.predict(vec_prova)
    print(num)
    exit(3)
    # Wrapper for multi-environment
    def make_env(cfg):
        env = CanvasModeling(cfg)
        k = check_env(env, warn=True)
        env = Monitor(env, LOGS_PATH)  # record stats such as returns
        return env

    # Generate instances
    if cfg.envs == 1:
        env = make_vec_env(make_env, cfg.envs, env_kwargs={'cfg': args_env})
        # env = DummyVecEnv([make_env(**args_env)])
    else:
        env = make_vec_env(make_env, cfg.envs, env_kwargs={'cfg': args_env})

    # Wrapper for logging video
    env = VecVideoRecorder(env, f"videos/{wandb_logger.id}",
                           record_video_trigger=lambda x: x % cfg.log_video_steps == cfg.log_video_steps - 1,
                           video_length=max_steps)

    env.seed(cfg.seed)

    # define the model policy
    policy_kwargs = hydra.utils.instantiate(cfg.policies['model'], **cfg.policies.params)

    policy_kwargs = dict(activation_fn=torch.nn.ReLU, net_arch=dict(pi=[256, 256, 256], qf=[256, 256, 256]))

    if cfg.is_training == True:
        model = model_class("MlpPolicy", env, verbose=1, policy_kwargs=policy_kwargs,
                            tensorboard_log=f"runs/{wandb_logger.id}",
                            learning_rate=learning_rate, seed=cfg.seed)  # , n_steps = cfg.update_step)
        model = trainer.train(model, timesteps)
        # shutdown the logger
        if wandb_logger is not None:
            wandb_logger.finish()

        # Save
        model.save(os.path.join(MODELS_PATH, model_name))
        env.close()
    else:
        target_shape_path = cfg.inits_s.from_shape.shape_path
        target_shape_name = target_shape_path[target_shape_path.rfind("/") + 1:]
        canvas_shape_path = cfg.inits_c.from_shape.shape_path
        canvas_shape_name = canvas_shape_path[canvas_shape_path.rfind("/") + 1:]
        shape_name = target_shape_name + '--' + canvas_shape_name

        res_path = os.path.join(RESULTS_PATH, shape_name)

        model = model_class.load(os.path.join(MODELS_PATH, model_name), env=env, policy_kwargs=policy_kwargs,
                                 tensorboard_log=f"runs/{wandb_logger.id}")
        print('oohoh')
        exit(4)
        area_diff, abs_dist, centroid_x_diff, centroid_y_diff = tester.test(cfg, env, model, wandb_logger, model_name,
                                                                            save_animation_gif, res_path)
        env.close()

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
            avg_shape_metrics[shape_name]['centr_x_difference'] = np.mean(
                shape_metrics[shape_name]['centr_x_difference'])
            avg_shape_metrics[shape_name]['centr_y_difference'] = np.mean(
                shape_metrics[shape_name]['centr_y_difference'])




@hydra.main(config_path='confs', config_name="configs_test.yaml")
def main_test(cfg: DictConfig) -> None:
    print('ok')
    OmegaConf.resolve(cfg)
    print(OmegaConf.to_yaml(cfg, resolve=True))
    run(cfg)

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


@hydra.main(config_path='confs', config_name="configs.yaml")
def main_train(cfg: DictConfig) -> None:
    OmegaConf.resolve(cfg)
    print(OmegaConf.to_yaml(cfg, resolve=True))
    run(cfg)

if __name__ == '__main__':
    import sys
    type = sys.argv[1]

    if type == 'type=test':
        main_test()
    else:
        main_train()

    # main()
