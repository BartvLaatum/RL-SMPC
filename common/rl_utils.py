import os
from os.path import join
from typing import Dict, Any, Tuple, List

import yaml
import wandb
import numpy as np

from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv, SubprocVecEnv, VecNormalize, VecMonitor

from common.results import Results
from common.callbacks import SaveVecNormalizeCallback, CustomWandbCallback
from envs.lettuce_greenhouse import LettuceGreenhouse
from envs.lettuce_greenhouse_state_noise import LettuceGreenhouseStateNoise

def load_rl_params(env_id: str, algorithms: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Load environment parameters from a yaml file.
    Arguments:
        file_name (str): Name of the MAT file containing the environment parameters.
        algorithms (str): Name of the algorithms to load the parameters for, e.g., "ppo", "sac", etc.
    Returns:
        Dict[str, Any]: Dictionary of environment parameters.
    """    
    # load the config file
    with open(f"configs/models/{algorithms}.yml", "r") as file:
        all_params = yaml.safe_load(file)

    return all_params["hyperparameters"], all_params[env_id]

def load_sweep_config(path: str, env_id: str, algorithm: str) -> Dict[str, Any]:
    with open(join(path, algorithm + ".yml"), "r") as f:
        sweep_config = yaml.load(f, Loader=yaml.FullLoader)
    return sweep_config[env_id]


def wandb_init(hyperparameters: Dict[str, Any],
               env_seed: int,
               model_seed: int,
               project: str,
               group: str,
               save_code: bool = False,
               ):
    config= {
        "env_seed": env_seed,
        "model_seed": model_seed,
        **hyperparameters,
    }

    config_exclude_keys = []
    run = wandb.init(
        project=project,
        config=config,
        group=group,
        sync_tensorboard=True,
        config_exclude_keys=config_exclude_keys,
        save_code=save_code,
        allow_val_change=True,
    )
    return run, config

def make_env(env_id, rank, seed, env_params, eval_env):
    '''
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :return: (Gym Environment) The gym environment
    '''
    envs = {
        "LettuceGreenhouse": LettuceGreenhouse,
        "LettuceGreenhouseStateNoise": LettuceGreenhouseStateNoise
    }

    def _init():
        env = envs[env_id](**env_params)
        # call reset with seed due to new seeding syntax of gymnasium environments
        env.reset(seed+rank)
        env.action_space.seed(seed + rank)
        return env
    return _init

def make_vec_env(
    env_id: str,
    env_params: Dict[str, Any],
    seed: int,
    n_envs: int,
    monitor_filename: str | None = None,
    vec_norm_kwargs: Dict[str, Any] | None = None,
    eval_env: bool = False
    ) -> VecEnv:
    """
    Creates a vectorized environment, with n individual envs.
    """
    # make dir if not exists
    if monitor_filename is not None and not os.path.exists(os.path.dirname(monitor_filename)):
        os.makedirs(os.path.dirname(monitor_filename), exist_ok=True)
    env = SubprocVecEnv([make_env(env_id, rank, seed, env_params, eval_env=eval_env) for rank in range(n_envs)])
    env = VecMonitor(env, filename=monitor_filename)

    if vec_norm_kwargs is not None:
        env = VecNormalize(env, **vec_norm_kwargs)
        if eval_env:
            env.training = False
            env.norm_reward = False
    # env.seed(seed=seed) DO WE NEED TO SEED ENVS HERE??
    return env


def create_callbacks(n_eval_episodes: int,
                     eval_freq: int,
                     env_log_dir: str|None,
                     save_name: str,
                     model_log_dir: str|None,
                     eval_env: VecEnv,
                     run: Any | None = None,
                     results: Results | None = None,
                     save_env: bool = True,
                     verbose: int = 1,
                     ) -> List[BaseCallback]:
    if env_log_dir:
        save_vec_best = SaveVecNormalizeCallback(save_freq=1, save_path=env_log_dir, verbose=2)
    else:
        save_vec_best = None
    eval_callback = CustomWandbCallback(eval_env,
                                        n_eval_episodes=n_eval_episodes,
                                        eval_freq=eval_freq,
                                        best_model_save_path=model_log_dir,
                                        name_vec_env=save_name,
                                        path_vec_env=env_log_dir,
                                        deterministic=True,
                                        callback_on_new_best=save_vec_best,
                                        run=run,
                                        results=results,
                                        verbose=verbose)
    wandbcallback = WandbCallback(verbose=verbose)
    return [eval_callback, wandbcallback]

def normalize_state(x, x_min ,x_max):
    state_norm = ((x - x_min)/(x_max - x_min))*(10) - 5
    return np.array(state_norm, dtype=np.float32)