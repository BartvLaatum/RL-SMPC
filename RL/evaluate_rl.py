import argparse
import os
from os.path import join
from tqdm import tqdm

import pandas as pd
import numpy as np

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

from common.rl_utils import make_vec_env, load_rl_params
from common.utils import load_env_params
from common.results import Results

ALG = {"ppo": PPO, 
       "sac": SAC}

def load_env(env_id, model_name, env_params, load_path):
    # Setup new environment for training
    env = make_vec_env(
        env_id, 
        env_params, 
        seed=666, 
        n_envs=1, 
        monitor_filename=None, 
        vec_norm_kwargs=None,
        eval_env=True
    )
    env = VecNormalize.load(join(load_path + f"/envs", f"{model_name}/best_vecnormalize.pkl"), env)
    return env

def evaluate(model, env):
    L = env.get_attr("L")[0]
    dt = env.get_attr("dt")[0]
    time = (np.arange(0, L, dt)/86400).reshape(1, -1)
    N = env.get_attr("N")[0]
    nx = env.get_attr("nx")[0]
    ny = env.get_attr("ny")[0]
    nu = env.get_attr("nu")[0]

    x = np.zeros((nx, N+1))
    y = np.zeros((ny, N+1))
    u = np.zeros((nu, N+1))
    d = np.zeros((ny, N+1))
    episode_rewards = np.zeros((1, N))
    penalties = np.zeros((1, N))

    dones = np.zeros((1,), dtype=bool)
    episode_starts = np.ones((1,), dtype=bool)
    episode_epi = np.zeros((1, N))

    observations = env.reset()
    timestep = 0
    states = None

    x[:, 0] = env.get_attr("x0")[0]
    y[:, 0] = env.env_method("get_y")[0]
    u[:, 0] = env.get_attr("u0")[0]
    d[:, 0] = env.env_method("get_d")[0]

    for timestep in range(N):
        actions, states = model.predict(
            observations,  # type: ignore[arg-type]
            state=states,
            episode_start=episode_starts,
            deterministic=True,
    )
        observations, rewards, dones, infos = env.step(actions)
        episode_rewards[:, timestep] += rewards
        episode_epi[:, timestep] += infos[0]["EPI"]
        penalties[:, timestep] += infos[0]["penalty"]
        if dones.any():
            break
        x[:, timestep+1] = env.env_method("get_state")[0]
        y[:, timestep+1] = env.env_method("get_y")[0]
        u[:, timestep+1] = infos[0]["controls"]
        d[:, timestep+1] = env.env_method("get_d")[0]
    return np.concatenate([time, x[:,:N], y[:,:N], u[:,:N], d[:,:N], episode_epi, penalties, episode_rewards]).T

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str, default="matching-thesis", help="Name of the project (in wandb)")
    parser.add_argument("--env_id", type=str, default="LettuceGreenhouse", help="Environment ID")
    parser.add_argument("--model_name", type=str, default="cosmic-music-45", help="Name of the trained RL model")
    parser.add_argument("--algorithm", type=str, default="ppo", help="Name of the algorithm (ppo or sac)")
    parser.add_argument("--mode", type=str, choices=['deterministic', 'stochastic'], required=True)
    parser.add_argument("--uncertainty_value", type=float, help="Uncertainty scale value")
    args = parser.parse_args()

    assert args.mode in ['deterministic', 'stochastic'], "Mode must be either 'deterministic' or 'stochastic'"
    if args.mode == 'stochastic':
        assert args.uncertainty_value is not None, "Uncertainty scale must be provided for stochastic mode"
        assert (0 <= args.uncertainty_value < 1), "Uncertainty scale values must be between 0 and 1"
        n_sims = 30
    else:
        args.uncertainty_value = 0
        n_sims = 1

    load_path = f"train_data/{args.project}/{args.algorithm}/{args.mode}/"

    save_path = f"data/{args.project}/{args.mode}/rl"
    os.makedirs(save_path, exist_ok=True)

    env_params = load_env_params(args.env_id)
    hyperparameters, rl_env_params = load_rl_params(args.env_id, args.algorithm)
    env_params["uncertainty_value"] = args.uncertainty_value
    env_params.update(rl_env_params)
    eval_env = load_env(args.env_id, args.model_name, env_params, load_path)

    model = ALG[args.algorithm].load(join(load_path + f"models", f"{args.model_name}/best_model.zip"), device="cpu")

    col_names = [
        "time", "x_0", "x_1", "x_2", "x_3", "y_0", "y_1", "y_2", "y_3",
        "u_0", "u_1", "u_2", "d_0", "d_1", "d_2", "d_3", 
        "econ_rewards", "penalties", "rewards", "run"
    ]
    seed = 666
    results = Results(col_names)

    def run_experiment(run):
        eval_env.env_method("set_seed", seed+run)
        data = evaluate(model, eval_env)
        run_column = np.full((data.shape[0], 1), run)
        data = np.hstack((data, run_column))
        return data

    for run in tqdm(range(n_sims)):
        data = run_experiment(run)
        results.update_result(data)
    results.save(join(save_path, f"{args.model_name}.csv"))
