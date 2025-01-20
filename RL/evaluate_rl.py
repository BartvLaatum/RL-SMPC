import argparse
from os.path import join

import pandas as pd
import numpy as np

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

from common.evaluation import evaluate_policy
from common.rl_utils import make_vec_env, load_rl_params
from common.utils import co2dens2ppm, vaporDens2rh, load_env_params

ALG = {"ppo": PPO, 
       "sac": SAC}

def load_env(env_id, model_name, env_params, alg, stochastic):
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
    if stochastic:
        env = VecNormalize.load(join(path + f"{alg}/stochastic/envs", f"{model_name}/best_vecnormalize.pkl"), env)
    else:
        env = VecNormalize.load(join(path + f"{alg}/deterministic/envs", f"{model_name}/best_vecnormalize.pkl"), env)
    return env

def evaluate(model, env):
    N = env.get_attr("N")[0]
    nx = env.get_attr("nx")[0]
    ny = env.get_attr("ny")[0]
    nu = env.get_attr("nu")[0]

    x = np.zeros((nx, N+1))
    y = np.zeros((ny, N+1))
    u = np.zeros((nu, N+1))
    episode_rewards = np.zeros((1, N))

    dones = np.zeros((1,), dtype=bool)
    episode_starts = np.ones((1,), dtype=bool)
    episode_epi = np.zeros((1,N))

    observations = env.reset()
    timestep = 0
    states = None

    x[:, 0] = env.get_attr("x0")[0]
    y[:, 0] = env.env_method("get_y")[0]
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
        if dones.any():
            break
        x[:, timestep+1] = env.env_method("get_state")[0]
        y[:, timestep+1] = env.env_method("get_y")[0]
        u[:, timestep+1] = infos[0]["controls"]
    return x, u, y, episode_rewards, episode_epi

def save_results(env, x, u, y, rewards, epi):
    """
    """
    data = {}
    # transform the weather variables to the right units

    N = env.get_attr("N")[0]
    L = env.get_attr("L")[0]
    h = env.get_attr("h")[0]

    t = np.arange(0, L + h, h)[:-1]
    data["time"] = t / 86400

    for i in range(x.shape[0]):
        data[f"x_{i}"] = x[i, :N]
    for i in range(y.shape[0]):
        data[f"y_{i}"] = y[i, :N]
    for i in range(u.shape[0]):
        data[f"u_{i}"] = u[i, 1:]


    data["econ_rewards"] = epi.flatten()
    df = pd.DataFrame(data, columns=data.keys())
    df.to_csv(f"data/{args.project}/rl/{args.model_name}.csv", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", type=str, default="LettuceGreenhouse", help="Environment ID")
    parser.add_argument("--project", type=str, default="matching-thesis", help="Name of the project (in wandb)")
    parser.add_argument("--model_name", type=str, default="cosmic-music-45", help="Name of the trained RL model")
    parser.add_argument("--algorithm", type=str, default="ppo", help="Name of the algorithm (ppo or sac)")
    parser.add_argument("--stochastic", action="store_true", help="Whether to use stochastic control")
    args = parser.parse_args()

    path = f"train_data/{args.project}/"

    env_params = load_env_params(args.env_id)
    hyperparameters, rl_env_params = load_rl_params(args.env_id, args.algorithm)
    env_params.update(rl_env_params)
    eval_env = load_env(args.env_id, args.model_name, env_params, args.algorithm, args.stochastic)

    if args.stochastic:
        model = ALG[args.algorithm].load(join(path + f"{args.algorithm}/stochastic/models", f"{args.model_name}/best_model.zip"), device="cpu")
    else:
        model = ALG[args.algorithm].load(join(path + f"{args.algorithm}/deterministic/models", f"{args.model_name}/best_model.zip"), device="cpu")

    results = evaluate(model, eval_env)
    save_results(eval_env, *results)
