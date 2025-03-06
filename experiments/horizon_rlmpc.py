import os
from os.path import join
import argparse
from tqdm import tqdm
import multiprocessing

import numpy as np

from common.results import Results
from rl_mpc import RLMPC, Experiment
from common.rl_utils import load_rl_params
from common.utils import load_env_params, load_mpc_params, get_parameters
from functools import partial

def run_experiment(run, env_params, mpc_params, rl_env_params, args, env_path, rl_model_path, vf_path, p, seed, save_name):
    rl_mpc = RLMPC(
        env_params, 
        mpc_params, 
        rl_env_params, 
        args.algorithm,
        env_path,
        rl_model_path,
        vf_path,
        use_trained_vf=args.use_trained_vf,
        run=run
    )
    rl_mpc.define_nlp(p)
    rng = np.random.default_rng(seed + run)
    exp = Experiment(rl_mpc, save_name, args.project, args.weather_filename, args.uncertainty_value, p, rng)
    exp.solve_nmpc()
    return exp.get_results(run)


if __name__ == "__main__":
    ctx = multiprocessing.get_context("spawn")
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str, default="matching-thesis")
    parser.add_argument("--env_id", type=str, default="LettuceGreenhouse")
    parser.add_argument("--model_name", type=str, default="thesis-agent")
    parser.add_argument("--algorithm", type=str, default="sac")
    parser.add_argument("--save_name", type=str)
    parser.add_argument("--mode", type=str, choices=['deterministic', 'stochastic'], required=True)
    parser.add_argument("--uncertainty_value", type=float, help="List of uncertainty values")
    parser.add_argument("--use_trained_vf", action="store_true")
    parser.add_argument("--weather_filename", default="outdoorWeatherWurGlas2014.csv", type=str)
    args = parser.parse_args()
    load_path = f"train_data/{args.project}/{args.algorithm}/{args.mode}"
    save_path = f"data/{args.project}/{args.mode}/rlmpc"

    assert args.mode in ['deterministic', 'stochastic'], "Mode must be either 'deterministic' or 'stochastic'"
    if args.mode == 'stochastic':
        assert args.uncertainty_value is not None, "Uncertainty value must be provided for stochastic mode"
        assert (0 <= args.uncertainty_value < 1), "Uncertainty values must be between 0 and 1"
    else:
        args.uncertainty_value = 0
    os.makedirs(save_path, exist_ok=True)


    # load in the trained model
    rl_model_path = f"{load_path}/models/{args.model_name}/best_model.zip"
    vf_path = f"{load_path}/models/{args.model_name}/vf.zip"
    env_path = f"{load_path}/envs/{args.model_name}/best_vecnormalize.pkl"

    # load the config file
    env_params = load_env_params(args.env_id)
    mpc_params = load_mpc_params(args.env_id)
    dt = env_params["dt"]
    p = get_parameters()

    # load the RL parameters
    hyperparameters, rl_env_params = load_rl_params(args.env_id, args.algorithm)
    rl_env_params.update(env_params)
    rl_env_params["uncertainty_value"] = 0

    Pred_H = [1, 2, 3, 4, 5, 6]
    seed = 666

    col_names = [
        "time", "x_0", "x_1", "x_2", "x_3", "y_0", "y_1", "y_2", "y_3",
        "u_0", "u_1", "u_2", "d_0", "d_1", "d_2", "d_3", 
        "J", "econ_rewards", "penalties", "rewards", "run"
    ]

    for H in tqdm(Pred_H):
        results = Results(col_names)
        if args.mode == "stochastic":
            print(f"Running stochastic case for horizon: {H},\n Uncertainty value: {args.uncertainty_value}")
            N_sims = 30
            save_name = f"{args.save_name}-{args.model_name}-{H}H-{args.uncertainty_value}.csv"
        else:
            print(f"Running for horizon: {H}")
            N_sims = 1
            save_name = f"{args.save_name}-{args.model_name}-{H}H.csv"
        
        mpc_params["Np"] = int(H * 3600 / dt)

        run_exp = partial(
            run_experiment, 
            env_params=env_params,
            mpc_params=mpc_params,
            rl_env_params=rl_env_params,
            args=args,
            env_path=env_path,
            rl_model_path=rl_model_path,
            vf_path=vf_path,
            p=p,
            seed=seed,
            save_name=save_name
        )
        for run in tqdm(range(N_sims)):
            data = run_exp(run)
            results.update_result(data)

        # with ctx.Pool(processes=10) as pool:
            # run_exp = partial(run_experiment, 
            #     env_params=env_params,
            #     mpc_params=mpc_params,
            #     rl_env_params=rl_env_params,
            #     args=args,
            #     env_path=env_path,
            #     rl_model_path=rl_model_path,
            #     vf_path=vf_path,
            #     p=p,
            #     seed=seed,
            #     save_name=save_name)
            # data_list = list(tqdm(pool.imap(run_exp, range(N_sims)), total=N_sims))

        # for data in data_list:
        #     results.update_result(data)

        results.save(join(save_path,save_name))
