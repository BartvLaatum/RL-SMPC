import os
from os.path import join
import argparse
from functools import partial
import multiprocessing
from tqdm import tqdm

import numpy as np

from common.results import Results
from common.utils import get_parameters
from rl_smpc import create_rl_smpc, load_experiment_parameters, Experiment

N_SIMS = 10

def main(args):
    ctx = multiprocessing.get_context("spawn")

    save_path = f"data/{args.project}/stochastic/rlsmpc"
    os.makedirs(save_path, exist_ok=True)

    uncertainty_values = [
        0.025, 0.05, 0.075, 0.1, \
        0.125, 0.15, 0.175, 0.2
    ]
    model_names = [
        "mild-rain-8", "worthy-cosmos-1", "rare-shadow-9", "restful-pyramid-7",\
        "lyric-sky-10", "volcanic-valley-2", "peach-haze-11", "blooming-glade-3"
    ]

    col_names = [
        "time", "x_0", "x_1", "x_2", "x_3", "y_0", "y_1", "y_2", "y_3",
        "u_0", "u_1", "u_2", "d_0", "d_1", "d_2", "d_3", 
        "J", "econ_rewards", "penalties", "rewards", "run"
    ]

    H = [1, 2, 3, 4, 5, 6]

    # run the experiment
    for uncertainty_value, model_name in zip(uncertainty_values, model_names):

        # already have 10 simulations for this one..
        # if model_name == "restful-pyramid-7":
        #     continue

        print(f"Running experiment for delta = {uncertainty_value}")
        (env_params, mpc_params, rl_env_params, env_path, rl_model_path, vf_path) = \
            load_experiment_parameters(
                args.project, 
                args.env_id, 
                args.algorithm, 
                args.mode, 
                model_name, 
                uncertainty_value)

        for h in H:
            results = Results(col_names)
            print(f"Running for prediction horizon {h} hours")
            save_name = f"{model_name}-{args.save_name}-{h}H-{uncertainty_value}.csv"
            p = get_parameters()

            run_exp = partial(
                run_experiment, 
                h=h,
                uncertainty_value=uncertainty_value,
                env_params=env_params,
                mpc_params=mpc_params,
                rl_env_params=rl_env_params,
                args=args,
                env_path=env_path,
                rl_model_path=rl_model_path,
                vf_path=vf_path,
                p=p,
                save_name=save_name, 
            )

            num_processes = 10
            with ctx.Pool(processes=num_processes) as pool:
                data_list = list(tqdm(pool.imap(run_exp, range(N_SIMS)), total=N_SIMS))

            for data in data_list:
                results.update_result(data)

            results.save(join(save_path, save_name))

def run_experiment(
        run, 
        h,
        uncertainty_value,
        env_params, 
        mpc_params,
        rl_env_params,
        args,
        env_path,
        rl_model_path,
        vf_path,
        p,
        save_name,
    ):
    # Add a small delay based on run ID to avoid resource contention

    rl_mpc = create_rl_smpc(
        h, 
        env_params, 
        mpc_params,
        rl_env_params,
        args.algorithm,
        env_path,
        rl_model_path,
        vf_path,
        run,
        args.use_trained_vf
    )

    rl_mpc.define_zero_order_snlp(p)
    exp_rng = np.random.default_rng(666 + run)
    exp = Experiment(rl_mpc, save_name, args.project, args.weather_filename, uncertainty_value, p, exp_rng)
    exp.solve_nsmpc("zero")
    return exp.get_results(run)

if __name__ == "__main__":
    # Set the OMP_NUM_THREADS environment variable to limit threads per process
    # this fixed the optimization runtime..
    os.environ["OMP_NUM_THREADS"] = "1"

    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str, default="uncertainty-comparison")
    parser.add_argument("--env_id", type=str, default="LettuceGreenhouse")
    parser.add_argument("--save_name", type=str)
    parser.add_argument("--weather_filename", default="outdoorWeatherWurGlas2014.csv", type=str)
    parser.add_argument("--algorithm", type=str, default="sac")
    parser.add_argument("--mode", type=str, choices=['deterministic', 'stochastic'], required=True)
    parser.add_argument("--use_trained_vf", action="store_true")
    args = parser.parse_args()
    main(args)