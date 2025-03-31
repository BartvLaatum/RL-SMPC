import os
import argparse

import numpy as np

from common.utils import get_parameters
from rl_smpc import create_rl_smpc, load_experiment_parameters, Experiment

def main(args):
    save_path = f"data/{args.project}/stochastic/rlsmpc"
    os.makedirs(save_path, exist_ok=True)

    # uncertainty_values = [0.05, 0.15, 0.2]
    # model_names = ["worthy-cosmos-1", "volcanic-valley-2", "blooming-glade-3"]
    uncertainty_values = [0.15]
    model_names = ["volcanic-valley-2"]

    H = [1, 2, 3, 4, 5, 6]

    # run the experiment
    for uncertainty_value, model_name in zip(uncertainty_values, model_names):
        exp_rng = np.random.default_rng(666)
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
            print(f"Running for prediction horizon {h} hours")
            save_name = f"{model_name}-{args.save_name}-{h}H-{uncertainty_value}"
            p = get_parameters()
            rl_mpc = create_rl_smpc(
                h, 
                env_params, 
                mpc_params,
                rl_env_params,
                args.algorithm,
                env_path,
                rl_model_path,
                vf_path,
                args.use_trained_vf
            )
            rl_mpc.define_zero_order_snlp(p)

            exp = Experiment(rl_mpc, save_name, args.project, args.weather_filename, uncertainty_value, p, exp_rng)
            exp.solve_nsmpc("zero")
            exp.save_results(save_path)

if __name__ == "__main__":
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