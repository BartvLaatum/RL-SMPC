import argparse
import os
from mpc import MPC, Experiment
from common.utils import load_env_params, load_mpc_params, get_parameters
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True, type=str)
    parser.add_argument("--save_name", required=True, type=str)
    parser.add_argument("--env_id", type=str, default="LettuceGreenhouse")
    parser.add_argument("--mode", type=str, choices=['deterministic', 'stochastic'], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--weather_filename", default="outdoorWeatherWurGlas2014.csv", type=str)
    args = parser.parse_args()

    save_path = f"data/{args.project}/{args.mode}/mpc"
    os.makedirs(save_path, exist_ok=True)

    env_params = load_env_params(args.env_id)
    mpc_params = load_mpc_params(args.env_id)
    
    uncertainties = [0.05, 0.1, 0.15]
    dt = env_params["dt"]
    p = get_parameters()

    Pred_H = [1, 2, 3, 4, 5, 6]

    if args.mode == "stochastic":
        for uncertainty_scale in uncertainties:
            N_sims = 1
            for H in Pred_H:
                print("Running for horizon: {H},\n Uncertainty scale: {uncertainty_scale}")
                rng = np.random.default_rng(args.seed)
                mpc_params["Np"] = int(H * 3600 / dt)
                save_name = f"{args.save_name}-{H}H-{uncertainty_scale}"
                mpc = MPC(**env_params, **mpc_params)
                mpc.define_nlp(p)

                for run in range(N_sims):
                    exp = Experiment(mpc, save_name, args.project, args.weather_filename, uncertainty_scale, rng)
                    exp.solve_nmpc(p)
                    exp.save_results(save_path)
    else:
        for H in Pred_H:
            mpc_params["Np"] = int(H * 3600 / dt)
            save_name = f"{args.save_name}-{H}H"
            mpc = MPC(**env_params, **mpc_params)
            mpc.define_nlp(p)
            exp = Experiment(mpc, save_name, args.project, args.weather_filename, uncertainty_scale=0, rng=np.random.default_rng(args.seed))
            exp.solve_nmpc(p)
            exp.save_results(save_path)
