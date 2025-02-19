import os
import argparse

import numpy as np

from rl_mpc import RLMPC, Experiment
from common.utils import load_env_params, load_mpc_params, get_parameters
from common.rl_utils import load_rl_params

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str, default="matching-thesis")
    parser.add_argument("--env_id", type=str, default="LettuceGreenhouse")
    parser.add_argument("--save_name", type=str)
    parser.add_argument("--algorithm", type=str, default="sac")
    parser.add_argument("--model_name", type=str, default="thesis-agent")
    parser.add_argument("--mode", type=str, choices=['deterministic', 'stochastic'], required=True)
    parser.add_argument("--uncertainty_scale", type=float, help="List of uncertainty scale values")
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument("--use_trained_vf", action="store_true")
    parser.add_argument("--weather_filename", default="outdoorWeatherWurGlas2014.csv", type=str)
    args = parser.parse_args()
    load_path = f"train_data/{args.project}/{args.algorithm}/{args.mode}"
    save_path = f"data/{args.project}/{args.mode}/rlmpc"

    assert args.mode in ['deterministic', 'stochastic'], "Mode must be either 'deterministic' or 'stochastic'"
    if args.mode == 'stochastic':
        assert args.uncertainty_scale is not None, "Uncertainty scale must be provided for stochastic mode"
        assert (0 <= args.uncertainty_scale < 1), "Uncertainty scale values must be between 0 and 1"
    else:
        args.uncertainty_scale = 0
    os.makedirs(save_path, exist_ok=True)


    # load in the trained model
    rl_model_path = f"{load_path}/models/{args.model_name}/best_model.zip"
    vf_path = f"{load_path}/models/{args.model_name}/vf.zip"
    env_path = f"{load_path}/envs/{args.model_name}/best_vecnormalize.pkl"

    # load the config file
    env_params = load_env_params(args.env_id)
    mpc_params = load_mpc_params(args.env_id)

    # load the RL parameters
    hyperparameters, rl_env_params = load_rl_params(args.env_id, args.algorithm)
    rl_env_params.update(env_params)
    Pred_H = [1, 2, 3, 4, 5, 6]
    for H in Pred_H:
        rng = np.random.default_rng(args.seed)
        
        dt = env_params["dt"]
        Np = int(H * 3600 / dt)
        print(Np)
        mpc_params["Np"] = Np
        if args.mode == "stochastic":
            save_name = f"{args.save_name}-{args.model_name}-{H}H-{args.uncertainty_scale}"
        else:
            save_name = f"{args.save_name}-{args.model_name}-{H}H"
        p = get_parameters()
        rl_mpc = RLMPC(
            env_params, 
            mpc_params, 
            rl_env_params, 
            args.algorithm,
            env_path,
            rl_model_path,
            vf_path,
            use_trained_vf=args.use_trained_vf
        )
        rl_mpc.define_nlp(p)

        # run the experiment
        exp = Experiment(rl_mpc, save_name, args.project, args.weather_filename, uncertainty_scale=args.uncertainty_scale, rng=rng)
        exp.solve_nmpc(p)
        exp.save_results(save_path)
