import os
import argparse

from tqdm import tqdm
import numpy as np

from smpc import SMPC, Experiment
from rl_smpc import RLSMPC
from rl_smpc import Experiment as RLExp
from common.utils import load_env_params, load_mpc_params, get_parameters
from common.rl_utils import load_rl_params 

def smpc_ol(env_params, mpc_params, exp_rng, save_name, project, weather_filename, uncertainty_value):
    p = get_parameters()
    mpc = SMPC(**env_params, **mpc_params)
    mpc.define_nlp(p)
    exp = Experiment(mpc, save_name, project, weather_filename, uncertainty_value, p, exp_rng)
    exp.solve_smpc_OL_predictions()

def rlsmpc_ol(env_params, mpc_params, exp_rng, save_name, project, weather_filename, uncertainty_value, 
              env_id, algorithm, model_name, mode="stochastic", use_trained_vf=True, order="zero"):

    # load the RL parameters
    hyperparameters, rl_env_params = load_rl_params(env_id, algorithm)
    rl_env_params.update(env_params)
    rl_env_params["uncertainty_value"] = uncertainty_value

    load_path = f"train_data/{project}/{algorithm}/{mode}"

    # the paths to the RL models and environment
    rl_model_path = f"{load_path}/models/{model_name}/best_model.zip"
    vf_path = f"{load_path}/models/{model_name}/vf.zip"
    env_path = f"{load_path}/envs/{model_name}/best_vecnormalize.pkl"

    p = get_parameters()
    rl_mpc = RLSMPC(
        env_params,
        mpc_params, 
        rl_env_params, 
        algorithm,
        env_path,
        rl_model_path,
        use_trained_vf=use_trained_vf,
        vf_path=vf_path,
        run=0,
    )
    if order == "zero":
        rl_mpc.define_zero_order_snlp(p)
    elif order == "first":
        rl_mpc.define_first_order_snlp(p)

    exp = RLExp(rl_mpc, save_name, project, weather_filename, uncertainty_value, p, exp_rng)
    exp.solve_smpc_OL_predictions(order=order)

def main(args):
    h = 3
    mpc_rng = np.random.default_rng(42)
    exp_rng = np.random.default_rng(666)
    env_params = load_env_params(args.env_id)
    mpc_params = load_mpc_params(args.env_id)
    mpc_params["uncertainty_value"] = args.uncertainty_value
    mpc_params["rng"] = mpc_rng
    mpc_params["Np"] = int(h * 3600 / env_params["dt"])
    mpc_params["Ns"] = 20
    save_name = f"{args.save_name}-{h}H-{mpc_params['Ns']}Ns-{args.uncertainty_value}"

    # smpc_ol(
    #     env_params,
    #     mpc_params,
    #     exp_rng,
    #     save_name=save_name,
    #     project=args.project,
    #     weather_filename=args.weather_filename,
    #     uncertainty_value=args.uncertainty_value
    # )

    rlsmpc_ol(env_params, mpc_params, exp_rng, save_name, args.project, args.weather_filename, args.uncertainty_value, 
                args.env_id, "sac", "brisk-resonance-24", mode="stochastic", use_trained_vf=True, order="zero")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str)
    parser.add_argument("--env_id", type=str, default="LettuceGreenhouse")
    parser.add_argument("--save_name", type=str)
    parser.add_argument("--uncertainty_value", type=float, required=True)
    parser.add_argument("--weather_filename", default="outdoorWeatherWurGlas2014.csv", type=str)
    args = parser.parse_args()
    main(args)