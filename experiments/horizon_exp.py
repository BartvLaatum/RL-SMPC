import argparse
import os
from mpc import MPC, Experiment
from common.utils import load_env_params, load_mpc_params, get_parameters

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True, type=str)
    parser.add_argument("--save_name", required=True, type=str)
    parser.add_argument("--env_id", type=str, default="LettuceGreenhouse")
    parser.add_argument("--weather_filename", default="outdoorWeatherWurGlas2014.csv", type=str)
    args = parser.parse_args()

    save_path = f"data/{args.project}/mpc"
    os.makedirs(save_path, exist_ok=True)

    env_params = load_env_params(args.env_id)
    mpc_params = load_mpc_params(args.env_id)

    Pred_H = [1, 2, 3, 4, 5, 6]
    for H in Pred_H:
        dt = env_params["dt"]
        Np = int(H * 3600 / dt)
        print(Np)
        mpc_params["Np"] = Np
        save_name = f"{args.save_name}-{H}H"
        p = get_parameters()
        mpc = MPC(**env_params, **mpc_params)
        mpc.define_nlp(p)
        exp = Experiment(mpc, save_name, args.project, args.weather_filename)
        exp.solve_nmpc(p)
        exp.save_results(save_path)
