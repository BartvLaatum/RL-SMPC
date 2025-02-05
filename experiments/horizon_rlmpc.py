import argparse
import os
from rl_mpc import RLMPC, Experiment
from common.utils import load_env_params, load_mpc_params, get_parameters
from common.rl_utils import load_rl_params

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str, default="matching-thesis")
    parser.add_argument("--env_id", type=str, default="LettuceGreenhouse")
    parser.add_argument("--save_name", type=str)
    parser.add_argument("--weather_filename", default="outdoorWeatherWurGlas2014.csv", type=str)
    parser.add_argument("--algorithm", type=str, default="sac")
    parser.add_argument("--model_name", type=str, default="thesis-agent")
    parser.add_argument("--use_trained_vf", action="store_true")
    parser.add_argument("--stochastic", action="store_true")
    args = parser.parse_args()
    if args.stochastic:
        load_path = f"train_data/{args.project}/{args.algorithm}/stochastic"
    else:
        load_path = f"train_data/{args.project}/{args.algorithm}/deterministic"
    save_path = f"data/{args.project}/rlmpc"
    os.makedirs(save_path, exist_ok=True)

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
        dt = env_params["h"]
        Np = int(H * 3600 / dt)
        print(Np)
        mpc_params["Np"] = Np
        save_name = f"{args.save_name}-{H}H"
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
        exp = Experiment(rl_mpc, save_name, args.project, args.weather_filename)
        exp.solve_nmpc(p)
        exp.save_results(save_path)
