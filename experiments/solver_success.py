import os
from os.path import join

import numpy as np
import matplotlib.pyplot as plt
import cmcrameri.cm as cmc

from mpc import MPC, Experiment as MPCExperiment
from smpc import SMPC, Experiment as SMPCExperiment
from rl_smpc import RLSMPC, Experiment as RLSMPCExperiment, create_rl_smpc
from visualisations import plot_config
from common.utils import load_env_params, load_mpc_params, get_parameters
from common.rl_utils import load_rl_params

def run_mpc_experiment(env_id, save_path, save_name, project, weather_filename, uncertainty_value, horizons, n_days=5):
    """
    Create an experiment, solve nsmpc and nsmpc2, and return the experiment object.
    
    Parameters:
        order (str): Model order, e.g. "zero" or "first".
        project (str): Project identifier.
        weather_filename (str): Name of the weather file.
        uncertainty_value (float): Uncertainty value.

    Returns:
        exp: The Experiment object after solving.
    """
    mpcs = {}
    env_params = load_env_params(env_id)
    mpc_params = load_mpc_params(env_id)
    mpc_params["uncertainty_value"] = uncertainty_value
    env_params["n_days"] = n_days
    p = get_parameters()

    for h in horizons:
        new_save_name = f"{save_name}-{h}H-{uncertainty_value}.csv"
        mpc_params["Np"] = int(h * 3600 / env_params["dt"])

        mpc = MPC(
            **env_params,
            **mpc_params, 
        )

        mpc.define_nlp(p)

        exp = MPCExperiment(mpc, new_save_name, project, weather_filename, uncertainty_value, p,
                            rng=np.random.default_rng(42))
        exp.solve_nmpc2()
        # mpcs[h] = exp.retrieve_results()
        exp.save_results(save_path)

    return mpcs

def run_nsmpc_experiment(
        env_id,
        algorithm,
        env_path,
        rl_model_path,
        vf_path,
        save_path, 
        save_name, 
        order, 
        project, 
        weather_filename, 
        uncertainty_value, 
        horizons, 
        n_days
    ):
    """
    Create an experiment, solve nsmpc and nsmpc2, and return the experiment object.

    Parameters:
        order (str): Model order, e.g. "zero" or "first".
        project (str): Project identifier.
        weather_filename (str): Name of the weather file.
        uncertainty_value (float): Uncertainty value.

    Returns:
        exp: The Experiment object after solving.
    """
    # load the environment parameters
    env_params = load_env_params(env_id)
    mpc_params = load_mpc_params(env_id)
    mpc_params["uncertainty_value"] = uncertainty_value

    hyperparameters, rl_env_params = load_rl_params(env_id, algorithm)
    rl_env_params.update(env_params)
    rl_env_params["uncertainty_value"] = uncertainty_value

    rl_smpcs = {}
    env_params["n_days"] = n_days
    mpc_params["Ns"] = 10
    p = get_parameters()
    order = "zero"

    for h in horizons:

        mpc_params["rng"] = np.random.default_rng(666)
        mpc_params["Np"] = int(h * 3600 / env_params["dt"])
        new_save_name = f"{save_name}-{h}H-{uncertainty_value}.csv"

        rl_smpc = create_rl_smpc(
            h, 
            env_params, 
            mpc_params,
            rl_env_params,
            algorithm,
            env_path,
            rl_model_path,
            vf_path,
            run=0,
            use_trained_vf=True
        )

        if order == "zero":
            rl_smpc.define_zero_order_snlp(p)
        elif order == "first":
            rl_smpc.define_first_order_snlp(p)
    
        exp = RLSMPCExperiment(rl_smpc, new_save_name, project, weather_filename, uncertainty_value, p,
                            rng=np.random.default_rng(42))
        exp.solve_nsmpc2(order=order)
        rl_smpcs[h] = exp.retrieve_results()
        exp.save_results(save_path)

    return rl_smpcs

def run_smpc_experiment(env_id, save_path, save_name, project, weather_filename, uncertainty_value, horizons, n_days=5):
    """
    Create an experiment, solve nsmpc and nsmpc2, and return the experiment object.
    
    Parameters:
        order (str): Model order, e.g. "zero" or "first".
        project (str): Project identifier.
        weather_filename (str): Name of the weather file.
        uncertainty_value (float): Uncertainty value.

    Returns:
        exp: The Experiment object after solving.
    """
    smpcs = {}
    env_params = load_env_params(env_id)
    smpc_params = load_mpc_params(env_id)
    smpc_params["uncertainty_value"] = uncertainty_value
    env_params["n_days"] = n_days
    smpc_params["Ns"] = 10

    p = get_parameters()

    for h in horizons:
        new_save_name = f"{save_name}-{h}H-{uncertainty_value}.csv"

        smpc_params["rng"] = np.random.default_rng(666)
        smpc_params["Np"] = int(h * 3600 / env_params["dt"])

        smpc = SMPC(
            **env_params,
            **smpc_params, 
        )

        smpc.define_nlp(p)

        exp = SMPCExperiment(smpc, new_save_name, project, weather_filename, uncertainty_value, p,
                            rng=np.random.default_rng(42))
        exp.solve_nmpc2()
        smpcs[h] = exp.retrieve_results()
        exp.save_results(save_path)
    return smpcs


def main():
    # RL-SMPC parameters
    project = "solver-success-smpc"
    env_id = "LettuceGreenhouse"
    uncertainty_value = 0.1
    weather_filename = "outdoorWeatherWurGlas2014.csv"
    algorithm = "sac"
    mode = "stochastic"
    model_name = "restful-pyramid-7"

    load_path = f"train_data/{project}/{algorithm}/{mode}"

    # the paths to the RL models and environment
    rl_model_path = f"{load_path}/models/{model_name}/best_model.zip"
    vf_path = f"{load_path}/models/{model_name}/vf.zip"
    env_path = f"{load_path}/envs/{model_name}/best_vecnormalize.pkl"
    np.set_printoptions(suppress=True,precision=6)

    horizons = [1, 2, 3, 4, 5, 6]
    n_days = 40
    print("Running MPC experiment...")

    save_path = f"data/{project}/{mode}/mpc"
    os.makedirs(save_path, exist_ok=True)
    mpc = run_mpc_experiment(env_id, save_path, "mpc-clipped", project, weather_filename, uncertainty_value, horizons, n_days)
    
    # order="zero"
    # print("Running RL-SMPC experiment...")

    # save_path = f"data/{project}/stochastic/rlsmpc"
    # os.makedirs(save_path, exist_ok=True)
    # rl_smpc = run_nsmpc_experiment(env_id, algorithm, env_path, rl_model_path, vf_path,save_path, "rl-smpc", order, project, weather_filename, uncertainty_value, horizons, n_days)

    save_name = "smpc-clipped"

    save_path = f"data/{project}/stochastic/smpc"
    os.makedirs(save_path, exist_ok=True)
    smpc = run_smpc_experiment(env_id, save_path, save_name, project, weather_filename, uncertainty_value, horizons, n_days)

if __name__ == "__main__":
    main()
