import os
from os.path import join
import argparse
from functools import partial
import multiprocessing
from tqdm import tqdm

import numpy as np

from common.results import Results
from common.utils import get_parameters, load_env_params, load_mpc_params
from smpc import SMPC, Experiment

N_SIMS = 10

def main(args):
    """
    Execute the experiment varying prediction horizons for SMPC.

    This function orchestrates the complete horizon analysis experiment, including:
    - Loading experiment parameters and model configurations
    - Setting up parallel processing for efficient execution
    - Running experiments across multiple prediction horizons
    - Collecting and saving results

    Args:
        args: Command line arguments containing experiment configuration
    """

    ctx = multiprocessing.get_context("spawn")

    save_path = f"data/{args.project}/stochastic/smpc"
    os.makedirs(save_path, exist_ok=True)

    # Define column names for results class
    col_names = [
        "time", "x_0", "x_1", "x_2", "x_3", "y_0", "y_1", "y_2", "y_3",
        "u_0", "u_1", "u_2", "d_0", "d_1", "d_2", "d_3", 
        "J", "econ_rewards", "penalties", "rewards", "solver_times", "solver_success", "run"
    ]

    # Prediction horizons to evaluate (in hours)
    H = [1, 2, 3, 4, 5, 6, 7, 8]

    # run the experiment
    print(f"Running experiment for delta = {args.uncertainty_value}")
    env_params = load_env_params(args.env_id)
    mpc_params = load_mpc_params(args.env_id)

    # Run experiments for each prediction horizon
    for h in H:
        results = Results(col_names)
        print(f"Running for prediction horizon {h} hours")
        save_name = f"{args.save_name}-{h}H-{args.uncertainty_value}.csv"
        p = get_parameters()
        run_exp = partial(
            run_experiment, 
            h=h,
            uncertainty_value=args.uncertainty_value,
            env_params=env_params,
            mpc_params=mpc_params,
            args=args,
            p=p,
            save_name=save_name, 
        )

        # Execute parallel simulations
        num_processes = 10
        with ctx.Pool(processes=num_processes) as pool:
            data_list = list(tqdm(pool.imap(run_exp, range(N_SIMS)), total=N_SIMS))

        # Collect and save results
        for data in data_list:
            results.update_result(data)
        results.save(join(save_path, save_name))

def run_experiment(
        run, 
        h,
        uncertainty_value,
        env_params, 
        mpc_params,
        args,
        p,
        save_name,
        Ns=10,
    ):
    """
    Execute a single SMPC experiment for a specific prediction horizon.

    This function performs a complete closed-loop simulation SMPC for a given prediction horizon,
    including controller initialization, optimization problem setup, and result collection.

    Args:
        run (int): Monte Carlo run number (used for seeding)
        h (int): Prediction horizon in hours
        uncertainty_value (float): Level of parametric uncertainty
        env_params (dict): Environment parameters
        mpc_params (dict): MPC controller parameters
        args: Command line arguments
        p (np.ndarray): Model parameters
        save_name (str): Filename for saving results
        Ns (int, optional): Number of scenarios. Defaults to 10.

    Returns:
        np.ndarray: Simulation results including trajectories and performance metrics
    """

    # Add a small delay based on run ID to avoid resource contention
    smpc_rng = np.random.default_rng(42 + run)
    save_name = f"{args.save_name}-{h}H-{mpc_params['Ns']}Ns-{args.uncertainty_value}"
    mpc_params["rng"] = smpc_rng
    mpc_params["Np"] = int(h * 3600 / env_params["dt"])
    mpc_params["Ns"] = 10

    p = get_parameters()
    smpc = SMPC(**env_params, **mpc_params)

    smpc.define_nlp(p)
    exp_rng = np.random.default_rng(666 + run)
    exp = Experiment(smpc, save_name, args.project, args.weather_filename, uncertainty_value, p, exp_rng)
    exp.solve_nmpc()
    return exp.get_results(run)

if __name__ == "__main__":
    # Set the OMP_NUM_THREADS environment variable to limit threads per process
    # this fixed the optimization runtime..
    os.environ["OMP_NUM_THREADS"] = "1"

    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str, default="SMPC",
                       help="Project name for result organization")
    parser.add_argument("--env_id", type=str, default="LettuceGreenhouse",
                       help="Environment identifier")
    parser.add_argument("--save_name", type=str, required=True,
                       help="Base name for saving results")
    parser.add_argument("--weather_filename", default="outdoorWeatherWurGlas2014.csv", type=str,
                       help="Weather data filename")
    parser.add_argument("--mode", type=str, choices=['deterministic', 'stochastic'], required=True,
                       help="Mode for parametric uncertainty")
    parser.add_argument("--uncertainty_value", type=float, required=True,
                       help="Parametric uncertainty level")
    args = parser.parse_args()
    main(args)