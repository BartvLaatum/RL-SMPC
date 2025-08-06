import os
from os.path import join
import argparse
from tqdm import tqdm
from multiprocessing import Pool

import numpy as np

from mpc import MPC, Experiment
from common.results import Results
from common.utils import load_env_params, load_mpc_params, get_parameters

def main(args) -> None:
    """
    Execute the experiment varying prediction horizons for RL-SMPC.

    This function orchestrates the complete horizon analysis experiment, including:
    - Loading experiment parameters and model configurations
    - Setting up parallel processing for efficient execution
    - Running experiments across multiple prediction horizons
    - Collecting and saving results

    Args:
        args: Command line arguments containing experiment configuration
    """

    assert args.mode in ['deterministic', 'stochastic'], "Mode must be either 'deterministic' or 'stochastic'"
    if args.mode == 'stochastic':
        assert args.uncertainty_value is not None, "Uncertainty scale must be provided for stochastic mode"
        assert (0 <= args.uncertainty_value < 1), "Uncertainty scale values must be between 0 and 1"
    else:
        args.uncertainty_value = 0

    save_path = f"data/{args.project}/{args.mode}/mpc"
    os.makedirs(save_path, exist_ok=True)

    # Load environment and mpc parameters
    env_params = load_env_params(args.env_id)
    mpc_params = load_mpc_params(args.env_id)

    # Get discretization step and nominal parameters
    dt = env_params["dt"]
    p = get_parameters()

    Pred_H = [1, 2, 3, 4, 5, 6, 7, 8]
    seed = 666

    # Define column names for results class
    col_names = [
        "time", "x_0", "x_1", "x_2", "x_3", "y_0", "y_1", "y_2", "y_3",
        "u_0", "u_1", "u_2", "d_0", "d_1", "d_2", "d_3", 
        "J", "econ_rewards", "penalties", "rewards", "solver_times", "solver_success", "run"
    ]

    # Run experiments for each prediction horizon
    for H in tqdm(Pred_H):
        results = Results(col_names)
        if args.mode == "stochastic":
            print(f"Running stochastic case for horizon: {H},\n Uncertainty value: {args.uncertainty_value}")
            N_sims = 10
            save_name = f"{args.save_name}-{H}H-{args.uncertainty_value}.csv"
        else:
            print(f"Running for horizon: {H}")
            N_sims = 1
            save_name = f"{args.save_name}-{H}H.csv"
        # update the prediction horizon
        mpc_params["Np"] = int(H * 3600 / dt)

        def run_experiment(run):
            """
            Executes a single closed-loop MPC experiment for a given prediction horizon and run index.

            This function initializes the MPC controller, defines the nonlinear programming (NLP) problem,
            sets up the experiment with the specified parameters and random seed, solves the NMPC problem,
            and returns the simulation results for the current run.

            Args:
                run (int): The index of the current Monte Carlo simulation run (used for seeding).

            Returns:
                np.ndarray: Simulation results including trajectories and performance metrics for this run.
            """
            mpc = MPC(**env_params, **mpc_params)
            mpc.define_nlp(p)
            rng = np.random.default_rng(seed + run)
            exp = Experiment(mpc, save_name, args.project, args.weather_filename, args.uncertainty_value, p, rng)
            exp.solve_nmpc()
            return exp.get_results(run)

        # Execute parallel simulations
        with Pool(processes=10) as pool:
            data_list = list(tqdm(pool.imap(run_experiment, range(N_sims)), total=N_sims))

        # Collect and save results
        for data in data_list:
            results.update_result(data)
        results.save(join(save_path,save_name))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Closed-loop simulation experiment with varying prediction horizon for MPC"
    )
    parser.add_argument("--project", type=str, default="SMPC",
                       help="Project name for result organization")
    parser.add_argument("--env_id", type=str, default="LettuceGreenhouse",
                       help="Environment identifier")
    parser.add_argument("--save_name", type=str, required=True,
                       help="Base name for saving results")
    parser.add_argument("--mode", type=str, choices=['deterministic', 'stochastic'], required=True,
                       help="Mode for parametric uncertainty")
    parser.add_argument("--uncertainty_value", type=float, required=True,
                       help="Parametric uncertainty level")
    parser.add_argument("--weather_filename", default="outdoorWeatherWurGlas2014.csv", type=str,
                       help="Weather data filename")
    args = parser.parse_args()
