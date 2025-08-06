import os
import argparse

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cmcrameri.cm as cmc

from performance_plots import create_plot
from visualisations.rl_smpc_performance import load_data
import plot_config

def load_data(
        model_names, 
        mode, 
        project,
        mpc=True,
        smpc=True,
        zero_order=True,
        terminal=False,
        first_order=False, 
        Ns=[], 
        uncertainty_value=None
    ):
    """
    Load and organize simulation data from MPC, RL, and RL-MPC experiments.
    This function reads CSV files containing results from different control strategies
    and organizes them into a nested dictionary structure.
    Parameters
    ----------
    model_names : list
        List of RL model names to load data for
    mode : str
        Operating mode of the simulation (e.g., 'train', 'test')  
    project : str
        Project name/folder containing the data
    uncertainty_value : float, optional
        Scale factor for uncertainty in MPC predictions. If provided, loads data 
        with specified uncertainty scale suffix.
    Returns
    -------
    tuple
        - data : dict
            Nested dictionary containing loaded dataframes organized by:
            - 'mpc': Dict of dataframes indexed by horizon
            - 'rl': Dict of dataframes indexed by model name  
            - 'rlmpc': Dict of dicts indexed by horizon then model name
        - horizons : list
            List of horizon values used in the simulations
    Notes
    -----
    Expected file structure:
    data/
        {project}/
            {mode}/
                rl/
                    {model}.csv
                mpc/
                    mpc-{horizon}-{uncertainty}.csv
                rlmpc/
                    rlmpc-{model}-{horizon}-{uncertainty}.csv
    """
    horizons = ['1H', '2H', '3H', '4H', '5H', '6H', '7H', '8H']
    data = {
        'mpc': {},
        'smpc': {},
        'rl': {},
        'rl-zero-terminal-smpc': {},
        'rl-first-terminal-smpc': {},
    }

    uncertainty_suffix = f'-{uncertainty_value}' if uncertainty_value else ''

    for model in model_names:
        # Load RL data
        rl_path = f'data/{project}/{mode}/rl/{model}.csv'
        if os.path.exists(rl_path):
            data['rl'][model] = pd.read_csv(rl_path)

        # Load MPC and RL-MPC data for each horizon
        for h in horizons:
            rlsmpc_terminal_path = f'data/{project}/{mode}/rlsmpc/{model}-no-tightening-{h}{uncertainty_suffix}.csv'

            if zero_order:
                if terminal:
                    if os.path.exists(rlsmpc_terminal_path):
                        if h not in data['rl-zero-terminal-smpc']:
                            data['rl-zero-terminal-smpc'][h] = {}
                        data['rl-zero-terminal-smpc'][h][model] = pd.read_csv(rlsmpc_terminal_path)

    for h in horizons:
        smpc_path = f'data/{project}/{mode}/smpc/no-tightening-{h}{uncertainty_suffix}.csv'

        if smpc:
            if os.path.exists(smpc_path):
                if h not in data['smpc']:
                    data['smpc'][h] = {}
                data['smpc'][h]= pd.read_csv(smpc_path)

    return data, horizons


def runtime_performance_plot(data, horizons, model_names):
    WIDTH = 60 * 0.0393700787
    HEIGHT = WIDTH * 0.75
    color_counter  = 0
    fig, ax = plt.subplots(figsize=(WIDTH, HEIGHT), dpi=300)

    mean_smpc_runtime = []
    mean_smpc_final_reward = []
    std_smpc_runtime = []
    std_smpc_final_reward = []
    for h in horizons:
        if h in data['smpc']:
            # if n in data['smpc'][h]:
            grouped_runs = data['smpc'][h].groupby("run")
            average_runtime = grouped_runs["solver_times"].mean()
            cumulative_rewards = grouped_runs["rewards"].sum()

            mean_smpc_runtime.append(average_runtime.mean())
            mean_smpc_final_reward.append(cumulative_rewards.mean())
            std_smpc_runtime.append(average_runtime.std())
            std_smpc_final_reward.append(cumulative_rewards.std())

    ax.errorbar(
        mean_smpc_runtime,
        mean_smpc_final_reward,
        xerr=std_smpc_runtime,
        yerr=std_smpc_final_reward,
        fmt='o',
        color="C0",
        label="SMPC"
    )

    for idx, model in enumerate(model_names):
        mean_rl_smpc_runtime = []
        mean_rl_smpc_final_reward = []
        for h in horizons:
            if h in data['rl-zero-terminal-smpc']:
                grouped_runs = data['rl-zero-terminal-smpc'][h][model].groupby("run")
                average_runtime = grouped_runs["solver_times"].mean()
                cumulative_rewards = grouped_runs["rewards"].sum()
                mean_rl_smpc_runtime.append(average_runtime.mean())
                mean_rl_smpc_final_reward.append(cumulative_rewards.mean())

        ax.scatter(mean_rl_smpc_runtime, mean_rl_smpc_final_reward, c="C3", label="RL-SMPC")
    ax.set_xlabel("Average calculation time (s)")
    ax.legend()
    ax.set_ylabel("Cumulative reward")
    plt.tight_layout()
    plt.savefig(f"figures/{args.figure_name}.svg", format='svg', bbox_inches='tight', dpi=300)
    plt.show()


def main(args):
    data, horizons = load_data(
            args.model_names, 
            args.mode, 
            args.project, 
            mpc=False,
            smpc=args.smpc,
            zero_order=args.zero_order,
            first_order=False,
            terminal=args.terminal, 
            Ns=[], 
            uncertainty_value=args.uncertainty_value
        )    
    runtime_performance_plot(data, horizons, args.model_names)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=str, default='SMPC',
                        help='Name of the project')
    parser.add_argument('--model_names', nargs='+', type=str, default=[],
                        help='List of model names to plot')
    parser.add_argument('--smpc', action=argparse.BooleanOptionalAction,
                        help='Whether to plot SMPC')
    parser.add_argument('--zero-order', action=argparse.BooleanOptionalAction,
                        help='Whether to RL-SMPC with zero-order approximation')
    parser.add_argument('--first-order', action=argparse.BooleanOptionalAction,
                        help='Whether to RL-SMPC with fisst-order approximation')
    parser.add_argument('--terminal', action=argparse.BooleanOptionalAction,
                        help='Whether to visualise RL-SMPC with terminal state/cost implemented')
    parser.add_argument('--Ns', nargs='+', type=int, default=[],
                        help='List of SMPC with Ns scenario samples to plot')
    parser.add_argument('--mode', type=str, 
                        choices=['deterministic', 'stochastic'], required=True)
    parser.add_argument('--uncertainty_value', type=float,
                        help='Uncertainty scale value for stochastic mode')
    parser.add_argument('--figure_name', type=str, required=True)
    args = parser.parse_args()
    main(args)
