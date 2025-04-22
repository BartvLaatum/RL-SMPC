import os
import argparse

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cmcrameri.cm as cmc

import plot_config

def load_data(model_names, mode, project, Ns=[], uncertainty_value=None):
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
    horizons = ['1H', '2H', '3H', '4H', '5H', '6H']
    data = {
        'mpc': {},
        'smpc': {},
        'smpc-tight': {},
        'rl': {},
        'rlmpc': {},
        'rlsmpc': {}
    }

    uncertainty_suffix = f'-{uncertainty_value}' if uncertainty_value else ''
    # Load data for each model
    for model in model_names:
        # Load RL data
        rl_path = f'data/{project}/{mode}/rl/{model}.csv'
        if os.path.exists(rl_path):
            data['rl'][model] = pd.read_csv(rl_path)
        
        # Load MPC and RL-MPC data for each horizon
        for h in horizons:
            mpc_path = f'data/{project}/{mode}/mpc/mpc-noise-correction-{h}{uncertainty_suffix}.csv'
            rlmpc_path = f'data/{project}/{mode}/rlmpc/rlmpc-{model}-{h}{uncertainty_suffix}.csv'

            if h not in data['mpc'] and os.path.exists(mpc_path):
                data['mpc'][h] = pd.read_csv(mpc_path)

            if os.path.exists(rlmpc_path):
                if h not in data['rlmpc']:
                    data['rlmpc'][h] = {}
                data['rlmpc'][h][model] = pd.read_csv(rlmpc_path)

    # for n in Ns:
    for uncertainty_suffix in ["-0.05", "-0.1"]:
        for h in horizons:
            smpc_path = f'data/{project}/{mode}/smpc/smpc-bounded-states-{h}-10Ns{uncertainty_suffix}.csv'
            # rlsmpc_path = f'data/{project}/{mode}/rlsmpc/rlsmpc-{model}-{h}-10Ns{uncertainty_suffix}.csv'
            if os.path.exists(smpc_path):
                if h not in data['smpc']:
                    data['smpc'][h] = {}
                data['smpc'][h][uncertainty_suffix] = pd.read_csv(smpc_path)
            # if os.path.exists(rlsmpc_path):
            #     if h not in data['rlsmpc']:
            #         data['rlsmpc'][h] = {}
            #     data['rlsmpc'][h][uncertainty_suffix] = pd.read_csv(rlsmpc_path)    
            
    return data, horizons

def create_plot(figure_name, project, data, horizons, model_names, mode, variable, Ns=[], uncertainty_value=None):
    """
    Creates a comparison plot between MPC, RL-MPC and RL approaches across different prediction horizons.
    The function generates a line plot showing the cumulative rewards achieved by different control approaches:
    - Model Predictive Control (MPC)
    - Reinforcement Learning with MPC (RL-MPC) 
    - Pure Reinforcement Learning (RL)
    Parameters
    ----------
    project : str
        Name of the project directory where figures will be saved
    data : dict
        Dictionary containing the results data with the following structure:
        {
            'mpc': {horizon: {'rewards': [...]}},
            'rlmpc': {horizon: {model_name: {'rewards': [...]}}},
            'rl': {model_name: {'rewards': [...]}}
        }
    horizons : list
        List of prediction horizon values to plot
    model_names : list
        List of model names used in the RL and RL-MPC approaches
    mode : str
        Either 'stochastic' or 'deterministic' to indicate the environment type
    uncertainty_value : float, optional
        Scale of uncertainty for stochastic environments, used in plot title
    Returns
    -------
    None
        Displays the plot and optionally saves it to disk
    """
    WIDTH = 87.5 * 0.03937
    HEIGHT = WIDTH * 0.75
    color_counter  = 0
    fig, ax = plt.subplots(figsize=(WIDTH, HEIGHT), dpi=300)
    
    # n_colors = max(len(data.keys()) + len(model_names), 8)  # Ensure at least 8 colors
    colors = cmc.batlowS
    horizon_nums = [int(h[0]) for h in horizons]

    # Plot MPC results
    mean_mpc_final_rewards = []
    std_mpc_final_rewards = []
    for h in horizons:
        if h in data['mpc']:
            grouped_runs = data['mpc'][h].groupby("run")
            cumulative_rewards = grouped_runs[variable].sum()

            mean_mpc_final_rewards.append(cumulative_rewards.mean())
            std_mpc_final_rewards.append(cumulative_rewards.std())

    if mean_mpc_final_rewards:
        ax.plot(horizon_nums, mean_mpc_final_rewards, 'o-', label='MPC', color=colors(color_counter))
        if mode == "stochastic":
            ax.fill_between(
                horizon_nums, 
                np.array(mean_mpc_final_rewards) - np.array(std_mpc_final_rewards),
                np.array(mean_mpc_final_rewards) + np.array(std_mpc_final_rewards),
                color=colors(color_counter), alpha=0.2
            )
        color_counter += 1

    # Plot Stochastic MPC results
    mean_smpc_final_rewards = []
    std_smpc_final_rewards = []
    for h in horizons:
        if h in data['smpc']:
            # if n in data['smpc'][h]:
            grouped_runs = data['smpc'][h].groupby("run")
            cumulative_rewards = grouped_runs[variable].sum()

            mean_smpc_final_rewards.append(cumulative_rewards.mean())
            std_smpc_final_rewards.append(cumulative_rewards.std())
    n2plot = len(mean_smpc_final_rewards)
    if mean_smpc_final_rewards:
        ax.plot(horizon_nums[:n2plot], mean_smpc_final_rewards[:n2plot], 'o-', label=f'SMPC', color=colors(color_counter))
        if mode == "stochastic":
            ax.fill_between(
                horizon_nums[:n2plot], 
                np.array(mean_smpc_final_rewards[:n2plot]) - np.array(std_smpc_final_rewards[:n2plot]),
                np.array(mean_smpc_final_rewards[:n2plot]) + np.array(std_smpc_final_rewards[:n2plot]),
                color=colors(color_counter), alpha=0.3
            )
        color_counter += 1

    # Plot Stochastic RL-SMPC zero-order results
    for idx, model in enumerate(model_names):
        mean_rlmpc_final_rewards = []
        std_rlmpc_final_rewards = []

        for h in horizons:
            if h in data['rl-zero-terminal-smpc']:
                grouped_runs = data['rl-zero-terminal-smpc'][h][model].groupby("run")
                cumulative_rewards = grouped_runs[variable].sum()

                mean_rlmpc_final_rewards.append(cumulative_rewards.mean())
                std_rlmpc_final_rewards.append(cumulative_rewards.std())

        n2plot = len(mean_rlmpc_final_rewards)
        if mean_rlmpc_final_rewards:
            ax.plot(horizon_nums[:n2plot], mean_rlmpc_final_rewards[:n2plot], 'o-', label=r'RL$^0$-SMPC', color=colors(color_counter))
            if mode == "stochastic":
                ax.fill_between(
                    horizon_nums[:n2plot], 
                    np.array(mean_rlmpc_final_rewards[:n2plot]) - np.array(std_rlmpc_final_rewards[:n2plot]),
                    np.array(mean_rlmpc_final_rewards[:n2plot]) + np.array(std_rlmpc_final_rewards[:n2plot]),
                    color=colors(color_counter), alpha=0.3
                )
            color_counter += 1


    # Plot Stochastic RL-SMPC first-order results
    for idx, model in enumerate(model_names):
        mean_rlmpc_final_rewards = []
        std_rlmpc_final_rewards = []

        for h in horizons:
            if h in data['rl-first-terminal-smpc']:
                grouped_runs = data['rl-first-terminal-smpc'][h][model].groupby("run")
                cumulative_rewards = grouped_runs[variable].sum()

                mean_rlmpc_final_rewards.append(cumulative_rewards.mean())
                std_rlmpc_final_rewards.append(cumulative_rewards.std())

        n2plot = len(mean_rlmpc_final_rewards)
        if mean_rlmpc_final_rewards:
            ax.plot(horizon_nums[:n2plot], mean_rlmpc_final_rewards[:n2plot], 'o-', label=r'RL$^1$-SMPC', color=colors(color_counter))
            if mode == "stochastic":
                ax.fill_between(
                    horizon_nums[:n2plot], 
                    np.array(mean_rlmpc_final_rewards[:n2plot]) - np.array(std_rlmpc_final_rewards[:n2plot]),
                    np.array(mean_rlmpc_final_rewards[:n2plot]) + np.array(std_rlmpc_final_rewards[:n2plot]),
                    color=colors(color_counter), alpha=0.3
                )
            color_counter += 1

    # Plot RL-MPC and RL results for each model
    for idx, model in enumerate(model_names, 1+2*len(Ns)):
        # Plot RL-MPC
        mean_rlmpc_final_rewards = []
        std_rlmpc_final_rewards = []
        for h in horizons:
            if h in data['rlmpc'] and model in data['rlmpc'][h]:
                grouped_runs = data['rlmpc'][h][model].groupby("run")
                cumulative_rewards = grouped_runs[variable].sum()

                mean_rlmpc_final_rewards.append(cumulative_rewards.mean())
                std_rlmpc_final_rewards.append(cumulative_rewards.std())
        n2plot = len(mean_rlmpc_final_rewards)
        if mean_rlmpc_final_rewards:
            ax.plot(horizon_nums[:n2plot], mean_rlmpc_final_rewards[:n2plot], 'o-', 
                   label=f'RL-MPC ({model})', color=colors(color_counter))
            if mode == "stochastic":
                ax.fill_between(
                    horizon_nums[:n2plot],
                    np.array(mean_rlmpc_final_rewards[:n2plot]) - np.array(std_rlmpc_final_rewards[:n2plot]),
                    np.array(mean_rlmpc_final_rewards[:n2plot]) + np.array(std_rlmpc_final_rewards[:n2plot]),
                    color=colors(color_counter), alpha=0.3
                )

    for idx, model in enumerate(model_names):
        # Plot RL horizontal line
        if model in data['rl']:
            sum_rewards = data['rl'][model].groupby("run")[variable].sum()
            rl_final_reward = sum_rewards.mean()
            ax.hlines(rl_final_reward, min(horizon_nums), max(horizon_nums),
                        label=f'RL', color=colors(color_counter), linestyle='--')
        color_counter += 1

    ax.set_xlabel('Prediction Horizon (H)')
    if variable == 'rewards':
        ax.set_ylabel(f'Cumulative {variable[:-1]}')
        ax.legend()
    elif variable == 'econ_rewards':
        ax.set_ylabel(f'Cumulative EPI')
    elif variable == 'penalties':
        ax.set_ylabel(f'Cumulative penalty')

    # if mode == 'stochastic':

    #     ax.set_title(f"Stochastic environment ($\delta={uncertainty_value}$)")
    # else:
    #     ax.set_title('Deterministic environment')
    ax.grid()
    plt.tight_layout()
    
    # Save plot
    dir_path = f'figures/{project}/{mode}/{figure_name}/'
    os.makedirs(dir_path, exist_ok=True)
    if variable == "econ_rewards":
        variable = "EPI"
    uncertainty_suffix = f'-{uncertainty_value}' if uncertainty_value else ''
    plt.savefig(f'{dir_path}{variable}{uncertainty_suffix}.png', 
                bbox_inches='tight', dpi=300)
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=str, default='matching-thesis',
                        help='Name of the project')
    parser.add_argument('--model_names', nargs='+', type=str, default=[],
                        help='List of model names to plot')
    parser.add_argument('--Ns', nargs='+', type=int, default=[],
                        help='List of SMPC with Ns scenario samples to plot')
    parser.add_argument('--mode', type=str, 
                        choices=['deterministic', 'stochastic'], required=True)
    parser.add_argument('--uncertainty_value', type=float,
                        help='Uncertainty scale value for stochastic mode')
    parser.add_argument('--figure_name', type=str, required=True)
    args = parser.parse_args()

    data, horizons = load_data(args.model_names, args.mode, args.project, args.Ns, args.uncertainty_value)
    create_plot(args.figure_name, args.project, data, horizons, args.model_names, args.mode, 'rewards', args.Ns, args.uncertainty_value)
    create_plot(args.figure_name, args.project, data, horizons, args.model_names, args.mode, 'econ_rewards', args.Ns, args.uncertainty_value)
    create_plot(args.figure_name, args.project, data, horizons, args.model_names, args.mode, 'penalties', args.Ns, args.uncertainty_value)

if __name__ == "__main__":
    main()
