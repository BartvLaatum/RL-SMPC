import os
import argparse

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cmcrameri.cm as cmc

import plot_config

def load_data(model_names, mode, project, uncertainty_value=None):
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
        'rl': {},
        'rlmpc': {}
    }

    # Load data for each model
    for model in model_names:
        # Load RL data
        rl_path = f'data/{project}/{mode}/rl/{model}.csv'
        if os.path.exists(rl_path):
            data['rl'][model] = pd.read_csv(rl_path)
        
        # Load MPC and RL-MPC data for each horizon
        for h in horizons:
            uncertainty_suffix = f'-{uncertainty_value}' if uncertainty_value else ''

            mpc_path = f'data/{project}/{mode}/mpc/mpc-{h}{uncertainty_suffix}.csv'
            rlmpc_path = f'data/{project}/{mode}/rlmpc/rlmpc-{model}-{h}{uncertainty_suffix}.csv'

            if h not in data['mpc'] and os.path.exists(mpc_path):
                data['mpc'][h] = pd.read_csv(mpc_path)
            
            if os.path.exists(rlmpc_path):
                if h not in data['rlmpc']:
                    data['rlmpc'][h] = {}
                data['rlmpc'][h][model] = pd.read_csv(rlmpc_path)
    
    return data, horizons

def create_plot(project, data, horizons, model_names, mode, uncertainty_value=None):
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
    fig, ax = plt.subplots(figsize=(WIDTH, HEIGHT), dpi=300)
    
    colors = cmc.batlow(np.linspace(0, 1, len(model_names) + 1))
    horizon_nums = [int(h[0]) for h in horizons]

    # Plot MPC results
    mean_mpc_final_rewards = []
    std_mpc_final_rewards = []
    for h in horizons:
        if h in data['mpc']:
            grouped_runs = data['mpc'][h].groupby("run")
            cumulative_rewards = grouped_runs['rewards'].sum()

            mean_mpc_final_rewards.append(cumulative_rewards.mean())
            std_mpc_final_rewards.append(cumulative_rewards.std())

    if mean_mpc_final_rewards:
        ax.plot(horizon_nums, mean_mpc_final_rewards, 'o-', label='MPC', color=colors[0])
        if mode == "stochastic":
            ax.fill_between(
                horizon_nums, 
                np.array(mean_mpc_final_rewards) - np.array(std_mpc_final_rewards),
                np.array(mean_mpc_final_rewards) + np.array(std_mpc_final_rewards),
                color=colors[0], alpha=0.2
            )


    # Plot RL-MPC and RL results for each model
    for idx, model in enumerate(model_names, 1):
        # Plot RL-MPC
        mean_rlmpc_final_rewards = []
        std_rlmpc_final_rewards = []
        for h in horizons:
            if h in data['rlmpc'] and model in data['rlmpc'][h]:
                grouped_runs = data['rlmpc'][h][model].groupby("run")
                cumulative_rewards = grouped_runs['rewards'].sum()

                mean_rlmpc_final_rewards.append(cumulative_rewards.mean())
                std_rlmpc_final_rewards.append(cumulative_rewards.std())

        if mean_rlmpc_final_rewards:
            ax.plot(horizon_nums[:5], mean_rlmpc_final_rewards, 'o-', 
                   label=f'RL-MPC ({model})', color=colors[idx])
            if mode == "stochastic":
                ax.fill_between(
                    horizon_nums[:5],
                    np.array(mean_rlmpc_final_rewards) - np.array(std_rlmpc_final_rewards),
                    np.array(mean_rlmpc_final_rewards) + np.array(std_rlmpc_final_rewards),
                    color=colors[0], alpha=0.2
                )


        # Plot RL horizontal line
        if model in data['rl']:
            sum_rewards = data['rl'][model].groupby("run")['rewards'].sum()
            rl_final_reward = sum_rewards.mean()
            ax.hlines(rl_final_reward, min(horizon_nums), max(horizon_nums),
                     label=f'RL ({model})', color=colors[idx], linestyle='--')

    ax.set_xlabel('Prediction Horizon (H)')
    ax.set_ylabel('Cumulative reward')
    if mode == 'stochastic':
        ax.set_title(f"Stochastic environment ($\delta={uncertainty_value}$)")
    else:
        ax.set_title('Deterministic environment')
    ax.legend()
    ax.grid()
    plt.tight_layout()
    
    # Save plot
    dir_path = f'figures/{project}/{mode}/'
    os.makedirs(dir_path, exist_ok=True)
    uncertainty_suffix = f'-{uncertainty_value}' if uncertainty_value else ''
    # plt.savefig(f'{dir_path}rl-mpc-comparison{uncertainty_suffix}.png', 
    #             bbox_inches='tight', dpi=300)
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=str, default='matching-thesis',
                        help='Name of the project')
    parser.add_argument('--model_names', nargs='+', required=True,
                        help='List of model names to plot')
    parser.add_argument('--mode', type=str, 
                        choices=['deterministic', 'stochastic'], required=True)
    parser.add_argument('--uncertainty_value', type=float,
                        help='Uncertainty scale value for stochastic mode')
    args = parser.parse_args()

    data, horizons = load_data(args.model_names, args.mode, args.project, args.uncertainty_value)
    create_plot(args.project, data, horizons, args.model_names, args.mode, args.uncertainty_value)

if __name__ == "__main__":
    main()
