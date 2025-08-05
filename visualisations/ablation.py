import os
import argparse

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cmcrameri.cm as cmc

import plot_config
from matplotlib.colors import LinearSegmentedColormap

def load_data(
        model_names, 
        mode, 
        project,
        uncertainty_value=None
    ):
    """
    Load and organize simulation results for various RL-SMPC ablation experiments.

    This function reads CSV files containing results from different RL-SMPC ablation
    variants (full RL-SMPC, no value function, no terminal cost, no feedback) for
    multiple models and prediction horizons. It organizes the loaded data into a
    nested dictionary structure for easy access and further analysis.

    Args:
        model_names (list of str): List of RL model names to load data for.
        mode (str): Simulation mode or subfolder (e.g., 'deterministic', 'stochastic').
        project (str): Project name/folder containing the data.
        uncertainty_value (float, optional): Uncertainty scale value. If provided,
            loads data files with the corresponding uncertainty suffix.

    Returns:
        tuple:
            - data (dict): Nested dictionary containing loaded pandas DataFrames,
                organized as:
                    {
                        'rl-smpc': {horizon: {model: DataFrame}},
                        'rl-smpc-no-vf': {horizon: {model: DataFrame}},
                        'rl-smpc-no-terminal': {horizon: {model: DataFrame}},
                        'rl-smpc-no-feedback': {horizon: {model: DataFrame}},
                    }
            - horizons (list of str): List of horizon values used in the simulations.
    """
    horizons = ['1H', '2H', '3H', '4H', '5H', '6H', '7H', '8H']
    data = {
        'rl-smpc': {},
        'rl-smpc-no-vf': {},
        'rl-smpc-no-terminal': {},
        'rl-smpc-no-feedback': {},
    }

    uncertainty_suffix = f'-{uncertainty_value}' if uncertainty_value else ''

    for model in model_names:
        # Load MPC and RL-MPC data for each horizon
        for h in horizons:
            rlsmpc_path = f'data/{project}/{mode}/rlsmpc/{model}-no-tightening-{h}{uncertainty_suffix}.csv'
            rlsmpc_novf_path = f'data/{project}/{mode}/rlsmpc/{model}-no-vf-{h}{uncertainty_suffix}.csv'
            rlsmpc_noterminal_path = f'data/{project}/{mode}/rlsmpc/{model}-no-terminal-{h}{uncertainty_suffix}.csv'
            rlsmpc_no_feedback_path = f'data/{project}/{mode}/rlsmpc/{model}-no-feedback-{h}{uncertainty_suffix}.csv'

            if os.path.exists(rlsmpc_path):
                if h not in data['rl-smpc']:
                    data['rl-smpc'][h] = {}
                data['rl-smpc'][h][model] = pd.read_csv(rlsmpc_path)

            if os.path.exists(rlsmpc_novf_path):
                if h not in data['rl-smpc-no-vf']:
                    data['rl-smpc-no-vf'][h] = {}
                data['rl-smpc-no-vf'][h][model] = pd.read_csv(rlsmpc_novf_path)

            if os.path.exists(rlsmpc_noterminal_path):
                if h not in data['rl-smpc-no-terminal']:
                    data['rl-smpc-no-terminal'][h] = {}
                data['rl-smpc-no-terminal'][h][model] = pd.read_csv(rlsmpc_noterminal_path)

            if os.path.exists(rlsmpc_no_feedback_path):
                if h not in data['rl-smpc-no-feedback']:
                    data['rl-smpc-no-feedback'][h] = {}
                data['rl-smpc-no-feedback'][h][model] = pd.read_csv(rlsmpc_no_feedback_path)


    return data, horizons


def runtime_performance_plot(data, horizons, model_names, variable):
    """
    Create performance comparison plots for RL-SMPC ablation studies.

    This function generates line plots comparing the performance of different RL-SMPC
    variants across prediction horizons. It computes cumulative performance metrics
    (rewards, penalties, economic rewards) for each ablation variant and creates
    visualizations with confidence intervals.

    The function plots four RL-SMPC variants:
    - Full RL-SMPC (baseline)
    - RL-SMPC without value function
    - RL-SMPC without feedback correction
    - RL-SMPC without terminal cost

    Args:
        data (dict): Nested dictionary containing loaded simulation data organized by
            ablation variant, horizon, and model name.
        horizons (list): List of prediction horizon values (e.g., ['1H', '2H', ...]).
        model_names (list): List of RL model names to include in the analysis.
        variable (str): Performance metric to plot ('rewards', 'penalties', 'econ_rewards').

    Returns:
        None

    Notes:
        - Computes 99% confidence intervals using standard error of the mean
        - Uses consistent color scheme across all ablation variants
        - Saves plots in both SVG and PNG formats
        - Automatically creates output directory structure
        - Handles missing data gracefully by skipping unavailable combinations
        - Plots are saved with uncertainty value suffix for identification
    """
    WIDTH = 60 * 0.0393700787
    HEIGHT = WIDTH * 0.75
    color_counter  = 0
    fig, ax = plt.subplots(figsize=(WIDTH, HEIGHT), dpi=300)
    horizon_nums = [int(h[0]) for h in horizons]

    colors_list = [
        "C3",  # very pale blue
        "#8FBBD9",  # light blue
        "#5799C7",  # medium blue
        "#1F77B4",  # your original blue
    ]

    mean_rlsmpc_final_reward = []
    ci_rlsmpc_final_reward = []
    for idx, model in enumerate(model_names):
        for h in horizons:
            if h in data['rl-smpc']:
                # if n in data['smpc'][h]:
                grouped_runs = data['rl-smpc'][h][model].groupby("run")
                cumulative_rewards = grouped_runs[variable].sum()
                ci = cumulative_rewards.std() / np.sqrt(len(cumulative_rewards))*2.58
                ci_rlsmpc_final_reward.append(ci)
                mean_rlsmpc_final_reward.append(cumulative_rewards.mean())
    n2plot = len(mean_rlsmpc_final_reward)
    ax.plot(horizon_nums[:n2plot], mean_rlsmpc_final_reward[:n2plot], 'o-', label='RL-SMPC', color=colors_list[0], alpha=0.8)
    ax.fill_between(
        horizon_nums[:n2plot], 
        np.array(mean_rlsmpc_final_reward[:n2plot]) - np.array(ci_rlsmpc_final_reward[:n2plot]),
        np.array(mean_rlsmpc_final_reward[:n2plot]) + np.array(ci_rlsmpc_final_reward[:n2plot]),
        color=colors_list[0], alpha=0.3
    )

    for idx, model in enumerate(model_names):
        mean_rl_smpc_no_vf_final_reward = []
        ci_rl_smpc_no_vf_final_reward = []
        for h in horizons:
            if h in data['rl-smpc-no-vf']:
                grouped_runs = data['rl-smpc-no-vf'][h][model].groupby("run")
                cumulative_rewards = grouped_runs[variable].sum()
                ci = cumulative_rewards.std() / np.sqrt(len(cumulative_rewards))*2.58
                ci_rl_smpc_no_vf_final_reward.append(ci)

                mean_rl_smpc_no_vf_final_reward.append(cumulative_rewards.mean())
    n2plot = len(mean_rl_smpc_no_vf_final_reward)
    ax.plot(horizon_nums[:n2plot], mean_rl_smpc_no_vf_final_reward[:n2plot], 'o--', label=r'No $\tilde{J}_{\phi}$', color=colors_list[1], alpha=0.8)
    ax.fill_between(
        horizon_nums[:n2plot], 
        np.array(mean_rl_smpc_no_vf_final_reward[:n2plot]) - np.array(ci_rl_smpc_no_vf_final_reward[:n2plot]),
        np.array(mean_rl_smpc_no_vf_final_reward[:n2plot]) + np.array(ci_rl_smpc_no_vf_final_reward[:n2plot]),
        color=colors_list[1], alpha=0.3
    )

    for idx, model in enumerate(model_names):
        mean_rl_smpc_no_feedback_final_reward = []
        ci_rl_smpc_no_feedback_reward = []
        for h in horizons:
            if h in data['rl-smpc-no-feedback']:
                grouped_runs = data['rl-smpc-no-feedback'][h][model].groupby("run")
                cumulative_rewards = grouped_runs[variable].sum()
                # print(grouped_runs["solver_success"].mean())
                ci = cumulative_rewards.std() / np.sqrt(len(cumulative_rewards))*2.58
                ci_rl_smpc_no_feedback_reward.append(ci)

                mean_rl_smpc_no_feedback_final_reward.append(cumulative_rewards.mean())
    n2plot = len(mean_rl_smpc_no_feedback_final_reward)
    ax.plot(horizon_nums[:n2plot], mean_rl_smpc_no_feedback_final_reward[:n2plot], 'o-', label=r'No $\hat{u}$', color=colors_list[2], alpha=0.8)
    ax.fill_between(
        horizon_nums[:n2plot], 
        np.array(mean_rl_smpc_no_feedback_final_reward[:n2plot]) - np.array(ci_rl_smpc_no_feedback_reward[:n2plot]),
        np.array(mean_rl_smpc_no_feedback_final_reward[:n2plot]) + np.array(ci_rl_smpc_no_feedback_reward[:n2plot]),
        color=colors_list[2], alpha=0.3
    )

    for idx, model in enumerate(model_names):
        mean_rl_smpc_no_terminal_final_reward = []
        ci_rl_smpc_no_terminal_reward = []
        for h in horizons:
            if h in data['rl-smpc-no-terminal']:
                grouped_runs = data['rl-smpc-no-terminal'][h][model].groupby("run")
                cumulative_rewards = grouped_runs[variable].sum()
                mean_rl_smpc_no_terminal_final_reward.append(cumulative_rewards.mean())
                ci = cumulative_rewards.std() / np.sqrt(len(cumulative_rewards))*2.58
                ci_rl_smpc_no_terminal_reward.append(ci)


    n2plot = len(mean_rl_smpc_no_terminal_final_reward)
    ax.plot(horizon_nums[:n2plot], mean_rl_smpc_no_terminal_final_reward[:n2plot], 'o-', label='No $X_N$', color=colors_list[3], alpha=0.8)
    ax.fill_between(
        horizon_nums[:n2plot], 
        np.array(mean_rl_smpc_no_terminal_final_reward[:n2plot]) - np.array(ci_rl_smpc_no_terminal_reward[:n2plot]),
        np.array(mean_rl_smpc_no_terminal_final_reward[:n2plot]) + np.array(ci_rl_smpc_no_terminal_reward[:n2plot]),
        color=colors_list[3], alpha=0.3
    )



    ax.set_xlabel("Horizon (H)")
    ax.set_ylabel("Cumulative reward")
    fig.tight_layout()
    # Save plot
    dir_path = f'figures/{args.project}/{args.mode}/{args.figure_name}/'
    os.makedirs(dir_path, exist_ok=True)

    if variable == 'rewards':
        ax.set_ylabel(f'Cumulative {variable[:-1]}')
        ax.legend()
    elif variable == 'econ_rewards':
        ax.set_ylabel(f'Cumulative EPI (EU/m$^2$)')
        variable = "EPI"
    elif variable == 'penalties':
        ax.set_ylabel(f'Cumulative penalty')

    fig.savefig(f'{dir_path}{variable}-{args.uncertainty_value}.svg', format='svg',
                bbox_inches='tight', dpi=300)
    fig.savefig(f'{dir_path}{variable}-{args.uncertainty_value}.png', format='png',
                bbox_inches='tight', dpi=300)
    plt.show()

def main(args):
    data, horizons = load_data(
            args.model_names, 
            args.mode, 
            args.project, 
            uncertainty_value=args.uncertainty_value
        )    
    runtime_performance_plot(data, horizons, args.model_names, "rewards")
    runtime_performance_plot(data, horizons, args.model_names, "penalties")
    runtime_performance_plot(data, horizons, args.model_names, "econ_rewards")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=str, default='SMPC',
                        help='Name of the project')
    parser.add_argument('--model_names', nargs='+', type=str, default=[],
                        help='List of model names to plot')
    parser.add_argument('--mode', type=str, 
                        choices=['deterministic', 'stochastic'], required=True)
    parser.add_argument('--uncertainty_value', type=float,
                        help='Uncertainty scale value for stochastic mode')
    parser.add_argument('--figure_name', type=str, required=True)
    args = parser.parse_args()
    main(args)
