import os
import argparse

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cmcrameri.cm as cmc

import plot_config
import numpy as np
from tabulate import tabulate

def load_data(model_names, mode, project, Ns=[], uncertainty_value=None):
    """
    Load and organize simulation data from MPC, SMPC, RL, and RL-MPC experiments.
    This function reads CSV files containing results from different control strategies
    into pandas DataFrames and organizes them into a nested dictionary structure.

    Args:
        model_names (list): List of RL model names to load data for
        mode (str): Operating mode of the simulation (e.g., 'train', 'test')  
        project (str): Project name/folder containing the data
        uncertainty_value (float, optional): 
            Scale factor for uncertainty in MPC predictions. If provided, loads data 
            with specified uncertainty scale suffix.
    
    Returns:
        (tuple)
            - data : dict
                Nested dictionary containing loaded dataframes organized by:
                - 'mpc': Dict of dataframes indexed by horizon
                - 'smpc': Dict of dataframes indexed by horizon
                - 'rl': Dict of dataframes indexed by model name  
                - 'rlsmpc': Dict of dicts indexed by model name then the horizon
            - horizons : list
                List of horizon values used in the simulations

    Notes:
        Expected file structure:
        data/
            {project}/
                {mode}/
                    rl/
                        {model_name}.csv
                    mpc/
                        mpc-{horizon}-{uncertainty}.csv
                    smpc/
                        smpc-bounded-states-{horizon}.csv
                    rlsmpc/
                        {model_name}-{horizon}-{uncertainty}.csv
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
            
    return data, horizons

def compute_confidence_interval(samples):
    """
    Computes the 99% confidence interval for a given set of samples.

    Args:
        samples (list): Samples to compute CI over.

    Returns:
        np.ndarray: Arrat with the calculated 99% CIs
    """
    samples = np.asarray(samples)
    std = np.std(samples)
    n_samples = len(samples)
    sem = std / np.sqrt(n_samples)
    
    return 2.58 * sem

def create_plot(
        figure_name,
        project, 
        data,
        horizons, 
        model_names, 
        mode, 
        variable, 
        Ns=[], 
        uncertainty_value=None,
    ):
    """
    Creates a comparison plot between MPC, SMPC, RL and RL-SMPC approaches across different prediction horizons.

    This function generates a line plot showing the cumulative rewards, EPI, or penalty achieved by different control approaches:
    For each approach, the function computes the mean, standard deviation, and confidence interval of the cumulative rewards across 
    runs for each prediction horizon. It then plots these statistics, including error bands for stochastic environments, and prints
    summary tables to the console. The resulting figure is saved to disk.

    Args:
        figure_name (str): Name for the output figure files.
    project (str): Name of the project directory where figures will be saved.
    data (dict): Dictionary containing the results data with the following structure:
            {
                'mpc': {horizon: DataFrame},
                'smpc': {horizon: DataFrame},
                'rl-zero-terminal-smpc': {horizon: {model_name: DataFrame}},
                'rl': {model_name: DataFrame}
            }
    horizons (list): List of prediction horizon values to plot.
    model_names (list): List of model names used in the RL and RL-MPC approaches.
    mode (str): Either 'stochastic' or 'deterministic' to indicate the environment type.
    variable (str): The metric to plot (e.g., 'rewards', 'econ_rewards', 'penalties').
    Ns (list, optional) Not used in this function, but included for compatibility.
    uncertainty_value (float, optional): Scale of uncertainty for stochastic environments, used in plot title and filenames.

    Returns: None
        Displays the plot and saves it to disk.
    """
    WIDTH = 60 * 0.0393700787
    HEIGHT = WIDTH * 0.75
    color_counter  = 0
    fig, ax = plt.subplots(figsize=(WIDTH, HEIGHT), dpi=300)

    # Convert horizon labels to integers for plotting
    horizon_nums = [int(h[0]) for h in horizons]

    # --- Plot MPC results ---
    mean_mpc_final_rewards = []
    std_mpc_final_rewards = []
    ci_mpc_final_rewards = []
    for h in horizons:
        if h in data['mpc']:
            # Group by run and sum the specified variable to get cumulative reward per run
            grouped_runs = data['mpc'][h].groupby("run")
            cumulative_rewards = grouped_runs[variable].sum()

            mean_mpc_final_rewards.append(cumulative_rewards.mean())
            std_mpc_final_rewards.append(cumulative_rewards.std())
            ci = compute_confidence_interval(cumulative_rewards)
            ci_mpc_final_rewards.append(ci)

    if mean_mpc_final_rewards:
        n2plot = len(mean_mpc_final_rewards)
        # Plot mean cumulative metric for MPC
        ax.plot(horizon_nums[:n2plot], mean_mpc_final_rewards[:n2plot], 'o-', label='MPC', color="#00a693", alpha=0.8)
        # Plot confidence interval as shaded region for stochastic mode
        if mode == "stochastic":
            ax.fill_between(
                horizon_nums[:n2plot], 
                np.array(mean_mpc_final_rewards[:n2plot]) - np.array(ci_mpc_final_rewards[:n2plot]),
                np.array(mean_mpc_final_rewards[:n2plot]) + np.array(ci_mpc_final_rewards[:n2plot]),
                color="#00a693", alpha=0.2
            )
        color_counter += 1
    # Print summary table for MPC
    mpc_results = []
    for i, mean_val in enumerate(mean_mpc_final_rewards):
        mpc_results.append([horizons[i], mean_val, ci_mpc_final_rewards[i]])
    print(f"MPC {variable} Results:")
    print(tabulate(mpc_results, headers=['Horizon', 'Mean Final Reward', 'Confidence Interval'], floatfmt=".3f"))

    # --- Plot SMPC results ---
    mean_smpc_final_rewards = []
    std_smpc_final_rewards = []
    ci_smpc_final_rewards = []
    for h in horizons:
        if h in data['smpc']:
            grouped_runs = data['smpc'][h].groupby("run")
            cumulative_rewards = grouped_runs[variable].sum()

            mean_smpc_final_rewards.append(cumulative_rewards.mean())
            std_smpc_final_rewards.append(cumulative_rewards.std())
            ci = compute_confidence_interval(cumulative_rewards)
            ci_smpc_final_rewards.append(ci)
    n2plot = len(mean_smpc_final_rewards)
    # Print summary table for SMPC
    smpc_results = []
    for i, mean_val in enumerate(mean_smpc_final_rewards):
        smpc_results.append([horizons[i], mean_val, ci_smpc_final_rewards[i]])
    print(f"SMPC {variable} Results:")
    print(tabulate(smpc_results, headers=['Horizon', 'Mean Final Reward', 'Confidence Interval'], floatfmt=".3f"))

    if mean_smpc_final_rewards:
        # Plot mean cumulative metric for SMPC
        ax.plot(horizon_nums[:n2plot], mean_smpc_final_rewards[:n2plot], 'o-', label=f'SMPC', color="C0", alpha=0.8)
        # Plot confidence interval as shaded region for stochastic mode
        if mode == "stochastic":
            ax.fill_between(
                horizon_nums[:n2plot], 
                np.array(mean_smpc_final_rewards[:n2plot]) - np.array(ci_smpc_final_rewards[:n2plot]),
                np.array(mean_smpc_final_rewards[:n2plot]) + np.array(ci_smpc_final_rewards[:n2plot]),
                color="C0", alpha=0.3
            )
        color_counter += 1

    # --- Plot RL-SMPC (zero-order) results for each model ---
    for idx, model in enumerate(model_names):
        mean_rlmpc_final_rewards = []
        std_rlmpc_final_rewards = []
        ci_rlsmpc_final_rewards = []

        for h in horizons:
            if h in data['rl-zero-terminal-smpc']:
                # Each model has its own DataFrame for each horizon
                grouped_runs = data['rl-zero-terminal-smpc'][h][model].groupby("run")
                cumulative_rewards = grouped_runs[variable].sum()

                mean_rlmpc_final_rewards.append(cumulative_rewards.mean())
                std_rlmpc_final_rewards.append(cumulative_rewards.std())
                ci = compute_confidence_interval(cumulative_rewards)
                ci_rlsmpc_final_rewards.append(ci)

        n2plot = len(mean_rlmpc_final_rewards)
        # Print summary table for RL-SMPC
        table_data = []
        for i, mean_val in enumerate(mean_rlmpc_final_rewards):
            table_data.append([horizons[i], mean_val, ci_rlsmpc_final_rewards[i]])
        print(f"RL-SMPC {variable} Results:")
        print(tabulate(table_data, headers=["Horizon", "Mean Final Reward", "Confidence Interval"], floatfmt=".3f"))
        if mean_rlmpc_final_rewards:
            # Plot mean cumulative reward for RL-SMPC
            ax.plot(horizon_nums[:n2plot], mean_rlmpc_final_rewards[:n2plot], 'o-', label=r'RL-SMPC', color="C3", alpha=0.8)
            # Plot confidence interval as shaded region for stochastic mode
            if mode == "stochastic":
                ax.fill_between(
                    horizon_nums[:n2plot], 
                    np.array(mean_rlmpc_final_rewards[:n2plot]) - np.array(ci_rlsmpc_final_rewards[:n2plot]),
                    np.array(mean_rlmpc_final_rewards[:n2plot]) + np.array(ci_rlsmpc_final_rewards[:n2plot]),
                    color="C3", alpha=0.3
                )
            color_counter += 1

    # --- Plot RL results for each model as a horizontal line ---
    for idx, model in enumerate(model_names):
        # Only plot if RL results are available for this model
        if model in data['rl']:
            sum_rewards = data['rl'][model].groupby("run")[variable].sum()
            rl_final_reward = sum_rewards.mean()
            # Plot a horizontal line for RL performance across all horizons
            ax.hlines(rl_final_reward, min(horizon_nums), max(horizon_nums),
                        label=f'RL', color="C7", linestyle='--', alpha=0.8)
            print(f"RL {model}; {variable} Result: {rl_final_reward:.3f}")
        color_counter += 1
        # Set y-axis ticks and formatting for clarity
        ax.yaxis.set_major_locator(plt.LinearLocator(3))
        ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))

        # rl_table = []
        # for model in model_names:
        #     if model in data['rl']:
        #         final_rewards = data['rl'][model].groupby("run")[variable].sum()
        #         rl_final_reward = final_rewards.mean()
        #         rl_table.append([model, rl_final_reward])
        # if rl_table:
        #     print("RL Final Rewards:")
        #     print(tabulate(rl_table, headers=["Model", "Final Reward"], floatfmt=".3f"))

    # Set axis labels and legend
    ax.set_xlabel('Prediction Horizon (H)')
    if variable == 'rewards':
        ax.set_ylabel(f'Cumulative {variable[:-1]}')
        ax.legend()
    elif variable == 'econ_rewards':
        ax.set_ylabel(f'Cumulative EPI (EU/m$^2$)')
    elif variable == 'penalties':
        ax.set_ylabel(f'Cumulative penalty')
    print("----------------------------------------")
    fig.tight_layout()

    # --- Save plot to disk ---
    dir_path = f'figures/{project}/{mode}/{figure_name}/'
    os.makedirs(dir_path, exist_ok=True)
    # Use a more descriptive filename for economic rewards
    if variable == "econ_rewards":
        variable = "EPI"
    uncertainty_suffix = f'-{uncertainty_value}' if uncertainty_value else ''
    fig.savefig(f'{dir_path}{variable}{uncertainty_suffix}.svg', format='svg',
                bbox_inches='tight', dpi=300)
    fig.savefig(f'{dir_path}{variable}{uncertainty_suffix}.png', format='png',
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
