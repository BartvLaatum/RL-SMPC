import os
import argparse

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cmcrameri.cm as cmc
from performance_plots import create_plot

import plot_config

from tabulate import tabulate
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib import colors as mcolors

def rl_seeds_plot(
        figure_name,
        project, 
        data,
        horizons, 
        model_names, 
        mode, 
        variable, 
        uncertainty_value=None,
        color=None,
        labels=None
    ):
    """
    Creates a comparison plot between different RL-SMPC approaches across prediction horizons.
    The function computes mean performance and confidence intervals across multiple runs
    for each horizon and model configuration, then creates publication-ready figures
    with appropriate styling and saves them in both PNG and SVG formats.

    
    This function generates a line plot comparing the performance of RL-SMPC configurations
    using different configurations of RL policies. Specifically, one policy using future weather disturbances
    and one without. The plot shows:
    - RL-SMPC performance as lines with markers for different prediction horizons
    - RL performance as horizontal dashed lines (horizon-independent)
    - Confidence intervals as shaded regions for stochastic environments
    - Custom legend distinguishing between RL-SMPC and RL approaches

    Args:
        figure_name (str): Name for the output figure files.
        project (str): Name of the project directory where figures will be saved.
        data (dict): Dictionary containing the results data with structure:
            {
                'rl-zero-terminal-smpc': {horizon: {model_name: DataFrame}},
                'rl': {model_name: DataFrame}
            }
        horizons (list): List of prediction horizon values to plot.
        model_names (list): List of model names used in the RL and RL-SMPC approaches.
        mode (str): Either 'stochastic' or 'deterministic' to indicate environment type.
        variable (str): The metric to plot (e.g., 'rewards', 'econ_rewards', 'penalties').
        Ns (list, optional): Not used in this function, included for compatibility.
        uncertainty_value (float, optional): Scale of uncertainty for stochastic environments.
        rlcolors (list, optional): List of colors for different RL-SMPC configurations.
        labels (list, optional): List of labels for different RL-SMPC configurations.

    Returns: None

    Displays the plot and saves it.
    """
    # Set up figure dimensions and create plot
    WIDTH = 130/3 * 0.0393700787
    HEIGHT = WIDTH * 1.0
    fig, ax = plt.subplots(figsize=(WIDTH, HEIGHT), dpi=300)
    
    # Convert horizon labels to integers for plotting
    horizon_nums = [int(h[0]) for h in horizons]
    returns_mean_models = []

    # --- Compute average RL-SMPC results for each model configuration ---
    for idx, model in enumerate(model_names):
        mean_rlmpc_final_rewards = []
        ci_rlsmpc_final_rewards = []
        
        # Compute statistics for each prediction horizon
        for h in horizons:
            if h in data['rl-zero-terminal-smpc']:
                # Group by run and sum the specified variable to get cumulative reward per run
                grouped_runs = data['rl-zero-terminal-smpc'][h][model].groupby("run")
                cumulative_rewards = grouped_runs[variable].sum()

                mean_rlmpc_final_rewards.append(cumulative_rewards.mean())
        returns_mean_models.append(mean_rlmpc_final_rewards)

    returns_mean_models = np.array(returns_mean_models)
    mean_over_seeds = returns_mean_models.mean(axis=0)
    std_over_seeds = returns_mean_models.std(axis=0)
    ci_over_seeds = std_over_seeds / np.sqrt(returns_mean_models.shape[0]) * 2.776  # 95% CI

    # Plot mean performance for this RL-SMPC configuration
    ax.plot(horizon_nums[:], mean_over_seeds[:], 'o-', 
            label=labels[idx], color=color, alpha=0.8)
        
    ax.fill_between(
        horizon_nums[:], 
        np.array(mean_over_seeds[:]) - np.array(ci_over_seeds[:]),
        np.array(mean_over_seeds[:]) + np.array(ci_over_seeds[:]),
        color=color, alpha=0.3
    )
        
    # Print summary table for this RL-SMPC configuration
    table_data = []
    for i, mean_val in enumerate(mean_over_seeds):
        table_data.append([horizons[i], mean_val, ci_over_seeds[i]])
    print(f"RL-SMPC Performance over seeds; {variable} Results:")
    print(tabulate(table_data, headers=["Horizon", "Mean Final Reward", "Confidence Interval"], floatfmt=".3f"))

    # --- Plot RL results as horizontal lines for each model ---
    rl_returns_mean = []
    for idx, model in enumerate(model_names):
        # Only plot if RL results are available for this model
        if model in data['rl']:
            sum_rewards = data['rl'][model].groupby("run")[variable].sum()
            rl_returns_mean.append(sum_rewards.mean())


    rl_returns_mean = np.array(rl_returns_mean)
    rl_mean_over_seeds = rl_returns_mean.mean()
    std_rl_final_reward = rl_returns_mean.std()
    rl_ci_over_seeds = std_rl_final_reward / np.sqrt(rl_returns_mean.shape[0]) * 2.776  # 95% CI

    table_data = []
    print(f"RL Performance over seeds; {variable} Results:")
    print(tabulate([[rl_mean_over_seeds, rl_ci_over_seeds]], headers=["Mean Final Reward", "Confidence Interval"], floatfmt=".3f"))

    ax.hlines(rl_mean_over_seeds, min(horizon_nums[:]), max(horizon_nums[:]),
                    color=color,  linestyle='--', alpha=0.8)

    ax.fill_between(
        [min(horizon_nums[:]), max(horizon_nums[:])], 
        rl_mean_over_seeds - rl_ci_over_seeds,
        rl_mean_over_seeds + rl_ci_over_seeds,
        color=color, alpha=0.3
    )

    # Set y-axis formatting for clarity
    ax.yaxis.set_major_locator(plt.LinearLocator(3))
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
    ax.set_xticks(horizon_nums[:])
    # Set x-axis label
    ax.set_xlabel('Prediction horizon ($H$)')

    # --- Create custom legend with both color patches and line styles ---
    # Define legend handles for different configurations
    # color_handles = [
    #     Patch(facecolor=color, edgecolor='None', alpha=0.8),
    #     Patch(facecolor=rlcolors[1], edgecolor='None', label='No future weather', alpha=0.8)
    # ]

    # Define legend handles for line styles
    line_handles = [
        Line2D([0], [0], color=color, marker='o', linestyle='-', label='RL-SMPC'),
        Line2D([0], [0], color=color, linestyle='--', label='RL')
    ]

    # Combine handles for unified legend
    handles = line_handles

    # Set y-axis label and create legend based on variable type
    if variable == 'rewards':
        ax.set_ylabel(f'Cumulative {variable[:-1]}')
        ax.legend(handles=handles, loc='center', ncol=2)
    elif variable == 'econ_rewards':
        ax.set_ylabel(f'Cumulative EPI (EUR/m$^2$)')
    elif variable == 'penalties':
        ax.set_ylabel(f'Cumulative penalty')

    # Add grid and finalize layout
    # ax.grid()
    fig.tight_layout()

    # --- Save plot ---
    dir_path = f'figures/{project}/{mode}/{figure_name}/'
    os.makedirs(dir_path, exist_ok=True)
    
    # Use descriptive filename for economic rewards
    if variable == "econ_rewards":
        variable = "EPI"
    uncertainty_suffix = f'-{uncertainty_value}' if uncertainty_value else ''
    
    # Save in both PNG and SVG formats
    fig.savefig(f'{dir_path}{variable}{uncertainty_suffix}.svg', format='svg',
                bbox_inches='tight', dpi=300)
    fig.savefig(f'{dir_path}{variable}{uncertainty_suffix}.png', format='png',
                bbox_inches='tight', dpi=300)
    print(f'Saved plot to {dir_path}{variable}{uncertainty_suffix}.svg')
    plt.show()

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
    Load and organize simulation data from MPC, SMPC, RL, and RL-SMPC experiments.
    
    This function reads CSV files containing results from different control strategies
    and organizes them into a nested dictionary structure. It supports loading data
    for various RL-SMPC configurations (zero-order, with/without terminal
    constraints) and traditional MPC/SMPC approaches.
    
    The function constructs file paths based on the provided parameters and loads
    data only if the corresponding files exist, making it robust to missing data.
    
    Args:
        model_names (list): List of RL model names to load data for.
        mode (str): Operating mode of the simulation (e.g., 'stochastic', 'deterministic').
        project (str): Project name/folder containing the data.
        mpc (bool, optional): Whether to load MPC data (default: True).
        smpc (bool, optional): Whether to load SMPC data (default: True).
        zero_order (bool, optional): Whether to load zero-order RL-SMPC data (default: True).
        terminal (bool, optional): Whether to load terminal constraint versions (default: False).
        first_order (bool, optional): Whether to load first-order RL-SMPC data (default: False).
        Ns (list, optional): Not used, included for compatibility.
        uncertainty_value (float, optional): Scale factor for uncertainty in predictions.
            If provided, loads data with specified uncertainty scale suffix.
    
    Returns:
        tuple: A tuple containing:
            - data (dict): Nested dictionary containing loaded dataframes organized by:
                - 'mpc': Dict of dataframes indexed by horizon
                - 'smpc': Dict of dataframes indexed by horizon
                - 'rl': Dict of dataframes indexed by model name
                - 'rl-zero-smpc': Dict of dicts indexed by horizon then model name
                - 'rl-zero-terminal-smpc': Dict of dicts indexed by horizon then model name
                - 'rl-first-smpc': Dict of dicts indexed by horizon then model name
                - 'rl-first-terminal-smpc': Dict of dicts indexed by horizon then model name
            - horizons (list): List of horizon values used in the simulations
    
    Notes:
        Expected file structure:
        data/
            {project}/
                {mode}/
                    rl/
                        {model}.csv
                    mpc/
                        warm-start-{horizon}-{uncertainty}.csv
                    smpc/
                        no-tightening-{horizon}-{uncertainty}.csv
                    rlsmpc/
                        {model}-zero-order-{horizon}-{uncertainty}.csv
                        {model}-no-tightening-{horizon}-{uncertainty}.csv
                        {model}-first-order-{horizon}-{uncertainty}.csv
                        {model}-first-order-terminal-{horizon}-{uncertainty}.csv
    """
    # Define standard prediction horizons
    horizons = ['1H', '2H', '3H', '4H']

    # Initialize data structure for all possible data types
    data = {
        'rl': {},
        'rl-zero-terminal-smpc': {},
        }

    # Create uncertainty suffix for file naming
    uncertainty_suffix = f'-{uncertainty_value}' if uncertainty_value else ''

    # --- Load RL and RL-SMPC data for each model ---
    for model in model_names:
        # Load pure RL data (horizon-independent)
        rl_path = f'data/{project}/{mode}/rl/{model}.csv'
        if os.path.exists(rl_path):
            data['rl'][model] = pd.read_csv(rl_path)

        # Load RL-SMPC data for each horizon
        for h in horizons:
            # Define file paths for different RL-SMPC configurations
            rlsmpc_path = f'data/{project}/{mode}/rlsmpc/{model}-no-tightening-{h}{uncertainty_suffix}-{args.Ns}Ns.csv'


            if os.path.exists(rlsmpc_path):
                if h not in data['rl-zero-terminal-smpc']:
                    data['rl-zero-terminal-smpc'][h] = {}
                data['rl-zero-terminal-smpc'][h][model] = pd.read_csv(rlsmpc_path)

    return data, horizons

def main(args):
    """
    Main entry point for RL-SMPC performance visualization script.
    
    This function orchestrates the complete workflow for creating performance comparison plots:
    1. Loads simulation data for different control approaches (MPC, SMPC, RL, RL-SMPC)
    2. Creates comparison plots for multiple performance metrics
    3. Supports two visualization modes:
       - Standard comparison: All control approaches across prediction horizons
       - RL comparison: Different RL-SMPC configurations with custom styling
    
    The function generates plots for three key metrics:
    - rewards: Cumulative control performance
    - econ_rewards: Economic performance (EPI - Economic Performance Index)
    - penalties: Constraint violation penalties
    
    Args:
        args (argparse.Namespace): Parsed command-line arguments containing:
            - project: Project name for data loading
            - model_names: List of RL model names to analyze
            - mode: Environment mode ('deterministic' or 'stochastic')
            - uncertainty_value: Uncertainty scale for stochastic environments
            - compare_rl: Whether to use RL comparison mode
            - Various boolean flags for data loading options
    
    Returns:
        None: Displays plots and saves them to disk.
    """
    # Load simulation data for all specified models and configurations
    data, horizons = load_data(
        args.model_names, 
        args.mode, 
        args.project, 
        mpc=args.mpc,
        smpc=args.smpc,
        zero_order=args.zero_order,
        first_order=args.first_order,
        terminal=args.terminal, 
        Ns=args.Ns, 
        uncertainty_value=args.uncertainty_value
    )
    
    # Define the performance metrics to visualize
    variables2plot = ["rewards", "econ_rewards", "penalties"]
    
    # Create figure name with uncertainty value for file naming
    figure_name = args.figure_name + f"-{args.uncertainty_value}"
    
    # Choose visualization mode based on command-line argument
    if args.compare_rl:
        # RL comparison mode: Compare different RL-SMPC configurations
        # Define colors and labels for different RL-SMPC approaches
        # Generate lighter gradients of C3
        base_color = mcolors.to_rgb("C3")
        rl_colors = [mcolors.to_hex(base_color)]
        # Create four lighter variants
        for factor in [0.8, 0.6, 0.4, 0.2]:
            lighter = tuple(1 - factor * (1 - c) for c in base_color)
            rl_colors.append(mcolors.to_hex(lighter))
        rl_labels = ["Seed-1", "Seed-2", "Seed-3", "Seed-4", "Seed-5"]
        
        # Generate plots for each performance metric
        for variable in variables2plot:
            rl_seeds_plot(
                figure_name, 
                args.project, 
                data, 
                horizons, 
                args.model_names, 
                args.mode, 
                variable, 
                args.uncertainty_value,
                base_color,
                rl_labels
            )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=str, default='SMPC',
                        help='Name of the project')
    parser.add_argument('--model_names', nargs='+', type=str, default=[],
                        help='List of model names to plot')
    parser.add_argument('--smpc', action=argparse.BooleanOptionalAction,
                        help='Whether to plot SMPC')
    parser.add_argument('--mpc', action=argparse.BooleanOptionalAction,
                        help='Whether to plot MPC')
    parser.add_argument('--zero-order', action=argparse.BooleanOptionalAction,
                        help='Whether to RL-SMPC with zero-order approximation')
    parser.add_argument('--first-order', action=argparse.BooleanOptionalAction,
                        help='Whether to RL-SMPC with fisst-order approximation')
    parser.add_argument('--terminal', action=argparse.BooleanOptionalAction,
                        help='Whether to visualise RL-SMPC with terminal state/cost implemented')
    parser.add_argument('--Ns', type=int, default=10,
                        help='Integer of SMPC with Ns scenario samples to plot')
    parser.add_argument('--mode', type=str, 
                        choices=['deterministic', 'stochastic'], required=True)
    parser.add_argument('--uncertainty_value', type=float,
                        help='Uncertainty scale value for stochastic mode')
    parser.add_argument('--compare_rl', action=argparse.BooleanOptionalAction,
                        help='Plot to compare RL models')
    parser.add_argument('--figure_name', type=str, required=True)
    args = parser.parse_args()
    main(args)
