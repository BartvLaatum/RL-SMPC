import os
import re
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import plot_config
import numpy as np
from matplotlib.colors import SymLogNorm, LogNorm

# Configure matplotlib plot styling
plt.rcParams["axes.spines.top"] = True
plt.rcParams["axes.spines.right"] = True
plt.rcParams["axes.linewidth"] = 1  # Axis border thickness

# Figure dimensions in inches (converted from mm)
WIDTH = 61 * 0.03937
HEIGHT = WIDTH * 0.75

def extract_params(filename):
    """
    Extract prediction horizon (h) and uncertainty (delta) parameters from filename.

    Expected filename pattern: "mpc-{h}H-{delta}.csv" where:
    - h: prediction horizon (integer)
    - delta: uncertainty level (float)

    Args:
        filename (str): Path to the CSV file

    Returns:
        tuple: (h, delta) if pattern matches, (None, None) otherwise
    """
    basename = os.path.basename(filename)
    pattern = r".+-(\d+)H-([\d.]+)\.csv"
    match = re.match(pattern, basename)
    if match:
        h = int(match.group(1))
        delta = float(match.group(2))
        return h, delta
    return None, None

def get_mean_reward(csv_file, column='rewards'):
    """
    Read CSV file and compute the mean of the cumulative reward across multiple runs.
    Args:
        csv_file (str): Path to the CSV file containing experimental results
        column (str): Column name for rewards (default: 'rewards')

    Returns:
        tuple: (mean_reward, mean_econ_rewards, mean_penalties) across all runs
    """
    df = pd.read_csv(csv_file)
    df_grouped = df.groupby('run')
    return df_grouped['rewards'].sum().mean(), df_grouped['econ_rewards'].sum().mean(), df_grouped['penalties'].sum().mean()

def compute_diff_matrices(mpc_results, rlsmpc_results, horizons, uncertainties):
    """
    Compute difference matrices for rewards between two algorithms.

    Calculates percentage differences between RL-SMPC and another algorithm (MPC, SMPC, or RL)
    across different prediction horizons and uncertainty levels. The difference is computed as:
    ((rlsmpc_mean_reward - other_mean_reward) / abs(other_mean_reward)) * 100

    Args:
        mpc_results (dict): Dictionary mapping (h, delta) pairs to mean reward values
        rlsmpc_results (dict): Dictionary mapping (h, delta) pairs to mean reward values  
        horizons (list): Sorted list of prediction horizons
        uncertainties (list): Sorted list of uncertainty levels

    Returns:
        numpy.ndarray: 2D array with percentage differences, shape (len(uncertainties), len(horizons))
    """
    # Determine common (h, delta) pairs available in both datasets
    common_keys = set(mpc_results.keys()) & set(rlsmpc_results.keys())

    # Initialize matrix with NaNs for missing data points
    diff_matrix = np.full((len(uncertainties), len(horizons)), np.nan)

    # Fill matrix with percentage differences
    for i, delta in enumerate(uncertainties):
        for j, h in enumerate(horizons):
            if (h, delta) in common_keys:
                mpc_reward = mpc_results[(h, delta)]
                rlsmpc_reward = rlsmpc_results[(h, delta)]
                diff_matrix[i, j] = ((rlsmpc_reward - mpc_reward) / abs(mpc_reward)) * 100
    return diff_matrix


def main():
    """
    Main function that orchestrates the uncertainty heatmap visualization process.

    This function:
    1. Defines data directories for different MPC algorithms
    2. Processes CSV files to extract performance metrics
    3. Computes difference matrices between algorithm pairs
    4. Creates and displays heatmap visualizations
    """
    # Define directories for the different MPC algorithm results
    mpc_dir = 'data/uncertainty-comparison/stochastic/mpc'
    smpc_dir = 'data/uncertainty-comparison/stochastic/smpc'
    rlsmpc_dir = 'data/uncertainty-comparison/stochastic/rlsmpc'
    rl_dir = "data/uncertainty-comparison/stochastic/rl"

    # Dictionaries to store mean rewards for each (h, delta) combination
    mpc_rewards = {}
    rlsmpc_rewards = {}
    rl_rewards = {}
    smpc_rewards = {}

    # Dictionaries to store mean EPI (Economic Performance Index) for each (h, delta)
    mpc_EPI = {}
    rlsmpc_EPI = {}

    # Dictionaries to store mean penalties for each (h, delta)
    mpc_penalty = {}
    rlsmpc_penalty = {}
    

    def process_csv_files(directory, pattern, result_dict, print_warning=False):
        """
        Process CSV files matching the given pattern in a directory.

        Extracts (h, delta) parameters from filenames and computes mean rewards using get_mean_reward().
        The mean reward is stored in result_dict keyed by (h, delta) tuple.

        Args:
            directory (str): The directory path to search for CSV files
            pattern (str): The file pattern to search for (e.g., "warm-start-*-*.csv")
            result_dict (dict): Dictionary to store results keyed by (h, delta)
            print_warning (bool): If True, print warnings when parameters cannot be extracted
        """
        full_pattern = os.path.join(directory, pattern)
        for filepath in glob.glob(full_pattern):
            h, delta = extract_params(filepath)
            if h is None:
                if print_warning:
                    print(f"Skipping file {filepath}: could not extract parameters.")
                continue
            try:
                mean_reward, mean_epi, mean_penalty = get_mean_reward(filepath)
            except Exception as e:
                print(f"Error reading {filepath}: {e}")
                continue

            result_dict[(h, delta)] = mean_reward
            # Uncomment the following lines to process additional metrics:
            # result_dict_epi[(h, delta)] = mean_epi
            # result_dict_penalty[(h, delta)] = mean_penalty

    # Process files for each algorithm type
    process_csv_files(mpc_dir, "warm-start-*-*.csv", mpc_rewards)
    process_csv_files(smpc_dir, "no-tightening-*-*.csv", smpc_rewards)
    process_csv_files(rlsmpc_dir, "*-no-tightening*-*.csv", rlsmpc_rewards, print_warning=True)

    # List of RL model names for processing individual RL results
    rl_models = [
        "ruby-serenity-54", "pretty-terrain-55", "dutiful-fire-56", "brisk-resonance-24",
        "solar-haze-57", "hardy-violet-58", "glorious-mountain-59","swift-morning-5"
    ]

    # Get common (h, delta) pairs available in both datasets
    common_keys = set(smpc_rewards.keys()) & set(rlsmpc_rewards.keys())
    if not common_keys:
        print("No matching (prediction horizon, uncertainty) pairs found between mpc and rl-smpc data.")
        print(mpc_rewards.keys())
        return

    # Create sorted lists of horizons and uncertainties for consistent plotting
    horizons = sorted({h for h, _ in common_keys})
    uncertainties = sorted({delta for _, delta in common_keys})
    print(uncertainties)

    # Process RL results - each model corresponds to a specific uncertainty level
    for i, model in enumerate(rl_models):
        delta = uncertainties[i]
        filepath = os.path.join(rl_dir, f"{model}.csv")
        try:
            mean_reward, mean_epi, mean_penalty = get_mean_reward(filepath)
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            continue

        # Assign the same reward to all horizons for this uncertainty level
        for h in horizons:
            rl_rewards[(h, delta)] = mean_reward

    # Create heatmap comparing RL-SMPC vs MPC
    diff_matrix_rewards = compute_diff_matrices(mpc_rewards, rlsmpc_rewards, horizons, uncertainties)
    m = max(abs(diff_matrix_rewards.max()), abs(diff_matrix_rewards.min()))

    fig, ax = create_heatmap_figure(horizons, uncertainties, title="RL-SMPC vs MPC")
    fig, ax = plot_heatmap(diff_matrix_rewards, ax, fig, 'reward', m)
    # fig.savefig("heatmap-rlsmpc-mpc.svg", dpi=300, bbox_inches='tight', format='svg')
    plt.show()

    # Create heatmap comparing RL-SMPC vs RL
    diff_matrix_rewards = compute_diff_matrices(rl_rewards, rlsmpc_rewards, horizons, uncertainties)
    m = max(abs(diff_matrix_rewards.max()), abs(diff_matrix_rewards.min()), m)
    print(diff_matrix_rewards)

    fig, ax = create_heatmap_figure(horizons, uncertainties, title="RL-SMPC vs RL")
    fig, ax = plot_heatmap(diff_matrix_rewards, ax, fig, 'reward', m)
    # fig.savefig("heatmap-rlsmpc-rl.svg", dpi=300, bbox_inches='tight', format='svg')
    plt.show()

    # Create heatmap comparing RL-SMPC vs SMPC
    diff_matrix_rewards = compute_diff_matrices(smpc_rewards, rlsmpc_rewards, horizons, uncertainties)
    m = max(abs(diff_matrix_rewards.max()), abs(diff_matrix_rewards.min()), m)

    fig, ax = create_heatmap_figure(horizons, uncertainties, title="RL-SMPC vs SMPC")
    fig, ax = plot_heatmap(diff_matrix_rewards, ax, fig, 'reward', m)
    # fig.savefig("heatmap-rlsmpc-smpc.svg", dpi=300, bbox_inches='tight', format='svg')
    plt.show()

def plot_heatmap(data, ax, fig, variable, m):
    """
    Create and display the heatmap.

    Args:
        data (numpy.ndarray): 2D array of data to visualize
        ax (matplotlib.axes.Axes): Matplotlib axes object
        fig (matplotlib.figure.Figure): Matplotlib figure object
        variable (str): Name of the variable being plotted (for labels)
        m (float): Maximum absolute value for color scale normalization

    Returns:
        tuple: (fig, ax) - Updated figure and axes objects
    """
    print(m)
    # Use coolwarm colormap for diverging data
    cmap="coolwarm"
    im = ax.imshow(data, cmap=cmap, origin='upper')
    
    # Use symmetric logarithmic normalization for better visualization of small differences
    norm = SymLogNorm(linthresh=1, vmin=-m, vmax=m, base=10)
    im.set_norm(norm)
    ax.invert_yaxis()

    # Colorbar configuration
    cbar = fig.colorbar(im, ax=ax, orientation='horizontal')
    cbar.ax.invert_xaxis()
    cbar.ax.xaxis.set_label_position('top')
    cbar.ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
    cbar.set_label(f'$\Delta$% Cumulative {variable}')
    ax.square()

    plt.tight_layout()
    return fig, ax

def create_heatmap_figure(horizons, uncertainties, title):
    """
    Create a matplotlib figure with proper formatting for heatmap visualization.

    Args:
        horizons (list): List of prediction horizons for x-axis
        uncertainties (list): List of uncertainty levels for y-axis
        title (str): Title for the plot

    Returns:
        tuple: (fig, ax) - Figure and axes objects
    """
    fig = plt.figure(figsize=(WIDTH, HEIGHT), dpi=300)
    ax = plt.gca()
    ax.set_title(title)

    # Set x-axis ticks and labels for prediction horizons
    ax.set_xticks(np.arange(start=1, stop=9, step=2))
    ax.set_xticklabels([2, 4, 6, 8])

    # Set y-axis ticks and labels for specific uncertainty values
    show_values = [0.05, 0.1, 0.15, 0.2]
    ytick_positions = [uncertainties.index(val) for val in show_values if val in uncertainties]
    ax.set_yticks(ytick_positions)
    ax.set_yticklabels(show_values)

    # Set axis labels
    ax.set_xlabel("Prediction Horizon (H)")
    ax.set_ylabel("Uncertainty $(\delta)$")
    return fig, ax

if __name__ == "__main__":
    main()
