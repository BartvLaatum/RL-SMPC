import os
import re
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import plot_config
import numpy as np

plt.rcParams["axes.spines.top"] = True
plt.rcParams["axes.spines.right"] = True
plt.rcParams["axes.linewidth"] = 1  # Axis border thickness


WIDTH = 61 * 0.03937
HEIGHT = WIDTH * 0.75

def extract_params_mpc(filename):
    """
    Extract h and delta from filename with pattern "mpc-{h}H-{delta}.csv".
    """
    basename = os.path.basename(filename)
    pattern = r"mpc-(\d+)H-([\d.]+)\.csv"
    match = re.match(pattern, basename)
    if match:
        h = int(match.group(1))
        delta = float(match.group(2))
        return h, delta
    return None, None

def extract_params_rlsmpc(filename):
    """
    Extract h and delta from filename with pattern "{model}-zero-order-terminal-{h}H-{delta}.csv".
    """
    basename = os.path.basename(filename)
    pattern = r".+-zero-order-terminal-box-constraints-(\d+)H-([\d.]+)\.csv"
    match = re.match(pattern, basename)
    if match:
        h = int(match.group(1))
        delta = float(match.group(2))
        return h, delta
    return None, None

def get_mean_reward(csv_file, column='rewards'):
    """
    Read CSV file and compute the mean of the cumulative reward.
    """
    df = pd.read_csv(csv_file)
    df_grouped = df.groupby('run')
    return df_grouped['rewards'].sum().mean(), df_grouped['econ_rewards'].sum().mean(), df_grouped['penalties'].sum().mean()

def compute_diff_matrices(mpc_results, rlsmpc_results, horizons, uncertainties):
    """
    Compute difference matrices for rewards, EPI, and penalties.

    Each of the input dictionaries maps (h, delta) pairs to a tuple:
        (mean_reward, mean_epi, mean_penalty)

    The differences are computed as follows:
        • diff_matrix_rewards: (rlsmpc mean reward) - (mpc mean reward)
        • diff_matrix_epi: (rlsmpc mean EPI) - (mpc mean EPI)
        • diff_matrix_penalty: (mpc mean penalty) - (rlsmpc mean penalty)

    Parameters:
        mpc_results (dict): Dictionary with mpc results.
        rlsmpc_results (dict): Dictionary with rl-smpc results.
        horizons (list): Sorted list of prediction horizons.
        uncertainties (list): Sorted list of uncertainties.

    Returns:
        tuple: Three NumPy arrays (diff_matrix_rewards, diff_matrix_epi, diff_matrix_penalty)
    """

    # Determine common (h, delta) pairs
    common_keys = set(mpc_results.keys()) & set(rlsmpc_results.keys())

    # Initialize matrices with NaNs
    diff_matrix = np.full((len(uncertainties), len(horizons)), np.nan)

    for i, delta in enumerate(uncertainties):
        for j, h in enumerate(horizons):
            if (h, delta) in common_keys:
                mpc_reward = mpc_results[(h, delta)]
                rlsmpc_reward = rlsmpc_results[(h, delta)]
                diff_matrix[i, j] = ((rlsmpc_reward - mpc_reward) / abs(mpc_reward)) * 100
    return diff_matrix


def main():
    # Define directories for the two algorithms
    mpc_dir = 'data/uncertainty-comparison/stochastic/mpc'
    rlsmpc_dir = 'data/uncertainty-comparison/stochastic/rlsmpc'
    rl_dir = "data/uncertainty-comparison/stochastic/rl"


    # Dictionaries to hold the mean reward for each (h, delta)
    mpc_rewards = {}
    rlsmpc_rewards = {}
    rl_rewards = {}

    # Dictionaries to hold the mean EPI for each (h, delta)
    mpc_EPI = {}
    rlsmpc_EPI = {}

    # Dictionaries to hold the mean EPI for each (h, delta)
    mpc_penalty = {}
    rlsmpc_penalty = {}

    # Process mpc files
    mpc_pattern = os.path.join(mpc_dir, "mpc-*-*.csv")
    for filepath in glob.glob(mpc_pattern):
        h, delta = extract_params_mpc(filepath)
        if h is None:
            continue
        try:
            mean_reward, mean_epi, mean_penalty = get_mean_reward(filepath)
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            continue

        mpc_rewards[(h, delta)] = mean_reward
        mpc_EPI[(h, delta)] = mean_epi
        mpc_penalty[(h, delta)] = mean_penalty

    # Process rl-smpc files
    rlsmpc_pattern = os.path.join(rlsmpc_dir, "*-zero-order-terminal-box-constraints*-*.csv")
    for filepath in glob.glob(rlsmpc_pattern):
        h, delta = extract_params_rlsmpc(filepath)
        if h is None:
            continue
        try:
            mean_reward, mean_epi, mean_penalty = get_mean_reward(filepath)
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            continue
        rlsmpc_rewards[(h, delta)] = mean_reward
        rlsmpc_EPI[(h, delta)] = mean_epi
        rlsmpc_penalty[(h, delta)] = mean_penalty
    
    rl_models = [
        "serene-vortex-31","clear-water-33","stoic-fire-34","fresh-durian-18",
        "cosmic-sun-35","dry-bee-36","soft-river-37","revived-snow-38"
    ]

    # Get common (h, delta) pairs available in both datasets
    common_keys = set(mpc_rewards.keys()) & set(rlsmpc_rewards.keys())
    if not common_keys:
        print("No matching (prediction horizon, uncertainty) pairs found between mpc and rl-smpc data.")
        return

    # Create sorted lists of horizons and uncertainties
    horizons = sorted({h for h, _ in common_keys})
    uncertainties = sorted({delta for _, delta in common_keys})

    for i, model in enumerate(rl_models):
        delta = uncertainties[i]
        filepath = os.path.join(rl_dir, f"{model}.csv")
        try:
            mean_reward, mean_epi, mean_penalty = get_mean_reward(filepath)
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            continue
        
        for h in horizons:
            rl_rewards[(h, delta)] = mean_reward


    diff_matrix_rewards = compute_diff_matrices(mpc_rewards, rlsmpc_rewards, horizons, uncertainties)
    m = max(abs(diff_matrix_rewards.max()), abs(diff_matrix_rewards.min()))

    # Plot heatmap REWARDS
    fig, ax = create_heatmap_figure(horizons, uncertainties, title="RL-SMPC vs MPC")
    fig, ax = plot_heatmap(diff_matrix_rewards, ax, fig, 'reward', m)
    fig.savefig("heatmap-rlsmpc-mpc.svg", dpi=300, bbox_inches='tight', format='svg')

    plt.show()

    diff_matrix_rewards = compute_diff_matrices(rl_rewards, rlsmpc_rewards, horizons, uncertainties)
    m = max(abs(diff_matrix_rewards.max()), abs(diff_matrix_rewards.min()), m)

    # Plot heatmap REWARDS
    fig, ax = create_heatmap_figure(horizons, uncertainties, title="RL-SMPC vs RL")
    fig, ax = plot_heatmap(diff_matrix_rewards, ax, fig, 'reward', m)
    fig.savefig("heatmap-rlsmpc-rl.svg", dpi=300, bbox_inches='tight', format='svg')

    plt.show()

def plot_heatmap(data, ax, fig, variable, m):
    print(m)
    im = ax.imshow(data, cmap="viridis", vmin=0, vmax=135, aspect='auto', origin='upper')
    cbar = fig.colorbar(im, ax=ax, ticks=[0, 65, 130])
    cbar.set_label(f'$\Delta$% Cumulative {variable}')
    ax.invert_yaxis()
    plt.tight_layout()

    return fig, ax

def create_heatmap_figure(horizons, uncertainties, title):
    fig = plt.figure(figsize=(WIDTH, HEIGHT), dpi=300)
    ax = plt.gca()
    # ax.set_title(title)
    ax.set_xticks(np.arange(len(horizons)))
    ax.set_xticklabels(horizons)

    # Only show specific yticks
    show_values = [0.05, 0.1, 0.15, 0.2]
    ytick_positions = [uncertainties.index(val) for val in show_values if val in uncertainties]
    ax.set_yticks(ytick_positions)
    ax.set_yticklabels(show_values)

    ax.set_xlabel("Prediction Horizon (H)")
    ax.set_ylabel("Uncertainty $(\delta)$")
    return fig, ax

if __name__ == "__main__":
    main()
