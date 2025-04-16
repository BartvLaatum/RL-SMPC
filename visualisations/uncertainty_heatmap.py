import os
import re
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import plot_config

plt.rcParams["axes.spines.top"] = True
plt.rcParams["axes.spines.right"] = True
plt.rcParams["axes.linewidth"] = 1  # Axis border thickness


WIDTH = 87.5 * 0.03937
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
    pattern = r".+-zero-order-terminal-(\d+)H-([\d.]+)\.csv"
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
    return df.groupby('run')[column].sum().mean()

def main():
    # Define directories for the two algorithms
    mpc_dir = 'data/uncertainty-comparison/stochastic/mpc'
    rlsmpc_dir = 'data/uncertainty-comparison/stochastic/rlsmpc'

    # Dictionaries to hold the mean reward for each (h, delta)
    mpc_rewards = {}
    rlsmpc_rewards = {}

    # Process mpc files
    mpc_pattern = os.path.join(mpc_dir, "mpc-*-*.csv")
    for filepath in glob.glob(mpc_pattern):
        h, delta = extract_params_mpc(filepath)
        if h is None:
            continue
        try:
            mean_reward = get_mean_reward(filepath)
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            continue
        mpc_rewards[(h, delta)] = mean_reward

    # Process rl-smpc files
    rlsmpc_pattern = os.path.join(rlsmpc_dir, "*-zero-order-terminal-*-*.csv")
    for filepath in glob.glob(rlsmpc_pattern):
        h, delta = extract_params_rlsmpc(filepath)
        if h is None:
            continue
        try:
            mean_reward = get_mean_reward(filepath)
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            continue
        rlsmpc_rewards[(h, delta)] = mean_reward

    # Get common (h, delta) pairs available in both datasets
    common_keys = set(mpc_rewards.keys()) & set(rlsmpc_rewards.keys())
    if not common_keys:
        print("No matching (prediction horizon, uncertainty) pairs found between mpc and rl-smpc data.")
        return

    # Create sorted lists of horizons and uncertainties
    horizons = sorted({h for h, _ in common_keys})
    uncertainties = sorted({delta for _, delta in common_keys})

    # Create a matrix for storing differences (rl-smpc mean reward - mpc mean reward)
    diff_matrix = np.full((len(uncertainties), len(horizons)), np.nan)
    for i, delta in enumerate(uncertainties):
        for j, h in enumerate(horizons):
            if (h, delta) in common_keys:
                diff = rlsmpc_rewards[(h, delta)] - mpc_rewards[(h, delta)]
                diff_matrix[i, j] = diff

    # Plot heatmap
    m = diff_matrix.max()

    fig = plt.figure(figsize=(WIDTH, HEIGHT), dpi=300)
    ax = plt.gca()
    im = ax.imshow(diff_matrix, cmap="coolwarm", vmin=-m, vmax=m, aspect='auto', origin='upper')
    ax.set_xticks(np.arange(len(horizons)))
    ax.set_xticklabels(horizons)
    
    # Only show specific yticks
    show_values = [0.05, 0.1, 0.15, 0.2]
    ytick_positions = [uncertainties.index(val) for val in show_values if val in uncertainties]
    ax.set_yticks(ytick_positions)
    ax.set_yticklabels(show_values)

    cbar = fig.colorbar(im, ax=ax, ticks=[-0.8, 0, 0.8])
    cbar.set_label('Diff in Mean Cumulative Reward')

    ax.set_xlabel("Prediction Horizon (H)")
    ax.set_ylabel("Uncertainty $(\delta)$")
    ax.set_title(r"RL$^0$-SMPC vs MPC")
    # Invert y-axis to make it ascending
    ax.invert_yaxis()
    plt.tight_layout()
    fig.savefig("figures/uncertainty-comparison/diff-rew-rlsmpc-mpc.png")
    # plt.show()

if __name__ == "__main__":
    main()
