import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import plot_config

WIDTH = 87.5 * 0.03937
HEIGHT = WIDTH * 0.75


def plot_mpc_vs_rl_smpc(data, labels):
    # Define default markers for different algorithms
    marker_map = {
        0: 'o',
        1: '^',
        2: 's',
        3: 'D',
    }

    # Compute global colorbar limits (vmin and vmax) from all algorithm data
    all_avg_sol = []
    for algo, results in data.items():
        for h in results:
            all_avg_sol.append(results[h]['solver_success'].mean())
    global_vmin = min(all_avg_sol) if all_avg_sol else 0
    global_vmax = max(all_avg_sol) if all_avg_sol else 1

    fig, ax = plt.subplots(figsize=(WIDTH, HEIGHT), dpi=180)
    scatter_plots = []  # keep reference for colorbar

    # Iterate over each algorithm in the nested dictionary
    for i, (algo, results) in enumerate(data.items()):
        horizons = sorted(results.keys())
        sum_rewards = []
        avg_solver = []
        for h in horizons:
            # Sum rewards and compute average solver_success
            sum_rewards.append(results[h].groupby('run')['rewards'].sum().mean())
            avg_solver.append(results[h]['solver_success'].mean())
        
        # Choose marker (case-insensitive matching)
        marker = marker_map.get(i, 'o')
        sc = ax.scatter(horizons, sum_rewards, c=avg_solver, cmap='plasma',
                marker=marker, s=50, label=labels[i], alpha=0.8,
                vmin=global_vmin, vmax=global_vmax)
        # Add line connecting the scatter points
        ax.plot(horizons, sum_rewards, '-', color=sc.get_facecolor()[0], alpha=0.5)
        scatter_plots.append(sc)
    
    ax.set_xlabel("Prediction Horizon")
    ax.set_ylabel("Sum of Rewards")
    ax.legend()
    
    # Use one scatter plot instance for the colorbar reference
    if scatter_plots:
        cbar = plt.colorbar(scatter_plots[-1], ax=ax)
        cbar.set_label("Average Solver Failure Rate")
        
    plt.tight_layout()
    fig.savefig("reward-solver-success.png")
    plt.show()

def bar_plot_solver_success(data, labels, colors):
    # Compute average solver_success for MPC and RL-SMPC per horizon

    success_rates = []

    for key, result in data.items():
        
        horizons = result.keys()
        success_rate = [result[h]['solver_success'].mean() for h in horizons]
        success_rates.append(success_rate)
        print(f"{key}: {[round(x, 4) for x in success_rate]}")

    # Setup grouped bar plot
    x = np.arange(len(horizons))
    bar_width = 0.35

    WIDTH = 87.5 * 0.03937
    HEIGHT = WIDTH * 0.75

    fig, ax = plt.subplots(figsize=(WIDTH, HEIGHT), dpi=300)

    # Calculate bar positions for each group
    positions = []
    n_bars = len(success_rates)
    total_width = bar_width * n_bars

    # For odd number of bars, center the middle bar
    # For even number, center between the two middle bars
    if n_bars % 2 == 0:
        start = -(total_width/2) + (bar_width/2)
    else:
        start = -(total_width/2) + (bar_width)
    
    for i in range(n_bars):
        positions.append(x + start + i*bar_width)

    # Plot bars
    for i, success_rate in enumerate(success_rates):
        ax.bar(positions[i], success_rate, bar_width, 
               label=labels[i], color=colors[i], alpha=0.8)

    ax.set_xlabel('Prediction Horizon')
    ax.set_ylabel('Average Failure Rate')
    ax.set_xticks(x)
    ax.set_xticklabels(horizons)
    ax.legend()
    plt.tight_layout()
    fig.savefig("barplot-solver-success.png")
    plt.show()

def load_data():
    data = {
        "mpc": {},
        "mpc-clipped": {},
        "rlsmpc": {},
        "smpc": {},
        "smpc-clipped": {},
    }
    mpc_dir = 'data/solver-success/stochastic/mpc'
    rlsmpc_dir = 'data/solver-success/stochastic/rlsmpc'
    smpc_dir = 'data/solver-success/stochastic/smpc'
    horizons = [1, 2, 3, 4, 5, 6]

    for h in horizons:
        mpc_file = f"{mpc_dir}/mpc-{h}H-0.1.csv"
        mpc_clipped_file = f"{mpc_dir}/mpc-box-constraints-{h}H-0.1.csv"
        rlsmpc_file = f"{rlsmpc_dir}/box-constraints-{h}H-0.1.csv"
        smpc_file = f"{smpc_dir}/smpc-{h}H-0.1.csv"
        smpc_clipped_file = f"{smpc_dir}/smpc-clipped-{h}H-0.1.csv"

        # Load MPC data
        if h not in data['mpc'] and os.path.exists(mpc_file):
            data['mpc'][h] = pd.read_csv(mpc_file)

        if h not in data['mpc-clipped'] and os.path.exists(mpc_clipped_file):
            data['mpc-clipped'][h] = pd.read_csv(mpc_clipped_file)

        if h not in data['smpc-clipped'] and os.path.exists(smpc_clipped_file):
            data['smpc-clipped'][h] = pd.read_csv(smpc_clipped_file)



        if h not in data["rlsmpc"] and os.path.exists(rlsmpc_file):
            data["rlsmpc"][h] = pd.read_csv(rlsmpc_file)

        if h not in data["smpc"] and os.path.exists(smpc_file):
            data["smpc"][h] = pd.read_csv(smpc_file)
    return data

def main():
    data = load_data()

    # comparison mpc and mpc-clipped
    subset_data = {
        "smpc": data.get("smpc", {}),
        "smpc-clipped": data.get("smpc-clipped", {})
    }

    labels = ["SMPC", "SMPC Box constraints"]
    # plot_mpc_vs_rl_smpc(subset_data, labels=labels)
    bar_plot_solver_success(subset_data, labels, colors=["C0","C3"])

    # comparison s/mpc and s/mpc-clipped
    subset_data = {
        "mpc-clipped": data.get("mpc-clipped", {}),
        "smpc-clipped": data.get("smpc-clipped", {}),
        "RL-SMPC": data.get("rlsmpc", {}),
    }

    labels = ["MPC", "SMPC", "RL-SMPC"]
    # bar_plot_solver_success(subset_data, labels, colors=["C0", "C8", "C3"])
    # plot_mpc_vs_rl_smpc(subset_data, labels=labels)

    # bar_plot_solver_success(data, ["MPC", r"RL$^0$-SMPC", "SMPC"], colors=["C0", "C3", "C2"])
    # data = load_data_smpc()
    # plot_mpc_vs_rl_smpc(data, labels=["SMPC-1e-6", "SMPC-1e-7", "SMPC-1e-8", "SMPC"])
    # bar_plot_solver_success(data, ["SMPC-1e-6", "SMPC-1e-7", "SMPC-1e-8", "SMPC"], colors=["C0", "C1", "C2", "C3"])

if __name__ == "__main__":
    main()
