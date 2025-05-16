import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import plot_config

WIDTH = 87.5 * 0.03937
HEIGHT = WIDTH * 0.75


def plot_mpc_vs_rl_smpc(data, labels, colors):
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

    fig, ax = plt.subplots(figsize=(WIDTH, HEIGHT), dpi=300)
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
        # marker = marker_map.get(i, 'o')
        sc = ax.plot(horizons, sum_rewards, "-o", c=colors[i],
                label=labels[i], alpha=0.8,
                )
        # Add line connecting the scatter points
        # ax.plot(horizons, sum_rewards, '-', color=sc.get_facecolor()[0], alpha=0.5)
        # scatter_plots.append(sc)
    
    ax.set_xlabel("Prediction Horizon")
    ax.set_ylabel("Sum of Rewards")
    ax.legend()
    
    # Use one scatter plot instance for the colorbar reference
    if scatter_plots:
        cbar = plt.colorbar(scatter_plots[-1], ax=ax)
        cbar.set_label("Average Solver Failure Rate")
        
    plt.tight_layout()
    fig.savefig("reward-solver-success.png")
    # plt.show()

def bar_plot_solver_success(data, labels, colors):
    # Compute average solver_success for MPC and RL-SMPC per horizon

    success_rates = []
    success_rates_stds = []
    for key, result in data.items():

        horizons = result.keys()
        grouped_success_rate = [result[h].groupby("run")['solver_success'].mean() for h in horizons]
        mean_success_rate = [h.mean() for h in grouped_success_rate]
        std_success_rate = [h.std() for h in grouped_success_rate]
        # success_rate_std = [result[h]['solver_success'].std() for h in horizons]
        success_rates.append(mean_success_rate)
        success_rates_stds.append(std_success_rate)
        print(f"{key}: {[round(x, 4) for x in mean_success_rate]}")

    # Setup grouped bar plot
    x = np.arange(len(horizons))
    bar_width = 0.2  # Reduce bar width for more space between groups
    group_gap = 0.1  # Add gap between groups

    fig, ax = plt.subplots(figsize=(WIDTH, HEIGHT), dpi=300)

    n_bars = len(success_rates)
    total_group_width = n_bars * bar_width + (n_bars - 1) * 0  # no gap within group

    # Calculate bar positions for each group, with space between groups
    positions = []
    for i in range(n_bars):
        group_offsets = (bar_width * n_bars + group_gap) * x
        bar_offset = (i - (n_bars - 1) / 2) * bar_width
        positions.append(group_offsets + bar_offset)

    # Plot bars with error bars for standard deviation
    for i, (success_rate, std_success_rate) in enumerate(zip(success_rates, success_rates_stds)):
        ax.bar(positions[i], success_rate, bar_width, 
               label=labels[i], color=colors[i], alpha=0.8)#, yerr=std_success_rate, 
            #    capsize=3, error_kw={'elinewidth': 1})

    # Set x-ticks in the center of each group
    group_centers = (bar_width * n_bars + group_gap) * x
    ax.set_xlabel('Prediction Horizon')
    ax.set_ylabel('Average Failure Rate')
    ax.set_xticks(group_centers)
    ax.set_xticklabels(horizons)
    ax.legend()
    plt.tight_layout()
    fig.savefig("barplot-solver-success.png")
    plt.show()

def bar_plot_solver_time(data, labels, colors):
    # Compute average solver_success for MPC and RL-SMPC per horizon
    print("Solver times:")
    solver_times = []
    solver_times_stds = []
    for key, result in data.items():
        
        horizons = result.keys()
        grouped_solver_times = [result[h].groupby("run")['solver_times'].mean() for h in horizons]
        solver_time = [h.mean() for h in grouped_solver_times]
        solver_time_std = [h.std() for h in grouped_solver_times]
        solver_times.append(solver_time)
        solver_times_stds.append(solver_time_std)
        print(f"{key}: {[round(x, 4) for x in solver_time]}")

    # Setup grouped bar plot
    x = np.arange(len(horizons))
    bar_width = 0.35

    WIDTH = 87.5 * 0.03937
    HEIGHT = WIDTH * 0.75

    fig, ax = plt.subplots(figsize=(WIDTH, HEIGHT), dpi=300)

    # Calculate bar positions for each group
    positions = []
    n_bars = len(solver_times)
    total_width = bar_width * n_bars

    # For odd number of bars, center the middle bar
    # For even number, center between the two middle bars
    if n_bars % 2 == 0:
        start = -(total_width/2) + (bar_width/2)
    else:
        start = -(total_width/2) + (bar_width)
    
    for i in range(n_bars):
        positions.append(x + start + i*bar_width)

    # Plot bars with error bars for standard deviation
    for i, (solver_time, solver_time_std) in enumerate(zip(solver_times, solver_times_stds)):
        ax.bar(positions[i], solver_time, bar_width, 
               label=labels[i], color=colors[i], alpha=0.8, yerr=solver_time_std, 
               capsize=3, error_kw={'elinewidth': 1})

    # Set y-ticks
    ax.set_xlabel('Prediction Horizon')
    ax.set_ylabel('Average Solver Time (s)')
    ax.set_xticks(x)
    ax.set_xticklabels(horizons)
    ax.legend()
    plt.tight_layout()
    fig.savefig("barplot-solver-time.png")
    plt.show()


def load_data():
    data = {
        "mpc": {},
        "mpc-clipped": {},
        "mpc-warmstart": {},
        "mpc-warmstart-clipped": {},
        "rlsmpc": {},
        "smpc": {},
        "smpc-clipped": {},
        "smpc-warmstart": {},
        "smpc-warmstart-clipped": {},

    }
    mpc_dir = 'data/solver-success/stochastic/mpc'
    rlsmpc_dir = 'data/solver-success/stochastic/rlsmpc'
    smpc_dir = 'data/solver-success/stochastic/smpc'
    horizons = [1, 2, 3, 4, 5, 6]

    for h in horizons:
        mpc_file = f"{mpc_dir}/mpc-{h}H-0.1.csv"
        mpc_clipped_file = f"{mpc_dir}/mpc-box-constraints-{h}H-0.1.csv"
        mpc_warmstart_file = f"{mpc_dir}/warm-start-{h}H-0.1.csv"
        mpc_warmstart_clipped_file = f"{mpc_dir}/box-warm-start-{h}H-0.1.csv"
        rlsmpc_file = f"{rlsmpc_dir}/box-constraints-{h}H-0.1.csv"
        smpc_file = f"{smpc_dir}/smpc-{h}H-0.1.csv"
        smpc_clipped_file = f"{smpc_dir}/box-constraints-{h}H-0.1.csv"
        smpc_warmstart_file = f"{smpc_dir}/warm-start-{h}H-0.1.csv"
        smpc_warmstart_clipped_file = f"{smpc_dir}/box-warm-start-{h}H-0.1.csv"

        # Load MPC data
        if h not in data['mpc'] and os.path.exists(mpc_file):
            data['mpc'][h] = pd.read_csv(mpc_file)

        if h not in data['mpc-clipped'] and os.path.exists(mpc_clipped_file):
            data['mpc-clipped'][h] = pd.read_csv(mpc_clipped_file)

        if h not in data['mpc-warmstart'] and os.path.exists(mpc_warmstart_file):
            data['mpc-warmstart'][h] = pd.read_csv(mpc_warmstart_file)

        if h not in data['mpc-warmstart-clipped'] and os.path.exists(mpc_warmstart_clipped_file):
            data['mpc-warmstart-clipped'][h] = pd.read_csv(mpc_warmstart_clipped_file)

        if h not in data["smpc"] and os.path.exists(smpc_file):
            data["smpc"][h] = pd.read_csv(smpc_file)

        if h not in data['smpc-clipped'] and os.path.exists(smpc_clipped_file):
            data['smpc-clipped'][h] = pd.read_csv(smpc_clipped_file)

        if h not in data['smpc-warmstart'] and os.path.exists(smpc_warmstart_file):
            data['smpc-warmstart'][h] = pd.read_csv(smpc_warmstart_file)

        if h not in data['smpc-warmstart-clipped'] and os.path.exists(smpc_warmstart_clipped_file):
            data['smpc-warmstart-clipped'][h] = pd.read_csv(smpc_warmstart_clipped_file)

        if h not in data["rlsmpc"] and os.path.exists(rlsmpc_file):
            data["rlsmpc"][h] = pd.read_csv(rlsmpc_file)
    return data

def main():
    data = load_data()

    # comparison mpc and mpc-clipped
    subset_data = {
        "mpc": data.get("mpc", {}),
        "mpc-clipped": data.get("mpc-clipped", {}),
        "mpc-warmstart": data.get("mpc-warmstart", {}),
        "mpc-warmstart-clipped": data.get("mpc-warmstart-clipped", {})
    }

    colors  = ["C0", "C1", "C2", "C3"]
    labels = ["MPC", "MPC box constraints", "MPC warm start", "MPC warm start + box constraints"]
    # plot_mpc_vs_rl_smpc(subset_data, labels=labels, colors=colors)
    # bar_plot_solver_success(subset_data, labels, colors=colors)

    subset_data = {
        # "smpc": data.get("smpc", {}),
        "smpc-clipped": data.get("smpc-clipped", {}),
        "smpc-warmstart": data.get("smpc-warmstart", {}),
        "smpc-warmstart-clipped": data.get("smpc-warmstart-clipped", {})
    }
    labels = ["SMPC", "SMPC box constraints", "SMPC warm start", "SMPC warm start + box constraints"]
    plot_mpc_vs_rl_smpc(subset_data, labels=labels[1:], colors=colors[1:])
    # bar_plot_solver_success(subset_data, labels, colors=colors)

    # comparison s/mpc and s/mpc-clipped
    # subset_data = {
    #     "mpc-clipped": data.get("mpc-clipped", {}),
    #     "smpc-clipped": data.get("smpc-clipped", {}),
    #     "RL-SMPC": data.get("rlsmpc", {}),
    # }

    # labels = ["MPC", "SMPC", "RL-SMPC"]
    # bar_plot_solver_success(subset_data, labels, colors=["C0", "C8", "C3"])


    # bar_plot_solver_success(data, ["MPC", r"RL$^0$-SMPC", "SMPC"], colors=["C0", "C3", "C2"])
    # data = load_data_smpc()
    # plot_mpc_vs_rl_smpc(data, labels=["SMPC-1e-6", "SMPC-1e-7", "SMPC-1e-8", "SMPC"])
    # bar_plot_solver_success(data, ["SMPC-1e-6", "SMPC-1e-7", "SMPC-1e-8", "SMPC"], colors=["C0", "C1", "C2", "C3"])

if __name__ == "__main__":
    main()
