import os
import argparse

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable
import pandas as pd
import numpy as np
import  matplotlib.ticker as ticker
from visualisations.rl_smpc_performance import load_data
import plot_config
from tabulate import tabulate


WIDTH = 90 * 0.0393700787
HEIGHT = WIDTH * 0.75

def load_data(
        model_names, 
        mode, 
        project,
        mpc=False,
        smpc=True,
        zero_order=True,
        terminal=True,
        first_order=False, 
        Ns=10, 
        uncertainty_value=None
    ):
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
    # horizons = ['1H', '2H', '3H', '4H']
    data = {
        'mpc': {},
        'smpc': {},
        'rl': {},
        'rl-zero-terminal-smpc': {},
        'rl-first-terminal-smpc': {},
    }

    uncertainty_suffix = f'-{uncertainty_value}' if uncertainty_value else ''

    for model in model_names:
        # Load RL data
        rl_path = f'data/{project}/{mode}/rl/{model}.csv'
        if os.path.exists(rl_path):
            data['rl'][model] = pd.read_csv(rl_path)

        # Load MPC and RL-MPC data for each horizon
        for h in horizons:
            rlsmpc_terminal_path = f'data/{project}/{mode}/rlsmpc/{model}-no-tightening-{h}{uncertainty_suffix}-{Ns}Ns.csv'

            if zero_order:
                if terminal:
                    if os.path.exists(rlsmpc_terminal_path):
                        if h not in data['rl-zero-terminal-smpc']:
                            data['rl-zero-terminal-smpc'][h] = {}
                        data['rl-zero-terminal-smpc'][h][model] = pd.read_csv(rlsmpc_terminal_path)

    for h in horizons:
        smpc_path = f'data/{project}/{mode}/smpc/no-tightening-{h}{uncertainty_suffix}-{Ns}Ns.csv'

        if smpc:
            if os.path.exists(smpc_path):
                if h not in data['smpc']:
                    data['smpc'][h] = {}
                data['smpc'][h]= pd.read_csv(smpc_path)

    return data, horizons

def runtime_performance_plot_all_samples(fig, ax, data, horizons, model_names, frac, cmaps):
    mean_smpc_runtime = []
    mean_smpc_final_reward = []
    std_smpc_runtime = []
    std_smpc_final_reward = []
    for h in horizons:
        if h in data['smpc']:
            grouped_runs = data['smpc'][h].groupby("run")
            average_runtime = grouped_runs["solver_times"].mean()
            cumulative_rewards = grouped_runs["rewards"].sum()

            mean_smpc_runtime.append(average_runtime.mean())
            mean_smpc_final_reward.append(cumulative_rewards.mean())
            std_smpc_runtime.append(average_runtime.std())
            std_smpc_final_reward.append(cumulative_rewards.std())


    for idx, model in enumerate(model_names):
        mean_rl_smpc_runtime = []
        mean_rl_smpc_final_reward = []
        std_rl_smpc_runtime = []
        std_rl_smpc_final_reward = []

        for h in horizons:
            if h in data['rl-zero-terminal-smpc']:
                grouped_runs = data['rl-zero-terminal-smpc'][h][model].groupby("run")
                average_runtime = grouped_runs["solver_times"].mean()
                cumulative_rewards = grouped_runs["rewards"].sum()
                mean_rl_smpc_runtime.append(average_runtime.mean())
                mean_rl_smpc_final_reward.append(cumulative_rewards.mean())
                std_rl_smpc_runtime.append(average_runtime.std())
                std_rl_smpc_final_reward.append(cumulative_rewards.std())

    ax.errorbar(
        mean_rl_smpc_final_reward,
        mean_rl_smpc_runtime,
        xerr=std_rl_smpc_runtime,
        yerr=std_rl_smpc_final_reward,
        fmt='o',
        color=cmaps[1](frac),
        label="RL-SMPC",
        alpha=0.8,
    )

    ax.errorbar(
    mean_smpc_final_reward,
    mean_smpc_runtime,
    xerr=std_smpc_final_reward,
    yerr=std_smpc_runtime,
    fmt='o',
    color=cmaps[0](frac),
    label="SMPC",
    alpha=0.8,
    )

    plt.plot(mean_rl_smpc_final_reward, mean_rl_smpc_runtime, color=cmaps[1](frac))
    plt.plot(mean_smpc_final_reward, mean_smpc_runtime, color=cmaps[0](frac))

def runtime_performance_plot(data, horizons, model_names):
    WIDTH = (101.85-18) * 0.0393700787
    HEIGHT = WIDTH * 0.75
    color_counter  = 0
    fig, ax = plt.subplots(figsize=(WIDTH, HEIGHT), dpi=300)

    mean_smpc_runtime = []
    mean_smpc_final_reward = []
    std_smpc_runtime = []
    std_smpc_final_reward = []
    for h in horizons:
        if h in data['smpc']:
            # if n in data['smpc'][h]:
            grouped_runs = data['smpc'][h].groupby("run")
            average_runtime = grouped_runs["solver_times"].mean()
            cumulative_rewards = grouped_runs["rewards"].sum()

            mean_smpc_runtime.append(average_runtime.mean())
            mean_smpc_final_reward.append(cumulative_rewards.mean())
            std_smpc_runtime.append(average_runtime.std())
            N_runs = len(cumulative_rewards)
            std_smpc_final_reward.append(cumulative_rewards.std() / np.sqrt(N_runs) * 1.96)

    for idx, model in enumerate(model_names):
        mean_rl_smpc_runtime = []
        mean_rl_smpc_final_reward = []
        std_rl_smpc_runtime = []
        std_rl_smpc_final_reward = []

        for h in horizons:
            if h in data['rl-zero-terminal-smpc']:
                grouped_runs = data['rl-zero-terminal-smpc'][h][model].groupby("run")
                average_runtime = grouped_runs["solver_times"].mean()
                cumulative_rewards = grouped_runs["rewards"].sum()
                mean_rl_smpc_runtime.append(average_runtime.mean())
                mean_rl_smpc_final_reward.append(cumulative_rewards.mean())
                std_rl_smpc_runtime.append(average_runtime.std())
                N_runs = len(cumulative_rewards)
                std_rl_smpc_final_reward.append(cumulative_rewards.std() / np.sqrt(N_runs) * 1.96)

    ax.errorbar(
        mean_smpc_final_reward,
        mean_smpc_runtime,
        xerr=std_smpc_final_reward,
        yerr=std_smpc_runtime,
        fmt='o',
        color="C0",
        label="SMPC",
        alpha=0.8,
    )

    ax.errorbar(
        mean_rl_smpc_final_reward,
        mean_rl_smpc_runtime,
        xerr=std_rl_smpc_final_reward,
        yerr=std_rl_smpc_runtime,
        fmt='o',
        color="C3",
        label="RL-SMPC",
        alpha=0.8,
    )

    ax.plot(mean_smpc_final_reward, mean_smpc_runtime, color="C0", alpha=0.8)
    ax.plot(mean_rl_smpc_final_reward, mean_rl_smpc_runtime, color="C3", alpha=0.8)
    # Prepare data for tabulate
    rl_smpc_table = list(zip(horizons, mean_rl_smpc_runtime, std_rl_smpc_runtime, mean_rl_smpc_final_reward, std_rl_smpc_final_reward))
    smpc_table = list(zip(horizons, mean_smpc_runtime, std_smpc_runtime, mean_smpc_final_reward, std_smpc_final_reward))

    print("RL-SMPC Mean Runtime and Final Reward:")
    print(tabulate(rl_smpc_table, headers=["Horizon", "Mean Runtime", "95%CI Runtime", "Mean Final Reward", "95%CI Reward"], floatfmt=".4f"))

    print("\nSMPC Mean Runtime and Final Reward:")
    print(tabulate(smpc_table, headers=["Horizon", "Mean Runtime", "95%CI Runtime", "Mean Final Reward", "95%CI Reward"], floatfmt=".4f"))

    ax.set_xlabel("Cumulative reward")
    ax.legend()
    ax.set_ylabel("Average compute time (s)")
    ax.set_yticks([0, 1, 2, 3])
    ax.xaxis.set_major_locator(ticker.MaxNLocator(4))
    plt.tight_layout()
    plt.savefig(f"figures/{args.figure_name}.svg", format='svg', bbox_inches='tight', dpi=300)
    plt.savefig(f"figures/{args.figure_name}.png", format='png', bbox_inches='tight', dpi=300)
    plt.show()

def show_stats(args):
    Ns = [5, 10, 15, 20]
    table_smpc, table_rlsmpc = [], []
    for n in Ns:

        data, horizons = load_data(
            args.model_names, 
            args.mode, 
            args.project, 
            mpc=False,
            smpc=args.smpc,
            zero_order=args.zero_order,
            first_order=False,
            terminal=args.terminal, 
            Ns=n, 
            uncertainty_value=args.uncertainty_value
        )
        for h in horizons[:4]:
            try:
                grouped_runs = data['smpc'][h].groupby("run")
                average_runtime = grouped_runs["solver_times"].mean()
                cumulative_rewards = grouped_runs["rewards"].sum()
                table_smpc.append([n, h, average_runtime.mean(), average_runtime.std(), cumulative_rewards.mean(), cumulative_rewards.std()])
            except KeyError:
                pass

            grouped_runs = data['rl-zero-terminal-smpc'][h][args.model_names[0]].groupby("run")
            average_runtime = grouped_runs["solver_times"].mean()
            cumulative_rewards = grouped_runs["rewards"].sum()
            table_rlsmpc.append([n, h, average_runtime.mean(), average_runtime.std(), cumulative_rewards.mean(), cumulative_rewards.std()])
    df_smpc = pd.DataFrame(table_smpc, columns=["Ns", "Horizon", "Mean runtime", "Std runtime", "Mean reward", "Std reward"])
    df_rlsmpc = pd.DataFrame(table_rlsmpc, columns=["Ns", "Horizon", "Mean runtime", "Std runtime", "Mean reward", "Std reward"])
    print("SMPC")
    print(df_smpc)
    print("RL-SMPC")
    print(df_rlsmpc)

def single_sample_performance_plot(args):
    data, horizons = load_data(
            args.model_names, 
            args.mode, 
            args.project, 
            mpc=False,
            smpc=args.smpc,
            zero_order=args.zero_order,
            first_order=False,
            terminal=args.terminal, 
            Ns=args.Ns, 
            uncertainty_value=args.uncertainty_value
        )    
    runtime_performance_plot(data, horizons, args.model_names)

def multiple_sample_performance_plot(args):
    Ns = [5, 10, 15, 20]
    n_levels = len(Ns)
    # Build discrete color lists: light→dark for growing S
    # (skip the very lightest/darkest ends of the colormaps)
    pos = [(i + 1) / (n_levels + 1) for i in range(n_levels)]
    base_smpc = np.array(plt.get_cmap("tab10")(0))  # blueish
    base_rlsmpc = np.array(plt.get_cmap("tab10")(3))  # orange-red

    def make_shades(base_rgba, n, t_min=0.35, t_max=1.0):
        # t=0 white, t=1: base color
        t = np.linspace(t_min, t_max, n)[:, None]
        white = np.ones(4)[None, :]  # RGBA
        base = np.array(base_rgba)[None, :]
        return (1 - t) * white + t * base

    base_smpc   = plt.get_cmap("tab10")(0)  # blue-ish
    base_rlsmpc = plt.get_cmap("tab10")(3)  # orange-red

    smpc_shades   = make_shades(base_smpc,   n_levels)
    rlsmpc_shades = make_shades(base_rlsmpc, n_levels)

    # DISCRETE colormaps + matching norm
    cmap_smpc   = mcolors.ListedColormap(smpc_shades)
    cmap_rlsmpc = mcolors.ListedColormap(rlsmpc_shades)

    fig, ax = plt.subplots(figsize=(WIDTH, HEIGHT), dpi=300)
    for idx, n in enumerate(Ns):
        data, horizons = load_data(
            args.model_names, 
            args.mode, 
            args.project, 
            mpc=False,
            smpc=args.smpc,
            zero_order=args.zero_order,
            first_order=False,
            terminal=args.terminal, 
            Ns=n, 
            uncertainty_value=args.uncertainty_value
        )    
        frac = (idx) / (n_levels)
        runtime_performance_plot_all_samples(fig, ax, data, horizons, args.model_names, frac, cmaps=[cmap_smpc, cmap_rlsmpc])

    ax.set_xlabel("Cumulative reward")
    ax.set_yticks([0, 1, 2, 3])
    ax.set_ylabel("Average runtime (s)")

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(),
              ncol=2, frameon=False, loc='upper left')

    # Boundaries wrap each discrete color; ticks at centers
    bounds = np.arange(n_levels + 1)
    tick_locs = np.arange(n_levels) + 0.5
    norm_disc = mcolors.BoundaryNorm(bounds, ncolors=n_levels)

    # Put two thin cbar axes to the right of the plot
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax_smpc  = divider.append_axes("right", size="4%", pad=0.10)
    cax_rlsmpc = divider.append_axes("right", size="4%", pad=0.45)

    # Create the two bars
    sm_smpc  = ScalarMappable(cmap=cmap_smpc,  norm=norm_disc); sm_smpc.set_array([])
    sm_rlsmpc = ScalarMappable(cmap=cmap_rlsmpc, norm=norm_disc); sm_rlsmpc.set_array([])

    cb1 = fig.colorbar(sm_smpc,  cax=cax_smpc,  boundaries=bounds, ticks=tick_locs)
    cb2 = fig.colorbar(sm_rlsmpc, cax=cax_rlsmpc, boundaries=bounds, ticks=tick_locs)

    # Label & style
    cb1.ax.set_yticklabels(Ns)
    cb2.ax.set_yticklabels(Ns)
    cb2.set_label("Number of samples ($S$)", labelpad=10, rotation=270)
    cb1.ax.set_title("SMPC", pad=4, fontsize=8)
    cb2.ax.set_title("RL-SMPC", pad=4, fontsize=8)
    plt.tight_layout()
    plt.savefig(f"figures/{args.figure_name}.svg", format='svg', bbox_inches='tight', dpi=300)
    plt.savefig(f"figures/{args.figure_name}.png", format='png', bbox_inches='tight', dpi=300)

    # plt.show()

def main(args):
    single_sample_performance_plot(args)
    # multiple_sample_performance_plot(args)
    show_stats(args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=str, default='SMPC',
                        help='Name of the project')
    parser.add_argument('--model_names', nargs='+', type=str, default=[],
                        help='List of model names to plot')
    parser.add_argument('--smpc', action=argparse.BooleanOptionalAction,
                        help='Whether to plot SMPC')
    parser.add_argument('--zero-order', action=argparse.BooleanOptionalAction,
                        help='Whether to RL-SMPC with zero-order approximation')
    parser.add_argument('--first-order', action=argparse.BooleanOptionalAction,
                        help='Whether to RL-SMPC with first-order approximation')
    parser.add_argument('--terminal', action=argparse.BooleanOptionalAction,
                        help='Whether to visualise RL-SMPC with terminal state/cost implemented')
    parser.add_argument('--Ns', type=int, default=10,
                        help='List of SMPC with Ns scenario samples to plot')
    parser.add_argument('--mode', type=str, 
                        choices=['deterministic', 'stochastic'], required=True)
    parser.add_argument('--uncertainty_value', type=float,
                        help='Uncertainty scale value for stochastic mode')
    parser.add_argument('--figure_name', type=str, required=True)
    args = parser.parse_args()
    main(args)
