import os
import argparse

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plot_config

def compute_violations(
        df,
        lb_states=[0, 500, 10, 0],
        ub_states=[np.inf, 1600, 20, 80],
        lb_weights=[0, 5.e-5, 0.003, 7.e-4],
        ub_weights=[0, 5.e-5, 0.005, 7.e-4]
    ):
    state_cols = [col for col in df.columns if col.startswith('y')]
    violations = pd.DataFrame(0, index=df.index, columns=state_cols)
    for i, col in enumerate(state_cols):
        lb = lb_states[i]
        ub = ub_states[i]
        lb_violations = (df[col] < lb) * (lb - df[col]) * lb_weights[i]
        ub_violations = (df[col] > ub) * (df[col] - ub) * ub_weights[i]
        violations[col] = lb_violations + ub_violations
    df['violations'] = violations.sum(axis=1)
    return df

def load_data(
        model_names, 
        mode, 
        project,
        smpc=True,
        uncertainty_value=None,
        Ns=10,
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
    horizon = '1H'
    pen_weight_factors = [0.1, 0.316, 1.0, 3.16, 10.0]
    # pen_weight_factors = [0.316, 1.0, 3.16, 10.0]

    # Initialize data structure for all possible data types
    data = {
        'smpc': {},
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
        for penalty_weight_factor in pen_weight_factors:
            # Define file paths for different RL-SMPC configurations
            rlsmpc_terminal_path = f'data/{project}/{mode}/rlsmpc/{model}-no-tightening-{horizon}{uncertainty_suffix}-{Ns}Ns.csv'

            if os.path.exists(rlsmpc_terminal_path):
                if penalty_weight_factor not in data['rl-zero-terminal-smpc']:
                    data['rl-zero-terminal-smpc'][penalty_weight_factor] = {}
                data['rl-zero-terminal-smpc'][penalty_weight_factor][model] = pd.read_csv(rlsmpc_terminal_path)

    # --- Load MPC and SMPC data for each horizon ---
    for penalty_weight_factor in pen_weight_factors:
        # Define file paths for MPC and SMPC data
        smpc_path = f'data/{project}/{mode}/smpc/no-tightening-{horizon}{uncertainty_suffix}-{Ns}Ns-{penalty_weight_factor}.csv'
        print(smpc_path)
        # Load SMPC data if requested and file exists
        if smpc:
            if os.path.exists(smpc_path):
                if penalty_weight_factor not in data['smpc']:
                    data['smpc'][penalty_weight_factor] = {}
                data['smpc'][penalty_weight_factor]= pd.read_csv(smpc_path)

    return data, pen_weight_factors

def plot_pen_weight_factor(
        data,
        pen_weight_factors,
        args,
        var2plot,
        project,
        mode,
        uncertainty_value,
        figure_name
    ):
    WIDTH = 60 * 0.0393700787
    HEIGHT = WIDTH * 0.75
    fig, ax = plt.subplots(figsize=(WIDTH, HEIGHT), dpi=300)

    ### --- Plot SMPC data --- ###
    rl_data = data['rl']
    rl_rewards = []
    rl_violations = []
    rl_rewards_std = []
    rl_violations_std = []
    for model_name, df in rl_data.items():
        df = compute_violations(df)
        df['orig_rew'] = df["econ_rewards"] - df["violations"]
        grouped_runs = df.groupby("run")
        cumulative_rewards = grouped_runs[var2plot].sum()
        rl_rewards.append(cumulative_rewards.mean())
        rl_rewards_std.append(cumulative_rewards.std())
        cumulative_violations = grouped_runs['violations'].sum()
        rl_violations.append(cumulative_violations.mean())
        rl_violations_std.append(cumulative_violations.std())

    # ax.errorbar(rl_rewards, rl_violations, xerr=rl_rewards_std, yerr=rl_violations_std, fmt='o', color="grey", alpha=0.5, zorder=4)
    # ax.plot(rl_rewards, rl_violations, color="grey", alpha=0.3)
    print(rl_rewards)
    ax.plot(pen_weight_factors, rl_rewards, 'o-', color="grey", alpha=0.8, label="RL")

    ### --- Plot RL-SMPC data --- ###
    smpc_data = data['smpc']
    smpc_rewards = []
    smpc_violations = []
    smpc_rewards_std = []
    smpc_violations_std = []
    i = 0
    for pen_weight_factor, df in smpc_data.items():
        df = compute_violations(df)
        df['orig_rew'] = df["econ_rewards"] - df["violations"]

        grouped_runs = df.groupby("run")
        cumulative_rewards = grouped_runs[var2plot].sum()
        smpc_rewards.append(cumulative_rewards.mean())
        cumulative_violations = grouped_runs['violations'].sum()
        smpc_violations.append(cumulative_violations.mean())
        smpc_rewards_std.append(cumulative_rewards.std())
        smpc_violations_std.append(cumulative_violations.std())
    ax.plot(pen_weight_factors, smpc_rewards, 'o-', color="C0", alpha=0.8, label="SMPC")
    ax.fill_between(pen_weight_factors, np.array(smpc_rewards)-np.array(smpc_rewards_std), np.array(smpc_rewards)+np.array(smpc_rewards_std), color="C0", alpha=0.3)

    # ax.scatter(smpc_rewards, smpc_violations,  color="C0", alpha=0.8, label="SMPC", zorder=3)
    # ax.plot(smpc_rewards, smpc_violations,  color="C0", alpha=0.3, zorder=2)

    ### --- Plot RL-SMPC data --- ###
    rlsmpc_data = data['rl-zero-terminal-smpc']
    rlsmpc_rewards = []
    rlsmpc_violations = []
    rlsmpc_rewards_std = []
    rlsmpc_violations_std = []
    for i, (pen_weight_factor, model_dict) in enumerate(rlsmpc_data.items()):
        df = model_dict[args.model_names[i]]
        df = compute_violations(df)
        df['orig_rew'] = df["econ_rewards"] - df["violations"]

        grouped_runs = df.groupby("run")
        cumulative_rewards = grouped_runs[var2plot].sum()
        rlsmpc_rewards.append(cumulative_rewards.mean())
        cumulative_violations = grouped_runs['violations'].sum()
        rlsmpc_violations.append(cumulative_violations.mean())
        rlsmpc_rewards_std.append(cumulative_rewards.std())
        rlsmpc_violations_std.append(cumulative_violations.std())
    ax.plot(pen_weight_factors, rlsmpc_rewards, 'o-', color="C3", alpha=0.8, label="RL-SMPC")
    ax.fill_between(pen_weight_factors, np.array(rlsmpc_rewards)-np.array(rlsmpc_rewards_std), np.array(rlsmpc_rewards)+np.array(rlsmpc_rewards_std), color="C3", alpha=0.3)
    # ax.errorbar(rlsmpc_rewards, rlsmpc_violations, xerr=rlsmpc_rewards_std, yerr=rlsmpc_violations_std, fmt='o', color="C3", alpha=0.5, zorder=4)
    # ax.plot(rlsmpc_rewards, rlsmpc_violations,  color="C3", alpha=0.3, zorder=2)

    ax.set_xscale('log')
    ax.set_xlabel('penalty weight factor')
    ax.set_ylabel(f'Cumulative {var2plot.replace("_", " ")}')
    if var2plot == "violations":
        ax.set_yscale('log')
        ax.set_ylabel('Cumulative original penalty')
    elif var2plot == "orig_rew":
        ax.set_ylabel('Cumulative reward')
    if var2plot == "econ_rewards":
        ax.set_ylabel('EPI')
    ax.legend()
    plt.tight_layout()

    dir_path = f'figures/{project}/{mode}/{figure_name}/'
    os.makedirs(dir_path, exist_ok=True)
    # Use a more descriptive filename for economic rewards
    uncertainty_suffix = f'-{uncertainty_value}' if uncertainty_value else ''
    fig.savefig(f'{dir_path}{var2plot}-4points{uncertainty_suffix}.svg', format='svg',
                bbox_inches='tight', dpi=300)
    fig.savefig(f'{dir_path}{var2plot}-4points{uncertainty_suffix}.png', format='png',
                bbox_inches='tight', dpi=300)
    
    fig, ax = plt.subplots(figsize=(WIDTH, HEIGHT), dpi=300)
    # Compute relative difference, handling negative values correctly
    rlsmpc_diff = np.array(rlsmpc_rewards) - np.array(smpc_rewards)
    x = np.array(smpc_rewards)
    y = np.array(rlsmpc_rewards)
    sx = np.array(smpc_rewards_std)
    sy = np.array(rlsmpc_rewards_std)

    sign_x = np.sign(x)
    df_dy = 100 / np.abs(x)
    df_dx = -100 / np.abs(x) - 100 * (y - x) * sign_x / (x**2)
    r_std = np.sqrt((df_dx * sx)**2 + (df_dy * sy)**2)
    ax.plot(pen_weight_factors, rlsmpc_diff, 'o-', color="grey", alpha=0.8, label="RL-SMPC - SMPC")
    # ax.fill_between(pen_weight_factors, rlsmpc_diff - r_std, rlsmpc_diff + r_std, color="grey", alpha=0.3)
    ax.set_xscale('log')
    ax.set_xlabel('Penalty weight factor')
    ax.set_ylabel(r'$\Delta\%$ RL-SMPC vs SMPC')
    plt.tight_layout()
    fig.savefig(f'{dir_path}difference{uncertainty_suffix}.svg', format='svg',
                bbox_inches='tight', dpi=300)
    fig.savefig(f'{dir_path}difference{uncertainty_suffix}.png', format='png',
                bbox_inches='tight', dpi=300)


def main(args):
    data, pen_weights_factors = load_data(
        args.model_names, 
        args.mode, 
        args.project, 
        smpc=args.smpc,
        uncertainty_value=args.uncertainty_value
    )

    vars2plot = ["rewards", "orig_rew", "econ_rewards", "violations", "penalties", "difference"]
    # for var2plot in vars2plot:
        # plot_pen_weight_factor(data, pen_weights_factors, args, var2plot)
    plot_pen_weight_factor(data, pen_weights_factors, args, "violations", args.project, args.mode, args.uncertainty_value, args.figure_name)

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
    parser.add_argument('--terminal', action=argparse.BooleanOptionalAction,
                        help='Whether to visualise RL-SMPC with terminal state/cost implemented')
    parser.add_argument('--mode', type=str, 
                        choices=['deterministic', 'stochastic'], required=True)
    parser.add_argument('--uncertainty_value', type=float,
                        help='Uncertainty scale value for stochastic mode')
    parser.add_argument('--figure_name', type=str, required=True)
    args = parser.parse_args()
    main(args)
