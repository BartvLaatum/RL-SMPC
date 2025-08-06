import os
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cmcrameri.cm as cmc

import plot_config
import re

def create_plot(figure_name, project, data, horizons, mode, variable, models2plot, uncertainty_value=None, linestyles=["-", "--"]):

    WIDTH = 87.5 * 0.03937
    HEIGHT = WIDTH * 0.75
    color_counter  = 0
    fig, ax = plt.subplots(figsize=(WIDTH, HEIGHT), dpi=300)
    
    # n_colors = max(len(data.keys()) + len(model_names), 8)  # Ensure at least 8 colors
    colors = cmc.batlowS
    pattern = re.compile(r"[-+]?\d*\.?\d+")
    horizon_nums = [float(pattern.search(h).group()) for h in horizons]
    i = 0
    for key, results in data.items():
        if key not in models2plot:
            continue
        mean_smpc_final_rewards = []
        std_smpc_final_rewards = []
        for h in horizons:
            if h in results:
                grouped_runs = results[h].groupby("run")
                cumulative_rewards = grouped_runs[variable].sum()

                mean_smpc_final_rewards.append(cumulative_rewards.mean())
                std_smpc_final_rewards.append(cumulative_rewards.std())
        if mean_smpc_final_rewards:
            n2plot = len(mean_smpc_final_rewards)
            ax.plot(horizon_nums[:n2plot], mean_smpc_final_rewards[:n2plot], 'o-', label=key, color=colors(color_counter), linestyle=linestyles[i])
            if mode == "stochastic":
                ax.fill_between(
                    horizon_nums[:n2plot], 
                    np.array(mean_smpc_final_rewards[:n2plot]) - np.array(std_smpc_final_rewards[:n2plot]),
                    np.array(mean_smpc_final_rewards[:n2plot]) + np.array(std_smpc_final_rewards[:n2plot]),
                    color=colors(color_counter), alpha=0.2
                )
            i += 1
            color_counter += 1
    ax.set_xlabel('Prediction Horizon (H)')
    if variable == 'rewards':
        ax.set_ylabel(f'Cumulative {variable[:-1]}')
        ax.legend()
    elif variable == 'econ_rewards':
        ax.set_ylabel(f'Cumulative EPI')
    elif variable == 'penalties':
        ax.set_ylabel(f'Cumulative penalty')
    ax.grid()
    plt.tight_layout()
    
    # Save plot
    dir_path = f'figures/{project}/{mode}/{figure_name}/'
    os.makedirs(dir_path, exist_ok=True)
    if variable == "econ_rewards":
        variable = "EPI"
    uncertainty_suffix = f'-{uncertainty_value}' if uncertainty_value else ''
    plt.savefig(f'{dir_path}{variable}{uncertainty_suffix}.png', 
                bbox_inches='tight', dpi=300)
    print("Saved figure to: ", f'{dir_path}{variable}{uncertainty_suffix}.png')
    # plt.show()

def load_smpc(
        mode, 
        project,
        uncertainty_value=None,
        model_name="brisk-resonance-24",
    ):

    data = {
        "smpc-2Ns": {},
        "smpc-8Ns": {},
        "rl-smpc": {},
        "rl-smpc-no-tightening": {},
        "mpc": {},
        "mpc-warm-start": {},
    }

    horizons = ["1H", "2H", "3H", "4H", "5H", "6H"]
    uncertainty_suffix = f"-{uncertainty_value}" if uncertainty_value else ""

    for h in horizons:
        # mpc_path = f"data/{project}/{mode}/mpc/mpc-{h}{uncertainty_suffix}.csv"
        smpc_path = f"data/{project}/{mode}/smpc/tree-based-multiplicative-{h}{uncertainty_suffix}.csv"
        smpc_updated_path = f"data/{project}/{mode}/smpc/tree-based-multiplicative-{h}-8Ns{uncertainty_suffix}.csv"
        rlsmpc_path = f"data/{project}/{mode}/rlsmpc/{model_name}-zero-order-terminal-{h}{uncertainty_suffix}.csv"
        rlsmpc_updated_path = f"data/{project}/{mode}/rlsmpc/{model_name}-no-tightening-{h}{uncertainty_suffix}.csv"
        mpc_path = f"data/{project}/{mode}/mpc/mpc-{h}{uncertainty_suffix}.csv"
        mpc_warm_path_updated_path = f"data/{project}/{mode}/mpc/warm-start-{h}{uncertainty_suffix}.csv"


        if os.path.exists(smpc_path):
            if h not in data['smpc-2Ns']:
                data['smpc-2Ns'][h] = {}
            data['smpc-2Ns'][h]= pd.read_csv(smpc_path)

        if os.path.exists(smpc_updated_path):
            if h not in data['smpc-8Ns']:
                data['smpc-8Ns'][h] = {}
            data['smpc-8Ns'][h]= pd.read_csv(smpc_updated_path)
            
        if os.path.exists(rlsmpc_path):
            if h not in data['rl-smpc']:
                data['rl-smpc'][h] = {}
            data['rl-smpc'][h]= pd.read_csv(rlsmpc_path)

        if os.path.exists(rlsmpc_updated_path):
            if h not in data['rl-smpc-no-tightening']:
                data['rl-smpc-no-tightening'][h] = {}
            data['rl-smpc-no-tightening'][h]= pd.read_csv(rlsmpc_updated_path)

        if os.path.exists(mpc_path):
            if h not in data['mpc']:
                data['mpc'][h] = {}
            data['mpc'][h]= pd.read_csv(mpc_path)

        if os.path.exists(mpc_warm_path_updated_path):
            if h not in data['mpc-warm-start']:
                data['mpc-warm-start'][h] = {}
            data['mpc-warm-start'][h]= pd.read_csv(mpc_warm_path_updated_path)

    return data, horizons

def plot_trajectories(data):
    fig, ax = plt.subplots(4, 1, figsize=(10, 6), dpi=300)
    for key, results in data.items():
        for h in results:
            if key != "smpc":
                ax[0].plot(results[h]["y_0"], label=f"{key}")
                ax[1].plot(results[h]["y_1"], label=f"{key}")
                ax[2].plot(results[h]["y_2"], label=f"{key}")
                ax[3].plot(results[h]["y_3"], label=f"{key}")
            # mean_smpc_final_rewards = cumulative_rewards.mean()
            # std_smpc_final_rewards = cumulative_rewards.std()
            # print(f"{key} - {h}: {mean_smpc_final_rewards} +/- {std_smpc_final_rewards}")
    ax[0].legend()
    plt.show()

    fig, ax = plt.subplots(3, 1, figsize=(10, 6), dpi=300)
    for key, results in data.items():

        for h in results:
            if key != "smpc":
                ax[0].plot(results[h]["u_0"], label=f"{key}")
                ax[1].plot(results[h]["u_1"], label=f"{key}")
                ax[2].plot(results[h]["u_2"], label=f"{key}")
    ax[0].legend()
    plt.show()


def main(args):
    data, horizons = load_smpc(args.mode, args.project, args.uncertainty_value)

    # models2plot = ["SMPC", "SMPC-update-fx-1e-6", "SMPC-update-fx-1e-7", "SMPC-update-fx-1e-8"]
    # models2plot = ["smpc", "smpc-no-tightening", "rl-smpc", "rl-smpc-no-tightening"]
    # models2plot = ["smpc-2Ns", "smpc-8Ns"]
    models2plot = ["mpc", "mpc-warm-start"]
    print(data["mpc"]["1H"]["run"])
    # plot_trajectories(data)
    create_plot(args.figure_name, args.project, data, horizons, args.mode, 'rewards', models2plot, args.uncertainty_value)
    create_plot(args.figure_name, args.project, data, horizons, args.mode, 'econ_rewards', models2plot, args.uncertainty_value)
    create_plot(args.figure_name, args.project, data, horizons, args.mode, 'penalties', models2plot, args.uncertainty_value)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=str, default='smpc_test',
                        help='Name of the project')
    parser.add_argument('--mode', type=str, 
                        choices=['deterministic', 'stochastic'], required=True)
    parser.add_argument('--uncertainty_value', type=float,
                        help='Uncertainty scale value for stochastic mode')
    parser.add_argument('--figure_name', type=str, required=True)
    args = parser.parse_args()
    main(args)
