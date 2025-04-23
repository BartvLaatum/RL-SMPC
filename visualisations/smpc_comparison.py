import os
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cmcrameri.cm as cmc

import plot_config

def create_plot(figure_name, project, data, horizons, mode, variable, models2plot, uncertainty_value=None):

    WIDTH = 87.5 * 0.03937
    HEIGHT = WIDTH * 0.75
    color_counter  = 0
    fig, ax = plt.subplots(figsize=(WIDTH, HEIGHT), dpi=300)
    
    # n_colors = max(len(data.keys()) + len(model_names), 8)  # Ensure at least 8 colors
    colors = cmc.batlowS
    horizon_nums = [int(h[0]) for h in horizons]

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
            ax.plot(horizon_nums[:n2plot], mean_smpc_final_rewards[:n2plot], 'o-', label=key, color=colors(color_counter))
            if mode == "stochastic":
                ax.fill_between(
                    horizon_nums[:n2plot], 
                    np.array(mean_smpc_final_rewards[:n2plot]) - np.array(std_smpc_final_rewards[:n2plot]),
                    np.array(mean_smpc_final_rewards[:n2plot]) + np.array(std_smpc_final_rewards[:n2plot]),
                    color=colors(color_counter), alpha=0.2
                )
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
    plt.show()

def load_smpc(
        mode, 
        project,
        uncertainty_value=None
    ):

    data = {
        "MPC": {},
        "SMPC": {},
        "SMPC-update-fx-1e-6": {},
        "SMPC-update-fx-1e-7": {},
        "SMPC-update-fx-1e-8": {},
        "SMPC-LR-Feedback": {},
        "SMPC-LR-Feedback-update-fx": {},
    }

    horizons = ["1H", "2H", "3H", "4H", "5H", "6H"]
    uncertainty_suffix = f"-{uncertainty_value}" if uncertainty_value else ""

    for h in horizons:
        mpc_path = f"data/{project}/{mode}/mpc/mpc-{h}{uncertainty_suffix}.csv"
        smpc_path = f"data/{project}/{mode}/smpc/smpc-noise-correction-{h}-10Ns{uncertainty_suffix}.csv"
        smpc_updated_path = f"data/{project}/{mode}/smpc/smpc-update-fx-{h}-10Ns{uncertainty_suffix}.csv"
        smpc_updated2_path = f"data/{project}/{mode}/smpc/smpc-update-fx-1e-7-{h}-10Ns{uncertainty_suffix}.csv"
        smpc_updated3_path = f"data/{project}/{mode}/smpc/smpc-update-fx-1e-8-{h}-10Ns{uncertainty_suffix}.csv"
        # smpc_lr_feedback_path = f"data/{project}/{mode}/smpc/lr-feedback-{h}-10Ns{uncertainty_suffix}.csv"
        smpc_lr_feedback_updated_path = f"data/{project}/{mode}/smpc/lr-feedback-update-fx-{h}-10Ns{uncertainty_suffix}.csv"

        if os.path.exists(mpc_path):
            if h not in data['MPC']:
                data['MPC'][h] = {}
            data['MPC'][h]= pd.read_csv(mpc_path)

        if os.path.exists(smpc_path):
            if h not in data['SMPC']:
                data['SMPC'][h] = {}
            data['SMPC'][h]= pd.read_csv(smpc_path)

        if os.path.exists(smpc_updated_path):
            if h not in data['SMPC-update-fx-1e-6']:
                data['SMPC-update-fx-1e-6'][h] = {}
            data['SMPC-update-fx-1e-6'][h]= pd.read_csv(smpc_updated_path)

        if os.path.exists(smpc_updated2_path):
            if h not in data['SMPC-update-fx-1e-7']:
                data['SMPC-update-fx-1e-7'][h] = {}
            data['SMPC-update-fx-1e-7'][h]= pd.read_csv(smpc_updated2_path)

        if os.path.exists(smpc_updated3_path):
            if h not in data['SMPC-update-fx-1e-8']:
                data['SMPC-update-fx-1e-8'][h] = {}
            data['SMPC-update-fx-1e-8'][h]= pd.read_csv(smpc_updated3_path)


        # if os.path.exists(smpc_lr_feedback_path):
        #     if h not in data['SMPC-LR-Feedback']:
        #         data['SMPC-LR-Feedback'][h] = {}
        #     data['SMPC-LR-Feedback'][h]= pd.read_csv(smpc_lr_feedback_path)

        if os.path.exists(smpc_lr_feedback_updated_path):
            if h not in data['SMPC-LR-Feedback-update-fx']:
                data['SMPC-LR-Feedback-update-fx'][h] = {}
            data['SMPC-LR-Feedback-update-fx'][h]= pd.read_csv(smpc_lr_feedback_updated_path)

    return data, horizons

# def load_smpc_lr(
#         mode, 
#         project,
#         uncertainty_value=None
#     ):
#     data = {
#         "smpc-lr-feedback": {},
#         "smpc-lr-feedback-warm-init": {},
#     }

#     horizons = ["1H", "2H", "3H", "4H", "5H", "6H"]
#     uncertainty_suffix = f"-{uncertainty_value}" if uncertainty_value else ""

#     for h in horizons:
#         smpc_warm_init_path = f"data/{project}/{mode}/smpc/lr-feedback-warm-init-{h}-10Ns{uncertainty_suffix}.csv"
#         smpc_lr_feedback_path = f"data/{project}/{mode}/smpc/lr-feedback-{h}-10Ns{uncertainty_suffix}.csv"

#         if os.path.exists(smpc_path):
#             if h not in data['smpc-lr-feedback']:
#                 data['smpc-lr-feedback'][h] = {}
#             data['smpc-lr-feedback'][h]= pd.read_csv(smpc_path)
#         if os.path.exists(smpc_warm_init_path):
#             if h not in data['smpc-lr-feedback-warm-init']:
#                 data['smpc-lr-feedback-warm-init'][h] = {}
#             data['smpc-lr-feedback-warm-init'][h]= pd.read_csv(smpc_warm_init_path)
#     return data, horizons

# def load_smpc_and_lr(
#         mode, 
#         project,
#         uncertainty_value=None
#     ):
#     data = {
#         "smpc": {},
#         "smpc-lr-feedback": {},
#     }

#     horizons = ["1H", "2H", "3H", "4H", "5H", "6H"]
#     uncertainty_suffix = f"-{uncertainty_value}" if uncertainty_value else ""

#     for h in horizons:
#         smpc_path = f"data/{project}/{mode}/smpc/smpc-noise-correction-{h}-10Ns{uncertainty_suffix}.csv"
#         smpc_warm_init_path = f"data/{project}/{mode}/smpc/lr-feedback-{h}-10Ns{uncertainty_suffix}.csv"

#         if os.path.exists(smpc_path):
#             if h not in data['smpc']:
#                 data['smpc'][h] = {}
#             data['smpc'][h]= pd.read_csv(smpc_path)
#         if os.path.exists(smpc_warm_init_path):
#             if h not in data['smpc-lr-feedback']:
#                 data['smpc-lr-feedback'][h] = {}
#             data['smpc-lr-feedback'][h]= pd.read_csv(smpc_warm_init_path)
#     return data, horizons

def main(args):
    data, horizons = load_smpc(args.mode, args.project, args.uncertainty_value)

    # models2plot = ["SMPC", "SMPC-update-fx-1e-6", "SMPC-update-fx-1e-7", "SMPC-update-fx-1e-8"]
    models2plot = ["SMPC-update-fx-1e-6", "SMPC-LR-Feedback-update-fx"]
    create_plot(args.figure_name, args.project, data, horizons, args.mode, 'rewards', models2plot, args.uncertainty_value)
    create_plot(args.figure_name, args.project, data, horizons, args.mode, 'econ_rewards', models2plot, args.uncertainty_value)
    create_plot(args.figure_name, args.project, data, horizons, args.mode, 'penalties', models2plot, args.uncertainty_value)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=str, default='matching-thesis',
                        help='Name of the project')
    parser.add_argument('--mode', type=str, 
                        choices=['deterministic', 'stochastic'], required=True)
    parser.add_argument('--uncertainty_value', type=float,
                        help='Uncertainty scale value for stochastic mode')
    parser.add_argument('--figure_name', type=str, required=True)
    args = parser.parse_args()
    main(args)
