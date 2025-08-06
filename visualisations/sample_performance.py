import os 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plot_config

def load_data(
        model_names, 
        mode, 
        project,
        smpc=True,
        uncertainty_value=None,
        H=4,
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
    # horizons = ]
    samples = ["2Ns", "4Ns", "6Ns", "8Ns", "10Ns", "12Ns", "14Ns", "16Ns", "18Ns", "20Ns"]
    data = {
        'smpc': {},
        'rl-zero-terminal-smpc': {},
        'rl': {}
    }
    uncertainty_suffix = f'-{uncertainty_value}' if uncertainty_value else ''

    for model in model_names:
        # Load RL data
        rl_path = f'data/{project}/{mode}/rl/{model}.csv'
        if os.path.exists(rl_path):
            data['rl'][model] = pd.read_csv(rl_path)

        # Load MPC and RL-MPC data for each horizon
        for s in samples:
            rlsmpc_terminal_path = f'data/{project}/{mode}/rlsmpc/{model}-samples-{H}H-{s}{uncertainty_suffix}.csv'
            smpc_path = f'data/{project}/{mode}/smpc/samples-{H}H-{s}{uncertainty_suffix}.csv'
            print(rlsmpc_terminal_path)
            if os.path.exists(rlsmpc_terminal_path):
                if s not in data['rl-zero-terminal-smpc']:
                    data['rl-zero-terminal-smpc'][s] = {}
                data['rl-zero-terminal-smpc'][s] = pd.read_csv(rlsmpc_terminal_path)

            if smpc:
                if os.path.exists(smpc_path):
                    if s not in data['smpc']:
                        data['smpc'][s] = {}
                    data['smpc'][s]= pd.read_csv(smpc_path)
    samples = [s.replace("Ns", "") for s in samples]
    return data, samples


def normalize_rewards(reward_array):
    """
    Normalize rewards by dividing each column by the corresponding row's last column value
    
    Parameters:
        reward_array (numpy.ndarray): Array of rewards with shape (n_samples, n_horizons)
    
    Returns:
        numpy.ndarray: Normalized rewards array
    """
    normalized = np.zeros_like(reward_array)
    base = reward_array[-1]
    normalized = reward_array / base
    return normalized

# def extract_smpc_metrics(ns_SMPC):
#     rewards_list = []
#     solver_times_list = []
#     for s in sorted(ns_SMPC.keys()):
#         sample_rewards = []
#         sample_times = []
#         for h in sorted(ns_SMPC[s].keys()):
#             reward_sum = ns_SMPC[s][h]['rewards'].sum().values()
#             exec_time_mean = ns_SMPC[s][h]['solver_times'].mean()
#             sample_rewards.append(reward_sum)
#             sample_times.append(exec_time_mean)
#         rewards_list.append(sample_rewards)
#         solver_times_list.append(sample_times)
#     return np.array(rewards_list), np.array(solver_times_list)

def extract_smpc_metrics(data):
    rewards_list = []
    solver_times_list = []
    for s in data.keys():
        # sample_rewards = []
        # sample_times = []
        reward_sum = data[s].groupby("run")['rewards'].sum().values[:5]
        exec_time_mean = data[s].groupby("run")['solver_times'].mean().values[:5]
        rewards_list.append(reward_sum)
        solver_times_list.append(exec_time_mean)
    print(rewards_list), print(solver_times_list)
    return np.array(rewards_list), np.array(solver_times_list)

def main():
    H = 4
    data, samples = load_data(
        ["brisk-resonance-24"],
        "stochastic",
        "samples", 
        smpc=True, 
        uncertainty_value=0.1,
        H=H
    )
    WIDTH = 60 * 0.0393700787
    HEIGHT = WIDTH * 0.75

    smpc_rewards, smpc_times = extract_smpc_metrics(data['smpc'])
    rlsmpc_rewards, rlsmpc_times = extract_smpc_metrics(data['rl-zero-terminal-smpc'])
    smpc_rewards_mean = smpc_rewards.mean(axis=1)
    rlsmpc_rewards_mean = rlsmpc_rewards.mean(axis=1)

    norm_smpc_rewards = normalize_rewards(smpc_rewards_mean)    
    norm_rlsmpc_rewards = normalize_rewards(rlsmpc_rewards_mean)    

    # Plot the normalized rewards
    plt.figure(figsize=(WIDTH, HEIGHT), dpi=300)
    n2plot = len(norm_rlsmpc_rewards)
    plt.plot(samples[:n2plot], norm_rlsmpc_rewards[:n2plot], '-o', label='RL-SMPC', color="C3", alpha=0.8)
    n2plot = len(norm_smpc_rewards)
    plt.plot(samples[:n2plot], norm_smpc_rewards[:n2plot], '-o', label='SMPC', color="C0", alpha=0.8)



    # plt.colorbar(im, label='Normalized Reward')
    # plt.yticks([2, 4, 8])
    # plt.xticks([2, 4, 6])
    plt.tight_layout()
    plt.legend()
    plt.xlabel('Samples (S)')
    plt.ylabel('Norm cumulative reward')
    plt.savefig(f'figures/samples/norm_reward-H{H}.png', dpi=180, bbox_inches='tight')
    plt.show()

    # Plot the normalized rewards
    plt.figure(figsize=(WIDTH, HEIGHT), dpi=300)
    n2plot = len(norm_rlsmpc_rewards)
    plt.plot(samples[:n2plot], rlsmpc_rewards_mean[:n2plot], '-o', label='RL-SMPC', color="C3", alpha=0.8)
    n2plot = len(norm_smpc_rewards)
    plt.plot(samples[:n2plot], smpc_rewards_mean[:n2plot], '-o', label='SMPC', color="C0", alpha=0.8)

    

    # plt.colorbar(im, label='Normalized Reward')
    # plt.yticks([2, 4, 8])
    # plt.xticks([2, 4, 6])
    plt.tight_layout()
    plt.legend()
    plt.xlabel('Samples ($S$)')
    plt.ylabel('Cumulative reward')
    plt.savefig(f'figures/samples/reward-H{H}.png', dpi=180, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    main()
