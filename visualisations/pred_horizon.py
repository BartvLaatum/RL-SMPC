import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
import cmcrameri.cm as cmc


### Latex font in plots
# Ensure text is converted to paths
plt.rcParams["svg.fonttype"] = "path"  # Converts text to paths
plt.rcParams["text.usetex"] = False    # Use mathtext instead of full LaTeX
plt.rcParams["mathtext.fontset"] = "dejavuserif"  # Use a more Illustrator-friendly font
plt.rcParams["font.family"] = "serif"

plt.rcParams["font.size"] = 8  # General font size
plt.rcParams["axes.labelsize"] = 10  # Axis label size
plt.rcParams["xtick.labelsize"] = 8  # Tick labels
plt.rcParams["ytick.labelsize"] = 8
plt.rcParams["legend.fontsize"] = 8
plt.rcParams["font.family"] = "serif"  # Use a journal-friendly font

plt.rcParams["text.usetex"] = False
plt.rcParams["axes.grid"] = False
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

# plt.rcParams["text.usetex"] = True
# plt.rcParams["mathtext.fontset"] = "cm"

plt.rcParams["axes.linewidth"] = 1.5  # Axis border thickness
plt.rcParams["lines.linewidth"] = 1.5  # Line thickness
plt.rcParams["grid.linewidth"] = 0.5
plt.rcParams["xtick.major.width"] = 1
plt.rcParams["ytick.major.width"] = 1

plt.rc("axes", unicode_minus=False)

# Load data for different prediction horizons
horizons = ['1H', '2H', '3H', '4H', '5H', '6H']
mpc_data = {}
mpc_old_data = {}
rlmpc_data = {}

rl_data = pd.read_csv('data/matching-thesis/rl/cosmic-bird-70.csv')

for h in horizons:
    mpc_path = f'data/matching-thesis/mpc/mpc-rh80-dt1800-{h}.csv'
    mpc_path2 = f'data/matching-thesis/mpc/mpc-rh75-dt900-{h}.csv'
    rlmpc_path = f'data/matching-thesis/rlmpc/rl_mpc-40D-{h}.csv'

    if os.path.exists(mpc_path):
        mpc_data[h] = pd.read_csv(mpc_path)
    if os.path.exists(mpc_path2):
        mpc_old_data[h] = pd.read_csv(mpc_path2)
    if os.path.exists(rlmpc_path):
        rlmpc_data[h] = pd.read_csv(rlmpc_path)

WIDTH = 87.5 * 0.03937
HEIGTH = WIDTH*0.75
# Create figure
fig, ax = plt.subplots(figsize=(WIDTH, HEIGTH), dpi=300)

# Plot cumulative rewards for each horizon
colors = cmc.batlow(np.linspace(0, 1, len(horizons)))
# Get final cumulative rewards for each horizon
mpc_final_rewards = []
rlmpc_final_rewards = []
mpc_old_final_rewards = []
horizon_nums = []

for h in horizons:
    horizon_num = int(h[0]) # Extract numeric value from horizon string
    horizon_nums.append(horizon_num)

    if h in mpc_data:
        cum_rewards = np.cumsum(mpc_data[h]['rewards'])
        mpc_final_rewards.append(cum_rewards.iloc[-1])
    
    if h in rlmpc_data:
        cum_rewards = np.cumsum(rlmpc_data[h]['rewards']) 
        rlmpc_final_rewards.append(cum_rewards.iloc[-1])

    if h in mpc_old_data:
        cum_rewards = np.cumsum(mpc_old_data[h]['rewards'])
        mpc_old_final_rewards.append(cum_rewards.iloc[-1])

cum_rewards = np.cumsum(rl_data['rewards'])
rl_final_reward = cum_rewards.iloc[-1]

# Plot final values vs prediction horizon
ax.plot(horizon_nums, mpc_final_rewards, 'o-', label='MPC', color=colors[0])
ax.plot(horizon_nums[:], mpc_old_final_rewards[:], 'o-', label='MPC dt=900(s) RH=75', color=colors[1])
# ax.plot(horizon_nums[:6], rlmpc_final_rewards[:6], 'o-', label='RL-MPC', color=colors[-1])
# ax.hlines(rl_final_reward, 1, 6, label='RL', color='grey', linestyle='--')

ax.set_xlabel('Prediction Horizon (H)')
ax.set_ylabel('Cumulative Reward')
ax.legend()

plt.tight_layout()
plt.savefig('figures/matching-thesis/mpc-dt-rh.png', bbox_inches='tight', dpi=300)
plt.show()
