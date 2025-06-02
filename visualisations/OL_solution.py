import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from common.utils import define_model
import plot_config

H = 3

data = {}

# Load the SMPC results
rlsmpc_results = np.load(f"data/SMPC/rl-smpc-{H}H-20Ns-0.1-OL-predictions.npz")
# ys_opt_all = rlsmpc_results['ys_opt_all']  # Optimized output trajectories
data["rl-smpc"] = rlsmpc_results
smpc_results = np.load(f"data/SMPC/smpc-{H}H-20Ns-0.1-OL-predictions.npz")
# ys_opt_all = smpc_results['ys_opt_all']  # Optimized output trajectories
data["smpc"] = smpc_results

Ns, ny, Np, N = data["rl-smpc"]["ys_opt_all"].shape
print(f"Ns: {Ns}, ny: {ny}, Np: {Np}, N: {N}")


# Optional: Create a more detailed view for a specific time step
# This can be useful to see all scenarios more clearly for a particular time
selected_time = 0  # Choose one time step to examine in detail

WIDTH = 200 * 0.03937
HEIGHT = WIDTH * 0.25
fig_detail = plt.figure(figsize=(WIDTH, HEIGHT), dpi=180)

gs_detail = GridSpec(1, ny, figure=fig_detail)

y_labels = [r"Lettuce DW (kg/m$^2$)",r"CO$_2$ (ppm)", r"Temperature ($^\circ$C)", "RH (%)"]

for var_idx in range(ny):
    ax = fig_detail.add_subplot(gs_detail[0,var_idx])

    # Plot all scenarios for this time step
    for scenario in range(Ns):
        time_points = np.arange(selected_time, selected_time + Np)/2

        smpc_trajectory = data["smpc"]["ys_opt_all"][scenario, var_idx, :, selected_time]
        rl_smpc_trajectory = data["rl-smpc"]["ys_opt_all"][scenario, var_idx, :, selected_time]
        ax.step(time_points, smpc_trajectory, '-', alpha=0.3, color='C0')
        ax.step(time_points, rl_smpc_trajectory, '-', alpha=0.3, color='C3')


    # Plot the mean trajectory
    mean_trajectory = np.mean(data["smpc"]["ys_opt_all"][:, var_idx, :, selected_time], axis=0)
    mean_trajectory = np.mean(data["rl-smpc"]["ys_opt_all"][:, var_idx, :, selected_time], axis=0)

    # ax.set_title(f'Output Variable {var_idx+1} - Detailed View at t={selected_time}')
    ax.set_xlabel('Hours')
    ax.set_ylabel(y_labels[var_idx])

    # Only add the legend for the first scenario to avoid duplicates
    if var_idx == 0:
        # Add custom legend entries
        ax.plot([], [], '-', color='C0', linewidth=2, alpha=0.8, label='SMPC')
        ax.plot([], [], '-', color='C3', linewidth=2, alpha=0.8, label='RL-SMPC')
        # ax.plot([], [], '-', color='C3', linewidth=2, label='Simulated')
        ax.legend()

    ax.grid(True)


fig_detail.tight_layout()
# fig_detail.savefig(f"figures/SMPC-open-loop-{H}H-20Ns-vs-sim")
plt.show()
# plt.show()

# Create a figure to show the control trajectory
fig, axes  = plt.subplots(1, 3, figsize=(5 * 3, 6))

# gs_control = GridSpec(2, 1, figure=fig_control, height_ratios=[2, 1])

# Extract control inputs

for i, ax in enumerate(axes):
    smpc_us_opt = data["smpc"]['us_opt']  # Optimized control inputs
    rlsmpc_us_opt = data["rl-smpc"]['us_opt']  # Optimized control inputs
    
    # u = smpc_results['u']  # Actual control inputs applied

    smpc_control_trajectory = smpc_us_opt[i, :, selected_time]  # First control variable
    rlsmpc_control_trajectory = smpc_us_opt[i, :, selected_time]  # First control variable
    ax.step(time_points[:-1], smpc_control_trajectory, '-', color="C0", linewidth=2, label='SMPC')
    ax.step(time_points[:-1], rlsmpc_control_trajectory, '-', color="C3", linewidth=2, label='RL-SMPC')

plt.tight_layout()
plt.show()
