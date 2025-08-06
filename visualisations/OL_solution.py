"""
This script loads and visualizes the open-loop solution of a specific control step
for both SMPC and RL-SMPC controllers in greenhouse climate control experiments.
It plots the predicted output trajectories, and control inputs for a selected time step.
The average of the simulated scenarios and its standard deviation are visualized.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from common.utils import transform_disturbances, co2dens2ppm, vaporDens2rh
import plot_config

# Set prediction horizon (in hours)
H = 3

data = {}
# Load the RL-SMPC open-loop prediction results
rlsmpc_results = np.load(f"data/SMPC/rl-samples-{H}H-20Ns-0.1-OL-predictions.npz")
# Store in data dictionary for easy access
# ys_opt_all = rlsmpc_results['ys_opt_all']  # Optimized output trajectories
data["rl-smpc"] = rlsmpc_results
# Load the SMPC open-loop prediction results
smpc_results = np.load(f"data/SMPC/smpc-{H}H-20Ns-0.1-OL-predictions.npz")
# ys_opt_all = smpc_results['ys_opt_all']  # Optimized output trajectories
data["smpc"] = smpc_results

# Print available keys in the SMPC results file
for key in data["smpc"].keys():
    print(key)

# Convert units of CO2 and humidity to PPM and RH 
d = transform_disturbances(smpc_results["d"])

# Convert CO2 and humidity to ppm and RH for RL-SMPC state samples
data["rl-smpc"]["xs_samples_all"][:, 1, :, :] = co2dens2ppm(data["rl-smpc"]["xs_samples_all"][:, 2, :, :], data["rl-smpc"]["xs_samples_all"][:, 1, :, :])
data["rl-smpc"]["xs_samples_all"][:, 3, :, :] = vaporDens2rh(data["rl-smpc"]["xs_samples_all"][:, 2, :, :], data["rl-smpc"]["xs_samples_all"][:, 3, :, :])

# Extract shape information for plotting
Ns, ny, Np, N = data["rl-smpc"]["ys_opt_all"].shape
print(f"Ns: {Ns}, ny: {ny}, Np: {Np}, N: {N}")

# Select a specific time step to visualize in detail
selected_time = int(13.3*24*2)  # Example: 13.3 days, 30-min intervals

# Set up figure for detailed output trajectory visualization
WIDTH = 173.8 * 0.03937
HEIGHT = WIDTH * 0.4
fig_detail = plt.figure(figsize=(WIDTH, HEIGHT), dpi=180)
gs_detail = GridSpec(2, ny, figure=fig_detail)
# State variable axis limits
state_lims = [
    (0, 70), (400, 800), (5, 15), (60, 100)
]
# Axis labels for outputs and disturbances
y_labels = [r"$y_{\mathrm{DW}}$ (kg/m$^2$)",r"$y_{\mathrm{CO_2}}$ (ppm)", r"$y_{\mathrm{T}}$  ($^\circ$C)", r"$y_{\mathrm{RH}}$  (%)"]
y_labels2 = [r"$d_{\mathrm{iGlob}}$ (W/m$^2$)",r"$d_{\mathrm{CO_2}}$ (ppm)", r"$d_{\mathrm{T}}$ ($^\circ$C)", r"$d_{\mathrm{RH}}$ (%)"]

# Plot output trajectories and statistics for each output variable
for var_idx in range(ny):
    ax = fig_detail.add_subplot(gs_detail[0, var_idx])
    ax2 = fig_detail.add_subplot(gs_detail[1, var_idx], sharex=ax)

    time_points = np.arange(selected_time, selected_time + Np)/2/24  # Convert to days
    # Plot all scenarios for this time step (optional)
    # for scenario in range(Ns):

    print(data["smpc"]["ys_opt_all"][:, var_idx, :, selected_time].shape)

    # Compute standard deviation and mean trajectories for SMPC and RL-SMPC
    smpc_std_trajectory = np.std(data["smpc"]["ys_opt_all"][:, var_idx, :, selected_time], axis=0)
    rlsmpc_std_trajectory = np.std(data["rl-smpc"]["ys_opt_all"][:, var_idx, :, selected_time], axis=0)
    smpc_mean_trajectory = np.mean(data["smpc"]["ys_opt_all"][:, var_idx, :, selected_time], axis=0)
    rlsmpc_mean_trajectory = np.mean(data["rl-smpc"]["ys_opt_all"][:, var_idx, :, selected_time], axis=0)

    # RL-SMPC state sample mean (with unit conversion for CO2 and RH)
    rlsamples_mean_trajectory = np.mean(data["rl-smpc"]["xs_samples_all"][:, var_idx, :, selected_time], axis=0)
    if var_idx == 1:
        rlsamples_mean_trajectory = co2dens2ppm(np.mean(data["rl-smpc"]["xs_samples_all"][:, 2, :, selected_time], axis=0), rlsamples_mean_trajectory)
    elif var_idx == 3:
        rlsamples_mean_trajectory = vaporDens2rh(np.mean(data["rl-smpc"]["xs_samples_all"][:, 2, :, selected_time], axis=0), rlsamples_mean_trajectory)        

    # Plot mean trajectories for SMPC, RL-SMPC, and RL rollout
    ax.step(time_points, smpc_mean_trajectory, '-', color='C0', linewidth=2, alpha=0.8,)
    ax.step(time_points, rlsmpc_mean_trajectory, '-', color='C3', linewidth=2, alpha=0.8,)
    ax.step(time_points, rlsamples_mean_trajectory, '-', color='#B3B3E1', linewidth=2, alpha=0.8,)

    # Highlight the last data point with an error bar (for RL rollout)
    dt = time_points[1] - time_points[0]  # time step size
    x_semi = dt/2
    y_range = rlsamples_mean_trajectory[-1]
    y_semi = 0.05 * y_range
    ax.errorbar(
        time_points[-1],
        rlsamples_mean_trajectory[-1],
        yerr=y_semi,
        fmt='.',
        color='#B3B3E1',
        ecolor='#B3B3E1',
        linewidth=1,
        elinewidth=2,
        capsize=4,
        alpha=0.8
    )

    # Fill between mean +- std for SMPC and RL-SMPC
    ax.fill_between(time_points, smpc_mean_trajectory-smpc_std_trajectory, smpc_mean_trajectory+smpc_std_trajectory, alpha=0.3, color='C0', step="pre")
    ax.fill_between(time_points, rlsmpc_mean_trajectory-rlsmpc_std_trajectory, rlsmpc_mean_trajectory+rlsmpc_std_trajectory,  alpha=0.3, color='C3', step="pre")

    # Set axis labels and limits
    ax.set_ylabel(y_labels[var_idx])
    ax2.set_ylabel(y_labels2[var_idx])
    ax2.set_xlabel('Time (days)')
    ax.set_ylim(state_lims[var_idx])
    ax2.set_ylim(state_lims[var_idx])
    if var_idx == 0:
        ax.set_ylim((0.042, 0.052))
        # Add custom legend entries
        ax.plot([], [], '-', color='C0', linewidth=2, alpha=0.8, label='SMPC')
        ax.plot([], [], '-', color='C3', linewidth=2, alpha=0.8, label='RL-SMPC')
        ax.plot([], [], '-', color='#B3B3E1', linewidth=2, alpha=0.8, label='RL-rollout')
        ax.legend()

    # Set tick locators for clarity
    ax.yaxis.set_major_locator(plt.LinearLocator(3))
    ax2.yaxis.set_major_locator(plt.LinearLocator(3))
    # Plot disturbance trajectory below each output
    ax2.plot(time_points, d[var_idx, selected_time:selected_time + Np], "-", color="C7", linewidth=2)

# Finalize and save the detailed output trajectory figure
fig_detail.tight_layout()
fig_detail.savefig(f"figures/SMPC-open-loop-{H}H-20Ns-vs-sim-rl-samples.png", bbox_inches="tight", format="png", dpi=300)
fig_detail.savefig(f"figures/SMPC-open-loop-{H}H-20Ns-vs-sim-rl-samples.svg", bbox_inches="tight", format="svg", dpi=300)

# Create a figure to show the control trajectory for each control input
fig, axes  = plt.subplots(1, 3, figsize=(5 * 3, 6))

# Plot control trajectories for each control input
for i, ax in enumerate(axes):
    smpc_us_opt = data["smpc"]['us_opt']  # Optimized control inputs
    rlsmpc_us_opt = data["rl-smpc"]['us_opt']  # Optimized control inputs
    # Extract control trajectories for the selected time step
    smpc_control_trajectory = smpc_us_opt[i, :, selected_time]  # SMPC
    rlsmpc_control_trajectory = rlsmpc_us_opt[i, :, selected_time]  # RL-SMPC
    # Plot control trajectories
    ax.step(time_points[:-1], smpc_control_trajectory, '-', color="C0", linewidth=2, label='SMPC')
    ax.step(time_points[:-1], rlsmpc_control_trajectory, '-', color="C3", linewidth=2, label='RL-SMPC')

plt.tight_layout()
plt.show()
