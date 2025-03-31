import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from common.utils import define_model
import plot_config

H = 24

# Load the SMPC results
smpc_results = np.load(f"data/uncertainty-growth/smpc-{H}H-20Ns-0.1-OL-predictions.npz")
ys_opt_all = smpc_results['ys_opt_all']  # Optimized output trajectories
Ns, ny, Np, N = ys_opt_all.shape
x0= [0.0035, 1.e-3, 15, 0.008]
dt = 1800
x_min = [0.002, 0, 5, 0]
x_max= [0.6, 0.004, 40, 0.051]
F, g = define_model(dt, x_min, x_max)

x_sim = np.zeros((Ns, 4, Np))
y_sim = np.zeros((Ns, 4, Np))

for i in range (Ns):
    x_sim[i, :, 0]=  x0
    y_sim[i, :, 0] = g(x0).toarray().ravel()

us = smpc_results["us_opt"][:, :, 0]
ds = smpc_results["d"]
p_samples_all = smpc_results["p_samples_all"]

for i in range(Ns):
    p_samples = p_samples_all[i, :, :, 0]
    for ll in range(Np-1):
        pk = p_samples[:, ll]
        x_sim[i,:, ll+1] = F(x_sim[i,:, ll], us[:, ll], ds[:, ll], pk).toarray().ravel()
        y_sim[i,:, ll+1] = g(x_sim[i,:,ll+1]).toarray().ravel()

# Extract the data

# Get dimensions
# N = 481
print(f"Ns: {Ns}, ny: {ny}, Np: {Np}, N: {N}")


# # Select a few time steps to visualize (adjust as needed)
# # For example, evenly spaced time steps or specific ones of interest
num_timesteps_to_show = min(5, N)
selected_timesteps = np.linspace(0, N-1, num_timesteps_to_show, dtype=int)


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
        trajectory = ys_opt_all[scenario, var_idx, :, selected_time]
        time_points = np.arange(selected_time, selected_time + Np)
        ax.step(time_points, trajectory, '-', alpha=0.3, color='C0')

    # Plot the mean trajectory
    mean_trajectory = np.mean(ys_opt_all[:, var_idx, :, selected_time], axis=0)
    # ax.step(time_points, mean_trajectory, '-', color="C0", linewidth=2, label='Mean predicted')
    for i in range(Ns):
        ax.step(time_points, y_sim[i, var_idx,:], alpha=0.3, color="C3", linewidth=2)    

    # ax.set_title(f'Output Variable {var_idx+1} - Detailed View at t={selected_time}')
    ax.set_xlabel('Hours')
    ax.set_ylabel(y_labels[var_idx])

    # Only add the legend for the first scenario to avoid duplicates
    if var_idx == 0:
        # Add custom legend entries
        ax.plot([], [], '-', color='C0', linewidth=2, alpha=0.3, label='Open-Loop')
        ax.plot([], [], '-', color='C3', linewidth=2, label='Simulated')
        ax.legend()

    ax.grid(True)


fig_detail.tight_layout()
fig_detail.savefig(f"figures/SMPC-open-loop-{H}H-20Ns-vs-sim")
# plt.show()

# Create a figure to show the control trajectory
fig, axes  = plt.subplots(1, 3, figsize=(5 * 3, 6))

# gs_control = GridSpec(2, 1, figure=fig_control, height_ratios=[2, 1])

# Extract control inputs
us_opt = smpc_results['us_opt']  # Optimized control inputs
u = smpc_results['u']  # Actual control inputs applied

for i, ax in enumerate(axes):

    control_trajectory = us_opt[i, :, selected_time]  # First control variable
    ax.step(time_points[:-1], control_trajectory, '-', color="C1", linewidth=2, label='Planned control')


plt.tight_layout()