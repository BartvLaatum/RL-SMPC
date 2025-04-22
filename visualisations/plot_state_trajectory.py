from visualisations.rl_smpc_performance import load_data
import matplotlib.pyplot as plt
import plot_config
from matplotlib.lines import Line2D
import matplotlib.animation as animation


def plot_states(fig, axes, row, df, nplots, var, labels, ylabels, linestyle, color, bounds=None):
    """
    Plots the states of multiple dataframes with specified variables and labels.

    Args:
    data (dictionary with DataFrames): List of dataframes containing the data to plot.
    nplots (int): Number of subplots to create.
    var (str): Base name of the variable to plot.
    labels (list of str): List of labels for each dataframe.
    ylabels (list of str): List of y-axis labels for each subplot.
    bounds (list of tuple, optional): List of bounds for each subplot. Each tuple contains (lower_bound, upper_bound). Default is None.

    Returns:
    fig (matplotlib.figure.Figure): The created figure.
    ax (numpy.ndarray of matplotlib.axes._subplots.AxesSubplot): Array of subplot axes.
    """
    WIDTH = 87.5 * 0.03937
    HEIGHT = WIDTH * 0.25

    df_mean = df.mean().reset_index()
    df_mean = df_mean[25 < df_mean["time"]]
    df_mean = df_mean[df_mean["time"] < 28]
    df_std = df.std().reset_index()
    df_std = df_std[25 < df_std["time"]]
    df_std = df_std[df_std["time"] < 28]
    time = df_mean["time"].values


    for i in range(nplots):
        mean = df_mean['{}_{}'.format(var, i)].values
        std = df_std['{}_{}'.format(var, i)].values
        axes[row][i].step(time, mean, label=labels[0], linestyle=linestyle, color=color)
        # axes[row][i].fill_between(time, mean - std, mean + std, alpha=0.5, color=color)
        # axes[row][i].step(df['time'], df['{}_{}'.format(var, i)].mean(), label=labels[0], linestyle=linestyles[0], where='post')
        
        # axes[row][i].fill_between(df['time'], df['{}_{}'.format(var, i)].std(), df['{}_{}'.format(var, i)].std(), alpha=0.2)
        
        if bounds[i] is not None:
            axes[row][i].hlines(bounds[i], time[0], time[-1], linestyle='--', color='grey')
        axes[row][i].set_ylabel(ylabels[i])
    axes[0][0].legend(loc='upper left', bbox_to_anchor=(0, 0.8))
    fig.tight_layout()
    return fig, axes

def static_plot():
    model_name = "restful-pyramid-7"
    H = "1H"
    data, horizons = load_data([model_name], mode="stochastic", project="SMPC", Ns=[], smpc=True, zero_order=True, terminal=True, first_order=False, uncertainty_value=0.1)
    mpc1h = data['mpc'][H]
    rlmpc1h = data['rl-zero-terminal-smpc'][H][model_name]
    mpc6h = data['mpc']["6H"]
    rlmpc6h = data['rl-zero-terminal-smpc']["6H"][model_name]


    mpc1h = mpc1h.reset_index().groupby("time")
    rlmpc1h = rlmpc1h.reset_index().groupby("time")
    mpc6h = mpc6h.reset_index().groupby("time")
    rlmpc6h = rlmpc6h.reset_index().groupby("time")

    ylabels = [r"y$_{DW}$ (g/m$^2$)", r"y$_{CO_2}$ (ppm)", r"y$_{T}$ ($^\circ$C)", r"y$_{RH}$ (%)"]
    bounds = [None, (500, 1600), (10, 20), (0, 80)]

    fig, axes = plt.subplots(2, 4, sharex=True, figsize=(16, 4), dpi=180)
    row = 0
    fig, axes = plot_states(fig, axes, row,  mpc1h, 4, "y", ["MPC"], ylabels, "-", color="C0", bounds=bounds) 
    fig, axes = plot_states(fig, axes, row, rlmpc1h, 4, "y", ["RL-MPC"], ylabels, "-", color="C3", bounds=bounds) 

    fig, axes = plot_states(fig, axes, row, mpc6h, 4, "y", ["MPC"], ylabels, "--", color="C0", bounds=bounds) 
    fig, axes = plot_states(fig, axes, row, rlmpc6h, 4, "y", ["RL-MPC"], ylabels, "--", color="C3", bounds=bounds) 

    bounds = [None, None, None]
    ylabels = [r"u$_{CO_2}$ (mg/m$^2$/s)", r"u$_{vent}$ (m/s)", r"u$_{heat}$ (W/m$^2$)"]
    row=1
    fig, axes = plot_states(fig, axes, row,  mpc1h, 3, "u", ["MPC"], ylabels, "-", color="C0", bounds=bounds) 
    fig, axes = plot_states(fig, axes, row, rlmpc1h, 3, "u", ["RL-MPC"], ylabels, "-", color="C3", bounds=bounds) 

    fig, axes = plot_states(fig, axes, row, mpc6h, 3, "u", ["MPC"], ylabels, "--", color="C0", bounds=bounds) 
    fig, axes = plot_states(fig, axes, row, rlmpc6h, 3, "u", ["RL-MPC"], ylabels, "--", color="C3", bounds=bounds) 

    # Hide the last axes in the last row
    axes[-1, -1].axis('off')

    # Create custom legend handles for algorithm (color) and horizon (linestyle)
    color_handles = [
        Line2D([0], [0], color="C0", lw=2, linestyle="-"),
        Line2D([0], [0], color="C3", lw=2, linestyle="-")
    ]
    linestyle_handles = [
        Line2D([0], [0], color="grey", lw=2, linestyle="-"),
        Line2D([0], [0], color="grey", lw=2, linestyle="--")
    ]
    
    # Create the legend for algorithm type
    leg1 = axes[0][0].legend(color_handles, ["MPC", "RL-MPC"], title="Algorithm",
                        loc="upper left", bbox_to_anchor=(0, 1))
    # Create a second legend for horizon using line styles
    leg2 = axes[0][0].legend(linestyle_handles, ["1H", "6H"], title="Horizon",
                        loc="lower left", bbox_to_anchor=(0.6, 0.0))
    # Ensure both legends are displayed
    axes[0][0].add_artist(leg1)

    fig.savefig("figures/uncertainty-comparison/rl-mpc-trajectory.png", dpi=300, bbox_inches="tight")
    plt.show()

def animate_states(fig, ax, mpc_df, rlmpc_df, nplots, var, ylabels, bounds, colors=["C0", "C3"], linestyles=["-", "-"]):
    mpc_mean = mpc_df.mean().reset_index()
    rlmpc_mean = rlmpc_df.mean().reset_index()
    mpc_mean = mpc_mean[25 < mpc_mean["time"]]
    mpc_mean = mpc_mean[mpc_mean["time"] < 30]
    rlmpc_mean = rlmpc_mean[25< rlmpc_mean["time"]]
    rlmpc_mean = rlmpc_mean[rlmpc_mean["time"]<30]
    time = mpc_mean["time"].values

    lines = []
    for i in range(nplots):
        l1, = ax[i].plot([], [], label="MPC", linestyle=linestyles[0], color=colors[0])
        l2, = ax[i].plot([], [], label="RL-MPC", linestyle=linestyles[1], color=colors[1])
        lines.append((l1, l2))
        ax[i].set_ylabel(ylabels[i])
        if bounds[i] is not None:
            ax[i].hlines(bounds[i], time[0], time[-1], linestyle="--", color="grey")
            ax[i].set_ylim(bounds[i][0]*0.9, bounds[i][1]*1.1)

    ax[0].legend(loc="upper left", bbox_to_anchor=(0, 1))
    fig.tight_layout()

    def update(frame):
        x = time[:frame]
        for i in range(nplots):
            y1 = mpc_mean['{}_{}'.format(var, i)].values[:frame]
            y2 = rlmpc_mean['{}_{}'.format(var, i)].values[:frame]
            lines[i][0].set_data(x, y1)
            lines[i][1].set_data(x, y2)
            ax[i].set_xlim(time[0], time[-1])
        return [l for two_lines in lines for l in two_lines]

    ani = animation.FuncAnimation(fig, update, frames=len(time), interval=40, blit=True)
    return ani

if __name__ == "__main__":
    static_plot()

    #     Uncomment the following block to run the animation.
    # It will create a new figure with the same layout and animate the lines from both datasets.
    # fig_anim, ax_anim = plt.subplots(1, 4, sharex=True, figsize=(16, 4), dpi=180)
    # ani = animate_states(fig_anim, ax_anim, mpc1h, rlmpc1h, 4, "y", ylabels, bounds)
    # plt.show()

