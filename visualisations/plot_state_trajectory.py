import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.animation as animation

import plot_config
from rl_smpc_performance import load_data

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.lines import Line2D
import numpy as np


def plot_states(
    fig,
    axes,
    row,
    df,
    nplots,
    var,
    labels,
    ylabels,
    linestyle,
    color,
    bounds,
    y_lims,
    start_day,
    n_days
) -> None:
    """
    Plot state, input, or disturbance trajectories with mean and standard deviation
    over a specified time window.

    This function visualizes the mean and standard deviation of variables (states, inputs, or disturbances)
    from a grouped DataFrame, for a window from `start_day` to `start_day + n_days`.
    It clears and updates the relevant axes in the provided axes array.

    Args:
        fig (matplotlib.figure.Figure): The figure object to which the axes belong.
        axes (np.ndarray of matplotlib.axes.Axes): 2D array of axes to plot on (e.g., from plt.subplots).
        row (int): Row index in the axes array to plot on.
        df (pandas.core.groupby.DataFrameGroupBy): Grouped DataFrame (e.g., grouped by "time") containing the data to plot.
        nplots (int): Number of variables to plot (number of subplots in the row).
        var (str): Variable type: "y" for states, "u" for inputs, "d" for disturbances.
        labels (list of str): List of labels for the legend (typically algorithm names).
        ylabels (list of str): List of y-axis labels for each subplot.
        linestyle (str): Line style for the plot (e.g., '-', '--').
        color (str): Color for the plot lines and fill.
        bounds (list of tuple or None): List of (lower, upper) bounds for each variable, or None.
        y_lims (list of tuple or None): List of y-axis limits for each variable, or None.
        start_day (float): Start of the time window (in days).
        n_days (float): Length of the time window (in days).
    """
    end_day = start_day + n_days

    # Compute mean and std over the grouped DataFrame, and filter by time window
    df_mean = df.mean().reset_index()
    mask_mean = (df_mean["time"] >= start_day) & (df_mean["time"] < end_day)
    df_mean = df_mean[mask_mean]
    df_std = df.std().reset_index()
    mask_std = (df_std["time"] >= start_day) & (df_std["time"] < end_day)
    df_std = df_std[mask_std]

    time = df_mean["time"].values

    for i in range(nplots):
        # Select the correct axis: for inputs, skip the first column (usually reserved for disturbance)
        if var == "u":
            ax = axes[row][i + 1]
        else:
            ax = axes[row][i]

        if var == "d" and i == 0:
            # Special handling for the first disturbance: plot global radiation
            axes[1, 0].step(time, df_mean[f'{var}_{i}'].values, color=color)
            axes[1, 0].set_ylabel("Global Radiation (W/m$^2$)")
            axes[1, 0].set_ylim(0, 600)
        else:
            # Extract mean and std for the variable
            mean = df_mean[f'{var}_{i}'].values
            std = df_std[f'{var}_{i}'].values

            # Plot mean trajectory
            ax.step(
                time, mean, label=labels[0],
                linestyle=linestyle, color=color
            )
            # Fill between mean ± std
            ax.fill_between(
                time, mean - std, mean + std,
                alpha=0.5, color=color
            )

        # Set major tick locators for clarity
        ax.yaxis.set_major_locator(plt.LinearLocator(3))
        ax.xaxis.set_major_locator(plt.LinearLocator(3))

        # Draw bounds if provided
        if bounds[i] is not None:
            ax.hlines(
                bounds[i], time[0], time[-1],
                linestyle='--', color='black', alpha=0.5
            )
            ax.set_yticks(np.linspace(bounds[i][0], bounds[i][1], 3))

        # Set y-axis limits if provided
        if y_lims is not None:
            ax.set_ylim(y_lims[i])

        # Set y-axis label and x-axis limits
        ax.set_ylabel(ylabels[i])
        ax.set_xlim(start_day, end_day)

    # For the first state variable, set y-limits tightly around the data
    axes[0, 0].set_ylim(
        df_mean["y_0"].min() * 0.95,
        df_mean["y_0"].max() * 1.05
    )


def animated_plot():
    # --- load and group your data exactly as before ---
    model_name = "brisk-resonance-24"
    H = "1H"
    data, horizons = load_data(
        [model_name],
        mode="stochastic",
        project="SMPC",
        Ns=[],
        smpc=True,
        zero_order=True,
        terminal=True,
        first_order=False,
        uncertainty_value=0.1
    )

    mpc1h    = data['smpc'][H].reset_index().groupby("time")
    rlmpc1h  = data['rl-zero-terminal-smpc'][H][model_name].reset_index().groupby("time")
    rldata   = data["rl"][model_name].reset_index().groupby("time")
    # figure setup
    fig, axes = plt.subplots(
        3, 4, sharex=True,
        figsize=(12, 6), dpi=300,        
    )

    # static legend handles for algorithms
    color_handles = [
        Line2D([0], [0], color="C0", lw=2),
        Line2D([0], [0], color="C3", lw=2),
        Line2D([0], [0], color="C7", lw=2)
    ]

    fig.legend(color_handles,
               ["SMPC", "RL-SMPC"],
               title="Algorithm",
               loc="upper center")

    # prepare bounds & labels
    state_ylabels = [
        r"Lettuce DW (g/m$^2$)", r"CO_$2$ concentration (ppm)",
        r"Temperature ($^\circ$C)",    r"RH (%)"
    ]
    state_bounds = [None, (500, 1600), (10, 20), (0, 80)]
    state_lims = [
        None, None, None, (60, 100)
    ]

    input_ylabels = [
        r"CO$_2$-injection (mg/m$^2$/s)", r"Ventilation (m$^3$/m$^2$/s)",
        r"Heating (W/m$^2$)"
    ]
    input_bounds = [(0, 1.2), (0, 7.5), (0, 150)]

    dist_ylabels = [
        r"d$_{iGlob}$ (W/m$^2$)", r"d$_{CO_2}$ (ppm)",
        r"d$_{T}$ ($^\circ$C)",    r"d$_{RH}$ (%)"
    ]
    dist_bounds = [None, None, None, None]
    dist_lims = [
        (0, 600), (300, 600), (0, 21), (60, 100)
    ]

    # animation parameters
    window = 2.0   # days in each frame
    step   = 0.1   # days to shift per frame
    t_max  = 40.0   # total days of data
    n_frames = int((t_max - window)/step) + 1

    def animate(frame):
        start_day = frame * step
        # clear every axis once per frame
        for ax in axes.flatten():
            ax.clear()
            ax.xaxis.labelpad = 4
            ax.yaxis.labelpad = 4
        # row 0: states

        plot_states(fig, axes, 0, mpc1h,   4, "y", ["SMPC"],
                    state_ylabels, "-", "C0", state_bounds, state_lims,
                    start_day, window)
        y_lim_bottom = axes[0,0].get_ylim()

        plot_states(fig, axes, 0, rlmpc1h, 4, "y", ["RL-SMPC"],
                    state_ylabels, "-", "C3", state_bounds, state_lims,
                    start_day, window)
        y_lim_top = axes[0,0].get_ylim()

        # row 1: inputs
        plot_states(fig, axes, 1, mpc1h,   3, "u", ["SMPC"],
                    input_ylabels, "-", "C0", input_bounds, None,
                    start_day, window)
        plot_states(fig, axes, 1, rlmpc1h, 3, "u", ["RL-SMPC"],
                    input_ylabels, "-", "C3", input_bounds, None,
                    start_day, window)
        # row 2: disturbances
        plot_states(fig, axes, 2, mpc1h,   4, "d", [""],
                    dist_ylabels, "-", "C7", dist_bounds, dist_lims,
                    start_day, window)

        # hide the unused subplot in row 1, col 3
        axes[1, 3].axis('off')

        axes[0,0].set_ylim(y_lim_bottom[0], y_lim_top[1])
        # Set x-tick labels for bottom row
        for ax in axes[-1]:
            ax.set_xticks(np.linspace(start_day, start_day + window, 5))
            ax.set_xlabel("Time (days)")

        return axes.flatten()

    ani = animation.FuncAnimation(
        fig, animate,
        frames=n_frames,
        interval=10,   # milliseconds between frames
        blit=False
    )

    ani.save(
        f"closed_loop_trajectories-{H}.mp4",
        writer="ffmpeg",
        fps=5,          # adjust as needed
        dpi=300,        # match your figure’s dpi
        bitrate=2000,   # optional: tweak output quality/size
    )

def add_night_shading(axes, day_intervals, color='#DDEAFD', alpha=.25):
    """
    Shade night-time regions on one or more axes that span multiple days.

    Args:
        axes (list[matplotlib.axes.Axes] or matplotlib.axes.Axes): 
            One axis or an iterable of axes that share the same x-limits.
        day_intervals (iterable of tuple of float): 
            An ordered iterable of (sunrise, sunset) x-positions per day.
            Example: [(6, 18), (30, 42), …]  Make sure they are sorted.
        color (str, optional): Fill colour for night (default: light blue).
        alpha (float, optional): Transparency of the night rectangles (default: 0.25).

    Returns:
        list[matplotlib.axes.Axes]: The axes that were modified.
    """
    # Allow a single axis to be passed without wrapping it in a list
    try:
        axes = list(axes)
    except TypeError:
        axes = [axes]
    # Pre-compute the complementary *night* spans ---------------------------
    # Build a list of (start, end) tuples that represent darkness
    night_spans = []
    if not day_intervals:
        raise ValueError("day_intervals is empty")

    # Safe copy & sort
    day_intervals = sorted(day_intervals)

    # 1) night before the first sunrise
    night_spans.append(("xlim_left", day_intervals[0][0]))

    # 2) nights between each day
    for (_, sunset_prev), (sunrise_next, _) in zip(day_intervals[:-1], day_intervals[1:]):
        night_spans.append((sunset_prev, sunrise_next))

    # 3) night after the last sunset
    night_spans.append((day_intervals[-1][1], "xlim_right"))

    # ----------------------------------------------------------------------
    for ax in axes:
        y_min, y_max = ax.get_ylim()
        x_min, x_max = ax.get_xlim()

        for start, end in night_spans:
            # Replace the sentinel labels with the actual axis limits
            if start == "xlim_left":
                start = x_min
            if end == "xlim_right":
                end = x_max

            # Skip zero-width spans (can happen if daytime stretches to edge)
            if end <= start:
                continue

            ax.add_patch(plt.Rectangle(
                (start, y_min),
                width=end - start,
                height=y_max - y_min,
                facecolor=color,
                alpha=alpha,
                zorder=-1
            ))

    return axes


def static_plot():
    """
    Creates a static plot comparing SMPC and RL-SMPC closed-loop trajectories for greenhouse control.
    
    This function generates visualization showing:
    - State trajectories (lettuce dry weight, CO2 concentration, temperature, humidity)
    - Control input trajectories (CO2 injection, ventilation, heating)
    - Disturbance trajectories (solar radiation, outdoor CO2, temperature, humidity)

    The plot includes:
    - 2x4 subplot layout with shared x-axis
    - Night/day shading based on solar radiation data
    - Custom legend and axis formatting
    - Both PNG and SVG output formats
    
    The function loads data for a specific model and prediction horizon, then plots
    the mean trajectories for SMPC and RL-SMPC controllers, with appropriate
    styling and annotations for publication-quality figures.
    
    Returns: None
    Displays the plot and saves it to disk.
    """
    # --- Load and group data for the specified model and horizon ---
    model_name = "brisk-resonance-24"
    H = "6H"
    data, horizons = load_data(
        [model_name],
        mode="stochastic",
        project="SMPC",
        Ns=[],
        smpc=True,
        zero_order=True,
        terminal=True,
        first_order=False,
        uncertainty_value=0.1
    )

    # Group data by time to plot the mean and std over runs
    mpc1h    = data['smpc'][H].reset_index().groupby("time")
    rlmpc1h  = data['rl-zero-terminal-smpc'][H][model_name].reset_index().groupby("time")
    rldata   = data["rl"][model_name].reset_index().groupby("time")
    
    # Set up figure dimensions and create subplots
    WIDTH = 173.8 * 0.0393700787
    HEIGHT = WIDTH * 0.4
    fig, axes = plt.subplots(
        2, 4, sharex=True,
        figsize=(WIDTH, HEIGHT), dpi=300,
        # constrained_layout=True
    )

    # Create static legend handles for algorithms
    color_handles = [
        Line2D([0], [0], color="C0", lw=2),
        Line2D([0], [0], color="C3", lw=2),
        Line2D([0], [0], color="C7", lw=2)
    ]
    fig.legend(color_handles,
               ["SMPC", "RL-SMPC", "Outdoor"],
            #    title="Controller",
               loc="upper center",
               ncol=1)

    # Define plot configuration for states, inputs, and disturbances
    # State variable labels and bounds
    state_ylabels = [
        r"Lettuce DW (g/m$^2$)", r"CO$_2$ concentration (ppm)",
        r"Temperature ($^\circ$C)",    r"Relative Humidity (%)"
    ]
    state_bounds = [None, (500, 1600), (10, 20), (0, 80)]
    state_lims = [
        (0.1, 0.14), None, (9, 21), (60, 100)
        ]

    # Control input labels and bounds
    input_ylabels = [
        r"CO$_2$-injection (mg/m$^2$/s)", r"Ventilation (m$^3$/m$^2$/s)",
        r"Heating (W/m$^2$)"
    ]
    input_bounds = [(0, 1.2), (0, 7.5), (0, 150)]

    # Disturbance labels and bounds
    dist_ylabels = [
        r"$d_{iGlob}$ (W/m$^2$)", r"$d_{CO_2}$ (ppm)",
        r"$d_{T}$ ($^\circ$C)",    r"$d_{RH}$ (%)"
    ]
    dist_bounds = [None, None, None, None]
    dist_lims = [
        (0, 600), (350, 550), (0, 21), (60, 100)
        ]

    # Set animation parameters (used for time window selection)
    window = 2.0   # days in each frame
    start_day = 13

    # --- Plot state trajectories (row 0) ---
    # Plot SMPC state trajectories
    plot_states(fig, axes, 0, mpc1h,   4, "y", ["SMPC"],
                state_ylabels, "-", "C0", state_bounds, state_lims,
                start_day, window)
    y_lim_bottom = axes[0,0].get_ylim()

    # Plot RL-SMPC state trajectories
    plot_states(fig, axes, 0, rlmpc1h, 4, "y", ["RL-SMPC"],
                state_ylabels, "-", "C3", state_bounds, state_lims,
                start_day, window)
    y_lim_top = axes[0,0].get_ylim()

    # RL state trajectories (not used in this version)
    # plot_states(fig, axes, 0, rldata,  4, "y", ["RL"],
    #             state_ylabels, "-", "C7", state_bounds, state_lims,
    #             start_day, window)

    # --- Plot control input trajectories (row 1) ---
    # Plot SMPC control inputs
    plot_states(fig, axes, 1, mpc1h,   3, "u", ["SMPC"],
                input_ylabels, "-", "C0", input_bounds, None,
                start_day, window)

    # Plot RL-SMPC control inputs
    plot_states(fig, axes, 1, rlmpc1h, 3, "u", ["RL-SMPC"],
                input_ylabels, "-", "C3", input_bounds, None,
                start_day, window)

    # RL control inputs
    # plot_states(fig, axes, 1, rldata,  3, "u", ["RL"],
    #             input_ylabels, "-", "C7", input_bounds, None,
    #             start_day, window)

    # --- Plot disturbance trajectories (overlaid on state plots) ---
    plot_states(fig, axes, 0, mpc1h,   4, "d", [""],
                state_ylabels, "-", "C7", dist_bounds, None,
                start_day, window)

    # hide unused subplot
    # axes[1, 3].axis('off')

    # --- Customize axis limits and tick marks for better visualization ---
    axes[0, 0].set_ylim(y_lim_bottom[0], y_lim_top[1])
    axes[0, 0].set_yticks(np.linspace(0.039, 0.065, 3))
    axes[0, -1].set_yticks(np.linspace(60, 100, 3))
    axes[0, 1].set_yticks(np.linspace(400, 1600, 3))
    axes[0, 2].set_ylim(0, 21)
    axes[0, 2].set_yticks(np.linspace(0, 20, 3))

    # --- Calculate day/night periods for shading ---
    # Get mean data for the first day
    df_mean = mpc1h.mean().reset_index()
    mask_mean = (df_mean["time"] >= start_day) & (df_mean["time"] < start_day + 1)
    df_mean = df_mean[mask_mean]
    day_period = df_mean["d_0"] > 0.1  # Day periods have solar radiation > 0.1

    # Get mean data for the second day
    df_mean2 = mpc1h.mean().reset_index()
    mask_mean2 = (df_mean2["time"] >= start_day+1) & (df_mean2["time"] < start_day + 2)
    df_mean2 = df_mean2[mask_mean2]
    day_period2 = df_mean2["d_0"] > 0.1

    # Extract time intervals for day periods
    interval1 = df_mean["time"][day_period].values
    interval2 = df_mean2["time"][day_period2].values

    # Create day intervals for shading (sunrise to sunset)
    day_intervals = [
        (interval1[0], interval1[-1]),
        (interval2[0], interval2[-1])
    ]

    # Add night shading to both rows of subplots
    add_night_shading(axes[0], day_intervals, color='#DDEAFD', alpha=0.8)
    add_night_shading(axes[1], day_intervals, color='#DDEAFD', alpha=0.8)

    # --- Final formatting and saving ---
    # Add x-axis labels to bottom row only
    for ax in axes[-1]:
        ax.set_xlabel("Time (Days)")
    plt.tight_layout()
    
    # Save plot in both PNG and SVG formats
    fig.savefig(f'closed_loop_trajectories-daylight-{H}.png', format='png', dpi=300)
    fig.savefig(f'closed_loop_trajectories-daylight-{H}.svg', format='svg', dpi=300)
    plt.show()

if __name__ == "__main__":
    static_plot()
    # animated_plot()

