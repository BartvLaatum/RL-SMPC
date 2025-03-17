from performance_plots import load_data
import matplotlib.pyplot as plt


def plot_states(df, nplots, var, labels, ylabels, linestyles, bounds=None):
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
    fig, ax = plt.subplots(nplots, 1, sharex=True, figsize=(8, 12))

    for i in range(nplots):
        ax[i].step(df['time'], df['{}_{}'.format(var, i)], label=labels[0], linestyle=linestyles[0], where='post')
        ax[i].set_ylabel(ylabels[i])
        if bounds[i] is not None:
            ax[i].hlines(bounds[i], df['time'].iloc[0], df['time'].iloc[-1], linestyle='--', color='grey')
    ax[0].legend(loc='upper left', bbox_to_anchor=(0, 1))
    fig.tight_layout()
    fig.savefig("state_trajectory.png")
    return fig, ax


if __name__ == "__main__":
    data, horizons = load_data(["likely-frost-1"], "stochastic", "matching-salim", [], uncertainty_value=0.1)
    mpc6h = data['mpc']["6H"]
    run0_data = mpc6h[mpc6h['run'] == 0]
    ylabels = [r"DW (g/m$^2$)", r"CO$_2$ (ppm)", r"Temp ($^\circ$C)", r"RH (%)"]
    bounds = [None, (500, 1600), (10, 20), (0, 80)]
    plot_states(run0_data, 4, "y", ["Run 0"], ylabels, ["-"], bounds=bounds) 