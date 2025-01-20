from matplotlib import pyplot as plt
### Latex font in plots
plt.rcParams['font.serif'] = "cmr10"
plt.rcParams['font.family'] = "serif"
plt.rcParams['font.size'] = 24

plt.rcParams['legend.fontsize'] = 24
plt.rcParams['legend.loc'] = 'upper right'
plt.rcParams['axes.labelsize'] = 24
plt.rcParams['axes.formatter.use_mathtext'] = True
plt.rcParams['xtick.labelsize'] = 24
plt.rcParams['ytick.labelsize'] = 24
plt.rcParams['text.usetex'] = False
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams["axes.grid"] = False
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['axes.linewidth'] = 4   # Default for all spines
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
# plt.rcParams['text.usetex'] = True
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['xtick.major.size'] = 4  # Thicker major x-ticks
plt.rcParams['xtick.major.width'] = 2  # Thicker major x-
plt.rcParams['ytick.major.size'] = 4  
plt.rcParams['ytick.major.width'] = 2 
plt.rc('axes', unicode_minus=False)

def plot_states(dfs, nplots, var, labels, ylabels, linestyles, bounds=None):
    """
    Plots the states of multiple dataframes with specified variables and labels.

    Args:
    dfs (list of pd.DataFrame): List of dataframes containing the data to plot.
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

    for j, df in enumerate(dfs):
        for i in range(nplots):
            ax[i].step(df['time'], df['{}_{}'.format(var, i)], label=labels[j], linestyle=linestyles[j], where='post')
            ax[i].set_ylabel(ylabels[i])
            if bounds[i] is not None:
                ax[i].hlines(bounds[i], df['time'].iloc[0], df['time'].iloc[-1], linestyle='--', color='grey')
    ax[0].legend(loc='upper left', bbox_to_anchor=(0, 1))
    fig.tight_layout()
    return fig, ax


def plot_cumulative_rewards(dfs, labels, colors, ylabel, linestyles):
    """
    Plots the cumulative rewards of multiple dataframes.

    Arguments:
    dfs (list of pd.DataFrame): List of dataframes containing the data to plot.
    nplots (int): Number of subplots to create.
    var (str): Base name of the variable to plot.
    labels (list of str): List of labels for each dataframe.
    ylabels (list of str): List of y-axis labels for each subplot.
    bounds (list of tuple, optional): List of bounds for each subplot. Each tuple contains (lower_bound, upper_bound). Default is None.

    Returns:
    fig (matplotlib.figure.Figure): The created figure.
    ax (numpy.ndarray of matplotlib.axes._subplots.AxesSubplot): Array of subplot axes.
    """
    fig, ax = plt.subplots(sharex=True, figsize=(12, 8))
    
    for j, df in enumerate(dfs):
        cumulative = df["econ_rewards"].cumsum()
        ax.plot(df['time'], cumulative, label=labels[j], c=colors[j], linestyle=linestyles[j])
        ax.set_ylabel(ylabel)
    
    ax.legend(loc='upper left', bbox_to_anchor=(0, 1))
    fig.tight_layout()
    return fig, ax

