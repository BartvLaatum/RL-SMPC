import matplotlib.pyplot as plt
import pandas as pd
import os
import cmcrameri.cm as cmc

CONVERT_IDX = 96

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

def load_data(project_name, filename, alg):
    data = pd.read_csv(os.path.join(f"data/{project_name}/{alg}", filename))
    return data

def plot_states(dfs, nplots, var, start_day, n_days, labels, ylabels, colors, bounds=None, n=None):
    if n is not None:
        dfs = [df.iloc[:n] for df in dfs]
    fig, ax = plt.subplots(1, nplots, sharex=True, figsize=(16, 4))
    linestyles = ['-', '--', '-.', ':', '-']

    start_idx = int(start_day*CONVERT_IDX)
    end_idx = int(start_idx + (n_days*CONVERT_IDX))

    for j, df in enumerate(dfs):
        for i in range(nplots):
            ax[i].step(
                df['time'].iloc[start_idx:end_idx],
                df['{}_{}'.format(var, i)][start_idx:end_idx], 
                c=colors[j], 
                linestyle=linestyles[j], 
                label=labels[j], 
                where='post'
            )
            ax[i].set_ylabel(ylabels[i])
            if bounds[i] is not None:
                ax[i].hlines(
                    bounds[i], 
                    df['time'][start_idx],
                    df['time'][end_idx],
                    linestyle='--', 
                    color='grey'
                )

    return fig, ax

def remove_unusedu(dfm):
    for col in ['u_0', 'u_1', 'u_2']:
        # Extract the first value of the u_0 column
        first_value = dfm.loc[0, col]

        # Shift the u_0 column up by one position
        dfm[col] = dfm[col].shift(-1)

        # Set the last value of the u_0 column to the extracted first value
        dfm.loc[dfm.index[-1], col] = first_value

    dfm = dfm[:-1]
    return dfm


start_day = 30
n_days = 2

project = "matching-thesis"
mpcs =  ["mpc-1H-40D.csv", "mpc-6H-40D.csv"]
rl_mpcs = ["rl-mpc-1H-40D-wobbly-brook.csv", "rl-mpc-6H-40D-wobbly-brook.csv"]
rls = ["wobbly-brook-60.csv"]

dfs = [load_data(project, exp_name, "mpc") for exp_name in mpcs]
dfs += [load_data(project, model_name, "rlmpc") for model_name in rl_mpcs]
dfs += [load_data(project, model_name, "rl") for model_name in rls]

labels = ["MPC 1H", "MPC 6H", "RL-MPC 1H", "RL-MPC 6H", "SAC"]
# Plots
colormap = cmc.batlowS
colors = [colormap(i) for i in range(3, 3+len(dfs))]  # Picking spaced points
ylims = [None, None, None, None]
ylabels = [r"DW (g/m$^2$)", r"CO$_2$ (ppm)", r"Temp ($^\circ$C)", r"RH (%)"]


fig, ax = plot_states(dfs, 4, var='y', start_day=start_day, n_days=n_days, labels=labels, ylabels=ylabels, bounds=ylims, colors=colors)
fig.tight_layout()
fig.legend(labels=labels, loc='upper center' ,ncol=4, bbox_to_anchor=(0.5, 1.1))
plt.show()

ylims = [None, None, None, None]
ylabels = [r"$u_{CO_2}$ (mg/m$^2$/s)", r"$u_{vent}$ (mm/s)", r"$u_{heat}$ (W/m$^2$)"]
fig, ax = plot_states(dfs, 3, var='u', start_day=start_day, n_days=n_days, labels=labels, ylabels=ylabels, bounds=ylims, colors=colors)
fig.tight_layout()
fig.legend(labels=labels, loc='upper center' ,ncol=4, bbox_to_anchor=(0.5, 1.1))
plt.show()

ylabels = [r"Glob rad (W/m$^2$)", r"CO$_2$ (ppm)", r"Temp ($^\circ$C)", r"RH (%)"]
bounds = [None, None, None, None]

# Compare the cumulative rewards of several implementations
plot_cumulative_rewards(dfs[:], labels, colors, "Economic reward (EU/m2)", ["-","-", "-", "-", "-"])