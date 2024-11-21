from utils import *
import matplotlib.pyplot as plt


def plot_measurements(ops, y, c, title='Measurements', save=False):
    fig, ax = plt.subplots(ops['nx'])
    fig.set_size_inches(10.5, 10.5)
    for ll in range(ops['nx']):
        ax[ll].plot(ops['t'][:]/c, y[ll,0:], drawstyle='steps')
        ax[ll].grid(True)
        ax[ll].set_xlim(left=ops['t'][0], right=ops['t'][-1]/c)
        if ll == 0:
            ax[ll].set_ylabel(r'$y_1$ (g/m$^2$)', fontsize=16)
        elif ll == 1:
            ax[ll].set_ylabel(r'$y_2$ (ppm) $\cdot 10^3$', fontsize=16)
        elif ll == 2:
            ax[ll].set_ylabel(r'$y_3$ ($^\circ$C)', fontsize=16)
        elif ll == 3:
            ax[ll].set_ylabel(r'$y_4$ (\%)', fontsize=16)
            ax[ll].set_xlabel(r'Time (days)', fontsize=16)
    if save:
        fig.savefig(f"{title}.png")
    plt.show()

def plot_disturbances(ops, d, c, title='Weather',save=False):
    fig, ax = plt.subplots(ops['nd'])
    fig.set_size_inches(10.5, 10.5)
    for ll in range(ops['nd']):
        ax[ll].plot(ops['t'][:]/c, d[ll,:], drawstyle='steps')
        ax[ll].grid(True)
        ax[ll].set_xlim(left=ops['t'][0], right=ops['t'][-1]/c)
        if ll == 0:
            ax[ll].set_ylabel(r'$d_1$ (W/m$^2$)', fontsize=16)
        elif ll == 1:
            ax[ll].set_ylabel(r'$d_2$ (ppm) $\cdot 10^3$', fontsize=16)
        elif ll == 2:
            ax[ll].set_ylabel(r'$d_3$ ($^\circ$C)', fontsize=16)
        elif ll == 3:
            ax[ll].set_ylabel(r'$d_4$ (\%)', fontsize=16)
            ax[ll].set_xlabel(r'Time (days)', fontsize=16)
    if save:
        fig.savefig(f"{title}.png")
    plt.show()

def plot_controls(ops, u, c, title='Controls', save=False):

    fig, ax = plt.subplots(ops['nu'])
    fig.set_size_inches(10.5, 10.5)
    for ll in range(ops['nu']):
        ax[ll].plot(ops['t'][:]/c, u[ll,:], drawstyle='steps')
        ax[ll].grid(True)
        ax[ll].set_xlim(left=ops['t'][0], right=ops['t'][-1]/c)
        if ll == 0:
            ax[ll].set_ylabel(r'$u_1$ (mg/m$^2$/s)', fontsize=16)
        elif ll == 1:
            ax[ll].set_ylabel(r'$u_2$ (mm/s)', fontsize=16)
        elif ll == 2:
            ax[ll].set_ylabel(r'$u_3$ (W/m$^2$)', fontsize=16)
            ax[ll].set_xlabel(r'Time (days)', fontsize=16)
    if save:
        fig.savefig(f"{title}.png")
    plt.show()

if __name__ == '__main__':
    ops = {}
    nDays = 1
    c = 86400
    ops["h"] = 900                                      # control interval (s)
    ops["L"]  = nDays*c                                 # final time simulation
    ops['nx'] = 4
    ops['nu'] = 3
    ops['nd'] = 4
    ops["N"] = int(ops["L"]/ops["h"])                           # number of samples in initial time vector
    ops["Np"] = 0
    ops["t"]  = np.arange(0,ops["L"]+ops["h"],ops["h"]) # initial time vector
    d          = LoadDisturbances(ops)

    # reshape the constraints
    x          = np.zeros((ops['nx'],ops['N']+1))
    y          = np.zeros((ops['nx'],ops['N']))
    x[:,0]     = np.array([0.0035, 1e-03, 15, 0.008])
    u          = np.zeros((ops['nu'],ops['N']+1))
    u[:,0]     = np.array([0, 0, 0])
    c = 86400
    p = DefineParameters()
    uopt = np.load('Opt-controls-penalties.npy')

    # plot
    for kk in range(ops['N']):
        # propagate model one time step ahead
        x[:,kk+1]  = f(x[:,kk], u[:, kk], d[:,kk], p, ops['h'])
        # measure output from model
        y[:,kk]    = g(x[:,kk], u[:, kk], d[:,kk], p, ops['h'])
        u[:,kk+1]     = uopt[:,kk]

    # convert outside Co2 and humidity to desired units
    d[1,1:]    = co2dens2ppm(d[2,1:], d[1,1:])*1e-3
    d[3,1:]    = vaporDens2rh(d[2,1:], d[3,1:])


    plot_measurements(ops, y, c, title='Measurements-pen', save=True)
    plot_disturbances(ops, d, c, title='Weather-pen')
    plot_controls(ops, u, c, title='Controls-pen', save=True)
