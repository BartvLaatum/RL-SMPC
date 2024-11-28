import os

import yaml
import numpy as np
import pandas as pd

from mpc import MPC
from visualisations import plot_states
from utils import LoadDisturbancesMpccsv, get_parameters, define_model, co2ppm2dens, rh2vaporDens


def simulate_model(mpc, u, d, xmin, xmax, x0, h):
    model, output = define_model(h, xmin, xmax)
    p = get_parameters()
    
    Xcas = np.zeros((4, mpc.N+1))
    Xcas[:, 0] = x0
    Ycas = np.zeros((4, mpc.N+1))
    Ycas[:, 0] = np.array(output(Xcas[:, 0])).flatten()
    # d = LoadDisturbancesMpccsv(mpc)
    
    for i in range(mpc.N):
        Xcas[:, i+1] = np.array(model(Xcas[:, i], u[:, i], d[:, i], p)).flatten()
        Ycas[:, i+1] = np.array(output(Xcas[:, i+1])).flatten()
    
    return Xcas, Ycas

if __name__ == "__main__":

    with open("configs/mpc.yml", "r") as file:
        mpc_params = yaml.safe_load(file)
    mpc = MPC(**mpc_params["lettuce"])

    xmin = np.array([0.002, 0, 5., 0])
    xmax = np.array([.6, 0.004, 40, 0.051])
    h = 900.
    x0 = np.array([0.0035, 1.e-03, 15., 0.008])
    c = 86400
    time = np.arange(0, mpc.N*h+1, h)/c

    filenames = ["murray.csv"]
    murray_dfs = []
    bart_dfs = []
    for filename in filenames:
        df = pd.read_csv(os.path.join("data/mpc", filename))
        murray_dfs.append(df)
        u = np.array(df[["u_0", "u_1", "u_2"]].values).T
        d = np.array(df[["d_0", "d_1", "d_2", "d_3"]].values).T
        d[1,:] = co2ppm2dens(d[2,:], d[1,:])
        d[3,:] = rh2vaporDens(d[2,:], d[3,:])
        print(u.shape)
        Xcas, Ycas = simulate_model(mpc, u, d, xmin, xmax, x0, h)
        data = {
            "time": time,
            "x_0": Xcas[0, :],
            "x_1": Xcas[1, :],
            "x_2": Xcas[2, :],
            "x_3": Xcas[3, :],
            "y_0": Ycas[0, :],
            "y_1": Ycas[1, :],
            "y_2": Ycas[2, :],
            "y_3": Ycas[3, :]
        }

        df_simulation = pd.DataFrame(data)
        bart_dfs.append(df_simulation)
        
    print(df_simulation.head())
    bounds = [None, None, None, None]

    murray_propagated = pd.read_csv("data/mpc/murray-propagated.csv")

    ylabels = [r"DW (g/m$^2$)", r"CO$_2$ (ppm)", r"Temp ($^\circ$C)", r"RH (%)"]
    fig, ax = plot_states(
        [bart_dfs[-1], murray_dfs[-1], murray_propagated], 
        4, 
        'y',
        labels=["Bart propogated", "Murray MPC solution", "Murray propogated"], 
        ylabels=ylabels,
        linestyles=["-", "--", "-."],
        bounds=bounds
    )
    fig.savefig("comparison.png")
