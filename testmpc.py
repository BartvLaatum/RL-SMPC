import argparse
from typing import Any, Dict, List, Tuple

import yaml
import numpy as np
import casadi as ca
import pandas as pd

from common.utils import *
from mpc_unused import MPC, Experiment


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_name", type=str)
    parser.add_argument("--env_id", type=str, default="LettuceGreenhouse")
    parser.add_argument("--save_name", type=str)
    parser.add_argument("--weather_filename", default="outdoorWeatherWurGlas2014.csv", type=str)
    
    args = parser.parse_args()
    # load the config file
    with open(f"configs/envs/{args.env_id}.yml", "r") as file:
        env_params = yaml.safe_load(file)

    with open("configs/models/mpc.yml", "r") as file:
        mpc_params = yaml.safe_load(file)

    # p = DefineParameters()
    p = get_parameters()
    mpc = MPC(**env_params, **mpc_params[args.env_id])
    mpc.define_nlp(p)
    exp = Experiment(mpc, args.save_name, args.project_name, args.weather_filename)
    us_opt, Js_opt, sol, xs_opt = mpc.solve_nlp_problem(exp.x[:, 0], exp.u[:, 0], exp.d[:, 0:0+exp.mpc.Np], step=0)

    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    time = np.arange(0, mpc.Np+1)*900/86400

    for i in range(3):
        axs[i].plot(time[:-1], us_opt[i, :], label=f'u{i+1}')
        axs[i].set_ylabel(f'u{i+1}')
        axs[i].legend()
        axs[i].grid(True)

    axs[-1].set_xlabel('Time step')

    plt.tight_layout()
    fig.savefig("u_testmpc.png")

    fig, axs = plt.subplots(4, 1, figsize=(10, 8), sharex=True)

    for i in range(4):
        axs[i].plot(time, xs_opt[i, :], label=f'x{i+1}')
        axs[i].set_ylabel(f'x{i+1}')
        axs[i].legend()
        axs[i].grid(True)

    axs[-1].set_xlabel('Days')
    print(xs_opt[:,0])
    plt.tight_layout()
    # plt.show()
    fig.savefig("x_testmpc.png")

    
    # plt.show()
