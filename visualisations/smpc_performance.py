import os
import argparse

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cmcrameri.cm as cmc
from performance_plots import create_plot

import plot_config

def load_data(model_names, mode, project, Ns=[], uncertainty_value=None):
    """
    Load and organize simulation data from MPC, RL, and RL-MPC experiments.
    This function reads CSV files containing results from different control strategies
    and organizes them into a nested dictionary structure.
    Parameters
    ----------
    model_names : list
        List of RL model names to load data for
    mode : str
        Operating mode of the simulation (e.g., 'train', 'test')  
    project : str
        Project name/folder containing the data
    uncertainty_value : float, optional
        Scale factor for uncertainty in MPC predictions. If provided, loads data 
        with specified uncertainty scale suffix.
    Returns
    -------
    tuple
        - data : dict
            Nested dictionary containing loaded dataframes organized by:
            - 'mpc': Dict of dataframes indexed by horizon
            - 'rl': Dict of dataframes indexed by model name  
            - 'rlmpc': Dict of dicts indexed by horizon then model name
        - horizons : list
            List of horizon values used in the simulations
    Notes
    -----
    Expected file structure:
    data/
        {project}/
            {mode}/
                rl/
                    {model}.csv
                mpc/
                    mpc-{horizon}-{uncertainty}.csv
                rlmpc/
                    rlmpc-{model}-{horizon}-{uncertainty}.csv
    """
    horizons = ['1H', '2H', '3H', '4H', '5H', '6H']
    data = {
        'mpc': {},
        'smpc': {},
        'rl': {},
        'rlmpc': {},
        'rl-zero-smpc': {},
        'rl-first-smpc': {},
        'smpc-tight': {}
    }

    uncertainty_suffix = f'-{uncertainty_value}' if uncertainty_value else ''

    for model in model_names:
        # Load RL data
        rl_path = f'data/{project}/{mode}/rl/{model}.csv'
        if os.path.exists(rl_path):
            data['rl'][model] = pd.read_csv(rl_path)

        # Load MPC and RL-MPC data for each horizon
        for h in horizons:
            rlsmpc_path = f'data/{project}/{mode}/rlsmpc/{model}-zero-order-{h}{uncertainty_suffix}.csv'
            rl_first_smpc_path = f'data/{project}/{mode}/rlsmpc/{model}-first-order-{h}{uncertainty_suffix}.csv'

            if os.path.exists(rlsmpc_path):
                if h not in data['rl-zero-smpc']:
                    data['rl-zero-smpc'][h] = {}
                data['rl-zero-smpc'][h][model] = pd.read_csv(rlsmpc_path)

            if os.path.exists(rl_first_smpc_path):
                if h not in data['rl-first-smpc']:
                    data['rl-first-smpc'][h] = {}
                data['rl-first-smpc'][h][model] = pd.read_csv(rl_first_smpc_path)

    for h in horizons:
        mpc_path = f'data/{project}/{mode}/mpc/mpc-noise-correction-{h}{uncertainty_suffix}.csv'
        smpc_path = f'data/{project}/{mode}/smpc/smpc-noise-correction-{h}-10Ns{uncertainty_suffix}.csv'

        if h not in data['mpc'] and os.path.exists(mpc_path):
            data['mpc'][h] = pd.read_csv(mpc_path)

        if os.path.exists(smpc_path):
            if h not in data['smpc']:
                data['smpc'][h] = {}
            data['smpc'][h]= pd.read_csv(smpc_path)

        # if os.path.exists(smpc_horizon):
        #     print(h)
        #     if h not in data['smpc-tight']:
        #         data['smpc-tight'][h] = {}
        #     data['smpc-tight'][h] = pd.read_csv(smpc_horizon)    

    return data, horizons

def main(args):
    data, horizons = load_data(args.model_names, args.mode, args.project, Ns=[], uncertainty_value=args.uncertainty_value)
    create_plot(args.figure_name, args.project, data, horizons, args.model_names, args.mode, 'rewards', args.Ns, args.uncertainty_value)
    create_plot(args.figure_name, args.project, data, horizons, args.model_names, args.mode, 'econ_rewards', args.Ns, args.uncertainty_value)
    create_plot(args.figure_name, args.project, data, horizons, args.model_names, args.mode, 'penalties', args.Ns, args.uncertainty_value)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=str, default='SMPC',
                        help='Name of the project')
    parser.add_argument('--model_names', nargs='+', type=str, default=[],
                        help='List of model names to plot')
    parser.add_argument('--Ns', nargs='+', type=int, default=[],
                        help='List of SMPC with Ns scenario samples to plot')
    parser.add_argument('--mode', type=str, 
                        choices=['deterministic', 'stochastic'], required=True)
    parser.add_argument('--uncertainty_value', type=float,
                        help='Uncertainty scale value for stochastic mode')
    parser.add_argument('--figure_name', type=str, required=True)
    args = parser.parse_args()
    main(args)