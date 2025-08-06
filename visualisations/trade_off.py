from rl_smpc_performance import load_data
import argparse
import os
import pandas as pd
import plot_config
import matplotlib.pyplot as plt

def create_scatter_plot(data, horizon='3H'):
    plt.figure(figsize=(10, 6))

    # Colors for different algorithms
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    # Plot each algorithm's data
    for (algo, algo_data), color in zip(data.items(), colors):
        if horizon in algo_data:

            # Calculate means for each run
            grouped_data = algo_data[horizon].groupby('run').sum()
            
            epi_values = grouped_data['econ_rewards']
            penalty_values = grouped_data['penalties']

            # epi_values = algo_data[horizon]['econ_rewards']
            # penalty_values = algo_data[horizon]['penalties']
            plt.scatter(penalty_values, epi_values, label=algo, alpha=0.6, c=color)
    
    plt.xlabel('Penalty')
    plt.ylabel('EPI')
    plt.title(f'EPI vs Penalty for Different Algorithms (Horizon: {horizon})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f'epi_vs_penalty_{horizon}.png', dpi=300)

    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Load and plot data.')
    parser.add_argument('--model_names', nargs='+', help='List of model names')
    parser.add_argument('--mode', type=str, help='Mode of operation')
    parser.add_argument('--project', type=str, help='Project name')
    parser.add_argument('--smpc', action='store_true', help='Include SMPC data')
    parser.add_argument('--zero_order', action='store_true', help='Include zero order data')
    parser.add_argument('--first_order', action='store_true', help='Include first order data')
    parser.add_argument('--terminal', action='store_true', help='Include terminal data')
    args = parser.parse_args()

    data, horizons = load_data(
        args.model_names, 
        args.mode, 
        args.project, 
        smpc=args.smpc,
        zero_order=args.zero_order,
        first_order=args.first_order,
        terminal=args.terminal,
        uncertainty_value=0.1
    )
    print(data)
    create_scatter_plot(data)