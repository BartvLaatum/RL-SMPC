from visualisations.rl_smpc_performance import load_data
import numpy as np
import matplotlib.pyplot as plt
import plot_config

def compute_costs(df, algo_values):
    if "run" in df.columns and "y_0" in df.columns:
        run_diff = df.groupby("run")["y_0"].apply(lambda x: x.iloc[-1] - x.iloc[0])
        algo_values["y_0_diff"] = run_diff.mean()
    else:
        print("DataFrame does not contain required 'run' or 'y_0' columns.")

def main():
    model_name = "restful-pyramid-7"
    H = "1H"
    data, horizons = load_data([model_name], mode="stochastic", project="SMPC", Ns=[], smpc=True, zero_order=True, terminal=True, first_order=False, uncertainty_value=0.1)


    # Choose the horizon we want to plot
    horizon = "6H"
    metrics = ["rewards", "penalties", "econ_rewards"]

    # Prepare a dictionary to store the average cumulative averages for each algorithm
    results = {}
    algos = ["mpc", "rl-zero-terminal-smpc"]
    # For each algorithm (e.g., "mpc", "rl-zero-terminal-smpc", etc.) in the data dictionary:
    for algo in algos:
        algo_data = data[algo]
        print(f"Processing algorithm: {algo}")
        # Get the dataframe for the selected horizon
        if algo == "rl-zero-terminal-smpc":
            df = algo_data[horizon][model_name]
        else:
            df = algo_data[horizon]
        # Check if the dataframe is empty
        if df.empty:
            print(f"Warning: Dataframe for {algo} is empty.")
            continue
        # if horizon not in algo_data:
        #     continue
        
        # If there is a column indicating simulation run (e.g., 'simulation'), group by it.
        # Otherwise, assume all rows belong to a single simulation run.
        if "run" in df.columns:
            algo_values = {}
            for metric in metrics:
                # For each simulation run, compute the final value of the cumulative average.
                # Then average across runs.
                run_vals = df.groupby("run")[metric].apply(lambda x: x.expanding().sum().iloc[-1])
                algo_values[metric] = run_vals.mean()
                compute_costs(df, algo_values)
        else:
            algo_values = {}
            for metric in metrics:
                # Compute the running (cumulative) average and take its final value
                algo_values[metric] = df[metric].expanding().mean().iloc[-1]
        
        results[algo] = algo_values

    # Set up the grouped bar plot
    algos = list(results.keys())
    n_algos = len(algos)
    x = np.arange(len(metrics))
    width = 0.8 / n_algos  # total width spread for each group

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["C0", "C3"]
    for idx, algo in enumerate(algos):
        # extract the values in order of metrics
        values = [results[algo][m] for m in metrics]
        ax.bar(x + idx * width, values, width, label=algo, color=colors[idx])

    # Format the plot
    ax.set_xticks(x + width * (n_algos - 1) / 2)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Cumulative average")
    ax.set_title("Cumulative average per metric (" + horizon + ")")
    ax.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()