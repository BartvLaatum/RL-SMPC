import argparse

from RL.rl_experiment_manager import RLExperimentManager
from common.rl_utils import load_rl_params 
from common.utils import load_env_params

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str, default="SMPC",
                        help="Project name for result organization also used as Wandb project name")
    parser.add_argument("--env_id", type=str, default="LettuceGreenhouse",
                       help="Environment identifier")
    parser.add_argument("--algorithm", type=str, default="sac",
                        help="RL algorithm to use")
    parser.add_argument("--group", type=str, default="group1",
                        help="Wandb group name")
    parser.add_argument("--n_eval_episodes", type=int, default=1,
                        help="Number of episodes to evaluate the agent for")
    parser.add_argument("--n_evals", type=int, default=10,
                        help="Number times we evaluate algorithm during training")
    parser.add_argument("--mode", type=str, choices=['deterministic', 'stochastic'], required=True,
                        help="Mode for parametric uncertainty")
    parser.add_argument("--uncertainty_value", type=float, required=True,
                       help="Parametric uncertainty level")
    parser.add_argument("--env_seed", type=int, default=42,
                        help="Random seed for the environment for reproducibility")
    parser.add_argument("--model_seed", type=int, default=42,
                        help="Random seed for the RL-model for reproducibility")
    parser.add_argument("--device", type=str, default="cpu",
                        help="The device to run the experiment on")
    parser.add_argument("--save_model", default=True, action=argparse.BooleanOptionalAction,
                        help="Whether to save the trained model")
    parser.add_argument("--save_env", default=True, action=argparse.BooleanOptionalAction,
                        help="Whether to save the environment")
    parser.add_argument('--training_years', nargs='+', type=str, default=[],
                        help="List of years to train on for weather trajectories (unused)")
    parser.add_argument("--hyperparameter_tuning", default=False, action=argparse.BooleanOptionalAction,
                        help="Whether to execute hyperparameter tuning")
    args = parser.parse_args()

    assert args.mode in ['deterministic', 'stochastic'], "Mode must be either 'deterministic' or 'stochastic'"
    if args.mode == 'stochastic':
        assert args.uncertainty_value is not None, "Uncertainty scale must be provided for stochastic mode"
        assert (0 < args.uncertainty_value < 1), "Uncertainty scale values must be between 0 and 1"
        group = f"{args.algorithm}-{args.mode[:3]}-{args.uncertainty_value}"
    else:
        args.uncertainty_value = 0
        group = f"{args.algorithm}-{args.mode[:3]}"

    # Load gereral environment parameters
    env_params = load_env_params(args.env_id)

    # Load RL model hyperparameters and RL environment specific parameters
    hyperparameters, rl_env_params = load_rl_params(args.env_id, args.algorithm)
    env_params.update(rl_env_params)
    env_params["uncertainty_value"] = args.uncertainty_value
    eval_env_params = env_params.copy()

    # If you aim to train on multiple years of data and validate on different weather data.
    if args.training_years:
        weather_files = [f"train/KNMI{year}.csv" for year in args.training_years]
        env_params["weather_filename"] = weather_files
        env_params["obs_module"] = "FutureWeatherObservations"
        eval_env_params["obs_module"] = "FutureWeatherObservations"
        env_params["start_day"] += 10 # the KNMI files start at 1 January and evaluation data at 10th January.

    # Create the experiment manager instance
    experiment_manager = RLExperimentManager(
        env_id=args.env_id,
        project=args.project,
        env_params=env_params,
        eval_env_params=eval_env_params,
        hyperparameters=hyperparameters,
        group=group,
        n_eval_episodes=args.n_eval_episodes,
        n_evals=args.n_evals,
        algorithm=args.algorithm,
        env_seed=args.env_seed,
        model_seed=args.model_seed,
        stochastic=True,
        save_model=args.save_model,
        save_env=args.save_env,
        hp_tuning=args.hyperparameter_tuning,
        device=args.device
    )

    # Execute hyperparameter tuning or training
    if args.hyperparameter_tuning:
        # Perform hyperparameter tuning
        experiment_manager.hyperparameter_tuning()
    else:
        experiment_manager.run_experiment()
