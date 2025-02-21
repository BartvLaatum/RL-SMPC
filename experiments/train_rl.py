import argparse

from RL.rl_experiment_manager import RLExperimentManager
from common.rl_utils import load_rl_params 
from common.utils import load_env_params

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str, default="matching-thesis", help="Wandb project name")
    parser.add_argument("--env_id", type=str, default="LettuceGreenhouse", help="Environment ID")
    parser.add_argument("--algorithm", type=str, default="sac", help="RL algorithm to use")
    parser.add_argument("--group", type=str, default="group1", help="Wandb group name")
    parser.add_argument("--n_eval_episodes", type=int, default=1, help="Number of episodes to evaluate the agent for")
    parser.add_argument("--n_evals", type=int, default=10, help="Number times we evaluate algorithm during training")
    parser.add_argument("--mode", type=str, choices=['deterministic', 'stochastic'], required=True)
    parser.add_argument("--uncertainty_value", type=float, help="List of uncertainty scale values")
    parser.add_argument("--env_seed", type=int, default=666, help="Random seed for the environment for reproducibility")
    parser.add_argument("--model_seed", type=int, default=666, help="Random seed for the RL-model for reproducibility")
    parser.add_argument("--device", type=str, default="cpu", help="The device to run the experiment on")
    parser.add_argument("--save_model", default=True, action=argparse.BooleanOptionalAction, help="Whether to save the model")
    parser.add_argument("--save_env", default=True, action=argparse.BooleanOptionalAction, help="Whether to save the environment")
    args = parser.parse_args()

    assert args.mode in ['deterministic', 'stochastic'], "Mode must be either 'deterministic' or 'stochastic'"
    if args.mode == 'stochastic':
        assert args.uncertainty_value is not None, "Uncertainty scale must be provided for stochastic mode"
        assert (0 < args.uncertainty_value < 1), "Uncertainty scale values must be between 0 and 1"
        group = f"{args.algorithm}-{args.mode[:3]}-{args.uncertainty_value}"
    else:
        args.uncertainty_value = 0
        group = f"{args.algorithm}-{args.mode[:3]}"

    env_params = load_env_params(args.env_id)

    # Initialize the experiment manager
    # uncertainties = np.linspace(0.1, 0.3, 7)
    hyperparameters, rl_env_params = load_rl_params(args.env_id, args.algorithm)
    env_params.update(rl_env_params)
    env_params["uncertainty_scale"] = args.uncertainty_value

    experiment_manager = RLExperimentManager(
        env_id=args.env_id,
        project=args.project,
        env_params=env_params,
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
        device=args.device
    )
    experiment_manager.run_experiment()
