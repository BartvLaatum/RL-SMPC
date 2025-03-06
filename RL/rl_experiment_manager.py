import os
import gc
import argparse

import numpy as np

from torch.nn.modules.activation import ReLU, SiLU, Tanh, ELU
from torch.optim import Adam, RMSprop
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

from common.rl_utils import *
from common.utils import load_env_params

ACTIVATION_FN = {"relu": ReLU, "silu": SiLU, "tanh":Tanh, "elu": ELU}
OPTIMIZER = {"adam": Adam, "rmsprop": RMSprop}
ACTION_NOISE = {"normalactionnoise": NormalActionNoise, "ornstein_uhlenbeck": OrnsteinUhlenbeckActionNoise}

class RLExperimentManager:
    """
    Class to manage reinforcement learning experiments.

    The RLExperimentManager is responsible for setting up, running, and managing 
    reinforcement learning experiments using specified algorithms and environments. 
    It handles the initialization of environments, model parameters, and the training 
    process, including evaluation and logging of results.

    Attributes:
        env_id (str): ID of the environment to train on.
        project (str): Wandb project name for logging.
        env_params (dict): Parameters for the environment.
        hyperparameters (dict): Hyperparameters for the RL algorithm.
        n_envs (int): Number of parallel environments to use.
        total_timesteps (int): Total number of timesteps for training.
        stochastic (bool): Whether to run the experiment in stochastic mode.
        group (str): Wandb group name for logging.
        n_eval_episodes (int): Number of episodes to evaluate the agent.
        n_evals (int): Number of evaluations during training.
        algorithm (str): RL algorithm to use (e.g., PPO, SAC).
        env_seed (int): Seed for the environment for reproducibility.
        model_seed (int): Seed for the model for reproducibility.
        save_model (bool): Whether to save the trained model.
        save_env (bool): Whether to save the environment normalization.
        hp_tuning (bool): Whether hyperparameter tuning is being performed.
        device (str): Device to run the experiment on (e.g., "cpu", "cuda").
        models (dict): Mapping of algorithm names to their respective model classes.
        model_class: The specific model class to be used based on the algorithm.
        env: The training environment.
        eval_env: The evaluation environment.
        model: The RL model instance.

    Methods:
        init_envs() -> None:
            Initializes the training and evaluation environments.

        initialise_model() -> None:
            Initializes the model for training or continued training.

        build_model_parameters() -> dict:
            Constructs the model parameters from the hyperparameters.

        run_experiment() -> None:
            Executes the experiment with the initialized model and environments.

    Example:
        experiment_manager = RLExperimentManager(
            env_id="LettuceGreenhouse",
            project="rlmpc",
            env_params=env_params,
            hyperparameters=hyperparameters,
            group="group",
            n_eval_episodes=5,
            n_evals=10,
            algorithm="sac",
            env_seed=42,
            model_seed=42,
            stochastic=False
        )
        experiment_manager.run_experiment()
    """
    def __init__(
        self,
        env_id,
        project,
        env_params,
        hyperparameters,
        group,
        n_eval_episodes,
        n_evals,
        algorithm,
        env_seed,
        model_seed,
        stochastic,
        save_model=True,
        save_env=True,
        hp_tuning=False,
        device="cpu"
    ):
        """
        Initialize the ExperimentManager with the given parameters.

        Arguments:
            env_id (str): ID of the environment to train on.
            project (str): Wandb project name.
            group (str): Wandb group name.
            total_timesteps (int): Total number of timesteps for training.
            n_eval_episodes (int): Number of episodes to evaluate the agent.
            num_cpus (int): Number of CPUs to use during training.
            n_evals (int): Number of evaluations during training.
            algorithm (str): RL algorithm to use.
            env_seed (int): Seed for the environment.
            model_seed (int): Seed for the model.
            stochastic (bool): Whether to run the experiment in stochastic mode.
            save_model (bool): Whether to save the model.
            save_env (bool): Whether to save the environment.
            hp_tuning (bool): Whether hyperparameter tuning is being performed.
            device (str): Device to run the experiment on (e.g., "cpu", "cuda").
        """
        self.env_id = env_id
        self.project = project
        self.env_params = env_params
        self.n_envs = hyperparameters["n_envs"]
        self.total_timesteps = hyperparameters["total_timesteps"]
        self.stochastic = stochastic
        del hyperparameters["total_timesteps"] 
        del hyperparameters["n_envs"]
        self.hyperparameters=hyperparameters
        self.group = group
        self.n_eval_episodes = n_eval_episodes
        self.n_evals = n_evals
        self.algorithm = algorithm
        self.env_seed = env_seed
        self.model_seed = model_seed
        self.save_model = save_model
        self.save_env = save_env
        self.hp_tuning = hp_tuning
        self.device = device
        self.models = {"ppo": PPO, "sac": SAC}

        self.model_class = self.models[self.algorithm.lower()]


        # Load environment and model parameters
        self.hyp_config_path = f"configs/sweeps/"

        # Initialize the environments
        print("Tuning:", self.hp_tuning)
        if not self.hp_tuning:
            self.run, self.config = wandb_init(
                self.hyperparameters,
                self.env_seed,
                self.model_seed,
                project=self.project,
                group=self.group,
                save_code=False
            )

            self.init_envs(self.hyperparameters["gamma"])
            self.model_params = self.build_model_parameters()
            print(self.env.observation_space.shape)
            # Initialize the model
            self.initialise_model()

    def init_envs(self, gamma):
        """
        Initialize training and evaluation environments
        """
        self.monitor_filename = None
        vec_norm_kwargs = {
            "norm_obs": True,
            "norm_reward": False,
            "clip_obs": 10,
            "gamma": gamma
        }

        # Setup new environment for training
        self.env = make_vec_env(
            self.env_id,
            self.env_params,
            seed=self.env_seed,
            n_envs=self.n_envs,  # Number of environments to run in parallel
            monitor_filename=self.monitor_filename,
            vec_norm_kwargs=vec_norm_kwargs
        )

        self.eval_env = make_vec_env(
            self.env_id,
            self.env_params,
            seed=self.env_seed,
            n_envs=1,                               # Only one 'parallel' environment for evaluation at the moment
            monitor_filename=self.monitor_filename,
            vec_norm_kwargs=vec_norm_kwargs,
            eval_env=True,
        )

    def initialise_model(self):
        """
        Initialize the model for training or continued training.

        Args:
            runname (str): Name of the run.
            job_type (str): Type of job (default is 'train').
        """
        if self.stochastic:
            tensorboard_log = f"train_data/{self.project}/{self.algorithm}/stochastic/logs/{self.run.name}"
        else:
            tensorboard_log = f"train_data/{self.project}/{self.algorithm}/deterministic/logs/{self.run.name}"
        # Initialize a new model for training
        self.model = self.model_class(
            env=self.env,
            seed=self.model_seed,
            verbose=1,
            **self.model_params,
            tensorboard_log=tensorboard_log,
            device=self.device
        )

    def build_model_parameters(self):
        """
        Constructs and returns the model parameters for the reinforcement learning model.

        This method processes the hyperparameters provided during initialization to 
        configure the model parameters. It handles the conversion of string identifiers 
        for activation functions and optimizers into their respective classes. Additionally, 
        it sets up action noise parameters if the SAC algorithm is used.

        Returns:
            dict: A dictionary containing the model parameters, including any necessary 
                  conversions for activation functions, optimizers, and action noise.
        """
        model_params = self.hyperparameters.copy()
        if model_params["schedule"] == True:
            model_params["learning_rate"] = self.linear_schedule(model_params["learning_rate"])
        del model_params["schedule"]
        if "policy_kwargs" in self.hyperparameters:
            policy_kwargs = self.hyperparameters["policy_kwargs"].copy()  # Copy to avoid modifying the original

            if "activation_fn" in policy_kwargs:
                activation_fn_str = policy_kwargs["activation_fn"]
                if activation_fn_str in ACTIVATION_FN:
                    policy_kwargs["activation_fn"] = ACTIVATION_FN[activation_fn_str]
                else:
                    raise ValueError(f"Unsupported activation function: {activation_fn_str}")

            # Handle other necessary conversions (e.g., optimizers)
            if "optimizer_class" in policy_kwargs:
                optimizer_str = policy_kwargs["optimizer_class"]
                if optimizer_str in OPTIMIZER:
                    policy_kwargs["optimizer_class"] = OPTIMIZER[optimizer_str]
                else:
                    raise ValueError(f"Unsupported optimizer: {optimizer_str}")
            model_params["policy_kwargs"] = policy_kwargs
        if self.algorithm == "sac":
            if "action_noise" in self.hyperparameters:
                action_noise_key, noise_params = next(iter(self.hyperparameters["action_noise"].items()))
                
                if action_noise_key in ACTION_NOISE:
                    action_noise = ACTION_NOISE[action_noise_key](
                        mean=np.zeros(self.env.action_space.shape),
                        sigma=noise_params["sigma"] * np.ones(self.env.action_space.shape)
                    )
                model_params["action_noise"] = action_noise
        return model_params

    def run_experiment(self):
        """
        Executes the reinforcement learning experiment.

        This method manages the training process of the RL model using the initialized
        environments and model parameters. It sets up logging directories based on the 
        experiment's stochastic or deterministic nature, calculates evaluation frequency, 
        and creates necessary callbacks for model evaluation and environment saving. 
        The method then trains the model for the specified number of timesteps, saves 
        the final model and environment normalization if required, and performs cleanup 
        operations post-training.
        """
        if self.stochastic:
            model_log_dir = f"train_data/{self.project}/{self.algorithm}/stochastic/models/{self.run.name}/" if self.save_model else None
            env_log_dir = f"train_data/{self.project}/{self.algorithm}/stochastic/envs/{self.run.name}/" if self.save_env else None
        else:
            model_log_dir = f"train_data/{self.project}/{self.algorithm}/deterministic/models/{self.run.name}/" if self.save_model else None
            env_log_dir = f"train_data/{self.project}/{self.algorithm}/deterministic/envs/{self.run.name}/" if self.save_env else None

        eval_freq = self.total_timesteps // self.n_evals // self.n_envs
        save_name = "vec_norm"

        callbacks = create_callbacks(
            self.n_eval_episodes,
            eval_freq,
            env_log_dir,
            save_name,
            model_log_dir,
            self.eval_env,
            run=self.run,
            results=None,
            save_env=self.save_env,
            verbose=1 # verbose-2; debug messages.
        )

        # Train the model
        self.model.learn(total_timesteps=self.total_timesteps, callback=callbacks, reset_num_timesteps=False)
        if model_log_dir:
            self.model.save(os.path.join(model_log_dir, "last_model"))

        # Save the environment normalization
        if env_log_dir:
            env_save_path = os.path.join(env_log_dir, "last_vecnormalize.pkl")
        self.model.get_vec_normalize_env().save(env_save_path)

        # Clean up and finalize the run
        self.run.finish()
        self.env.close()
        self.eval_env.close()
        del self.model, self.env, self.eval_env
        gc.collect()

    def linear_schedule(self, initial_value: float):
        """
        Linear learning rate schedule.

        :param initial_value: Initial learning rate.
        :return: schedule that computes
        current learning rate depending on remaining progress
        """
        def func(progress_remaining: float) -> float:
            """
            Progress will decrease from 1 (beginning) to 0.

            :param progress_remaining:
            :return: current learning rate
            """
            return progress_remaining * initial_value

        return func


    # # def build_model_hyperparameters(self, config):
    # #     """Build the model hyperparameters from the given config."""

    # #     self.model_params["policy"] = config["policy"]
    # #     self.model_params["learning_rate"] = config["learning_rate"]
    # #     self.model_params["n_steps"] = config["n_steps"]
    # #     self.model_params["batch_size"] = config["batch_size"]
    # #     self.model_params["n_epochs"] = config["n_epochs"]
    # #     self.model_params["gamma"] = 1.0 - config["gamma_offset"]
    # #     self.model_params["gae_lambda"] = config["gae_lambda"]
    # #     self.model_params["clip_range"] = config["clip_range"]
    # #     self.model_params["ent_coef"] = config["ent_coef"]
    # #     self.model_params["vf_coef"] = config["vf_coef"]

    # #     self.model_params["use_sde"] = config["use_sde"]
    # #     self.model_params["sde_sample_freq"] = config["sde_sample_freq"]
    # #     self.model_params["target_kl"] = config["target_kl"]
    # #     self.model_params["normalize_advantage"] = config["normalize_advantage"]

    # #     policy_kwargs = {}
    # #     policy_kwargs["net_arch"] = {}
        
    # #     if self.algorithm == "ppo":
    # #         policy_kwargs["net_arch"]["pi"] = [config["pi"]]*3
    # #         policy_kwargs["net_arch"]["vf"] = [config["vf"]]*3

    # #     policy_kwargs["optimizer_kwargs"] = config["optimizer_kwargs"]
    # #     policy_kwargs["activation_fn"] = ACTIVATION_FN[config["activation_fn"]]

    # #     if self.algorithm == "recurrentppo":
    # #         policy_kwargs["net_arch"]["pi"] = [config["pi"]]*2
    # #         policy_kwargs["net_arch"]["vf"] = [config["vf"]]*2
    # #         policy_kwargs["lstm_hidden_size"] = config["lstm_hidden_size"]
    # #         policy_kwargs["enable_critic_lstm"] = config["enable_critic_lstm"]
    # #         if policy_kwargs["enable_critic_lstm"]:
    # #             policy_kwargs["shared_lstm"] = False
    # #         else:
    # #             policy_kwargs["shared_lstm"] = True
    # #     self.model_params["policy_kwargs"].update(policy_kwargs)

    # # def run_single_sweep(self):
    # #     """
    # #     Main function for hyperparameter tuning.
    # #     """
    # #     # wandb.tensorboard.patch(root_logdir="...")
    # #     with wandb.init(sync_tensorboard=True) as run:
    # #         self.run = run
    # #         self.config = wandb.config
    # #         self.build_model_hyperparameters(self.config)
    # #         self.init_envs()
    # #         self.initialise_model()
    # #         print(self.model.policy) 
    # #         self.run_experiment()

    # def hyperparameter_tuning(self):
    #     """
    #     Perform hyperparameter tuning for the model. Using the Sweep API from Weights and Biases.
    #     """
    #     continue_sweep = True
    #     sweep_config = load_sweep_config(self.hyp_config_path, self.env_id, self.algorithm)
    #     sweep_id = wandb.sweep(sweep=sweep_config, project=self.project)
    #     wandb.agent(sweep_id, function=self.run_single_sweep, count=100)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str, default="rlmpc", help="Wandb project name")
    parser.add_argument("--env_id", type=str, default="LettuceGreenhouse", help="Environment ID")
    parser.add_argument("--algorithm", type=str, default="ppo", help="RL algorithm to use")
    parser.add_argument("--group", type=str, default="group1", help="Wandb group name")
    parser.add_argument("--n_eval_episodes", type=int, default=1, help="Number of episodes to evaluate the agent for")
    parser.add_argument("--n_evals", type=int, default=5, help="Number times we evaluate algorithm during training")
    parser.add_argument("--env_seed", type=int, default=666, help="Random seed for the environment for reproducibility")
    parser.add_argument("--model_seed", type=int, default=666, help="Random seed for the RL-model for reproducibility")
    parser.add_argument("--stochastic", action="store_true", help="Whether to run the experiment in stochastic mode")
    parser.add_argument("--device", type=str, default="cpu", help="The device to run the experiment on")
    parser.add_argument("--save_model", default=True, action=argparse.BooleanOptionalAction, help="Whether to save the model")
    parser.add_argument("--save_env", default=True, action=argparse.BooleanOptionalAction, help="Whether to save the environment")
    parser.add_argument("--hyperparameter_tuning", default=False, action=argparse.BooleanOptionalAction, help="Perform hyperparameter tuning")
    args = parser.parse_args()

    # assert args.num_cpus <= cpu_count(), f"Number of CPUs requested ({args.num_cpus}) is greater than available ({cpu_count()})"
    env_params = load_env_params(args.env_id)
    hyperparameters, rl_env_params = load_rl_params(args.env_id, args.algorithm)
    env_params.update(rl_env_params)

    # Initialize the experiment manager
    experiment_manager = RLExperimentManager(
        env_id=args.env_id,
        project=args.project,
        env_params=env_params,
        hyperparameters=hyperparameters,
        group=args.group,
        n_eval_episodes=args.n_eval_episodes,
        n_evals=args.n_evals,
        algorithm=args.algorithm,
        env_seed=args.env_seed,
        model_seed=args.model_seed,
        stochastic=args.stochastic,
        save_model=args.save_model,
        save_env=args.save_env,
        hp_tuning=args.hyperparameter_tuning,
        device=args.device
    )

    # if args.hyperparameter_tuning:
    #     # Perform hyperparameter tuning
    #     experiment_manager.hyperparameter_tuning()
    # else:
    # # Run the experiment
    experiment_manager.run_experiment()
