"""
Greenhouse dyanmics modelled in OpenAI Gym environment.
The controller for this environment control the valves of the greenhouse.
That regulates amount of heating (W/m2) and CO_2 into the greenhouse.

For now used modelled as a temperature reference tracking problem.
""" 
from typing import Dict, List, Optional, Tuple
from copy import copy
import argparse
import yaml
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from observations import BaseObservations, StandardObservations
from utils import *

observations = {"StandardObservations": StandardObservations}

class LettuceGreenhouse(gym.Env):
    """
    This class implements the original lettuce greenhouse model by Eldert van Henten (1994),
    in a gymnasium environment. Such that off-the-shelf RL libraries can be applied to control this model.
    It solves the model equations using the numerical Runge-Kutta-4 method.

    Arguments:
        weather_data_dir    -- path to the weather data
        nx                  -- number of states
        ny                  -- number of measurements
        nd                  -- number of weather variables
        nu                  -- number of control inputs
        control_rate        -- control rate of the system in minutes
        c                   -- number of seconds in a day
        n_days              -- number of simulation days
        Np                  -- number of future weather predictions to use
        start_day           -- starting day of the simulation
        reward_coefs        -- coefficients for the reward function
        penalty_coefs       -- coefficients for the penalty function
    """
    def __init__(self,
        nx: int,                # number of greenhouse states
        ny: int,                # number of greenhouse measurements
        nd: int,                # number of disturbance (weather variables)
        nu: int,                # number of control inputs
        h: float,               # control rate of the system in minutes
        n_days: int,            # simulation days
        Np: int,                # number of future weather predictions to use
        start_day: int,        # start day of simulation
        x0: List[float],        # initial state of the greenhouse
        constraints: Dict[str, float], # constraints of the environment
        weather_filename: str, # path to the weather data
        lb_pen_w: List[float],
        ub_pen_w: List[float],
        obs_module: str,
        obs_names: List[str],
        ):
        super(LettuceGreenhouse, self).__init__()
        self.weather_filename = weather_filename

        # simulation parameters
        self.x0 = x0
        self.c = 86400                  # conversion between days and seconds               
        self.nx = nx
        self.nu = nu
        self.ny = ny
        self.nd = nd
        self.h = h
        self.n_days = n_days
        self.L = n_days * self.c
        self.start_day = start_day
        self.N = int(self.L//self.h)     # number of steps to take during episode
        # prediction horizon for weather forecast for observation space
        self.Np = Np

        # greenhouse model parameters
        self.p = get_parameters()

        # define the dynamical greenhouse model:
        self.F, self.g = define_model(self.h)

        # observation space based on user input
        self.observation_module = observations[obs_module](obs_names=obs_names)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.observation_module.n_obs,), dtype=np.float32)

        # action space
        self.action_space = spaces.Box(low=-1.*np.ones(nu, dtype=np.float32), high=np.ones(nu, dtype=np.float32))

        # min and max measurements of the environment
        self.obs_low = np.array(
            [constraints["co2_min"],
            constraints["temp_min"],
            constraints["rh_min"]],
            dtype=np.float32
        )
        self.obs_high = np.array(
            [constraints["co2_max"],
            constraints["temp_max"],
            constraints["rh_max"]],
            dtype=np.float32
        )

        # min and max control inputs and 
        self.min_u = np.array(
            [constraints["co2_supply_min"],
            constraints["vent_min"],
            constraints["heat_min"]], 
            dtype=np.float32
        )
        self.max_u = np.array(
            [constraints["co2_supply_max"],
            constraints["vent_max"],
            constraints["heat_max"],], 
            dtype=np.float32
        )
        # previous control and allowed change control
        self.delta_u = self.max_u/10

        # weights for penalty function
        self.lb_pen_w = np.expand_dims(lb_pen_w, axis=0)
        self.ub_pen_w = np.expand_dims(ub_pen_w, axis=0)

    def seed(self, seed):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def get_state(self):
        return self.x.toarray().flatten()

    def get_y(self):
        return self.g(np.copy(self.x)).toarray().flatten()

    def get_d(self):
        
        return transform_disturbances(np.copy(self.d[:, self.timestep]))

    def step(self, action: np.ndarray):
        """
        Step function that simulates one timestep into the future given input action.

        Args:
            actions -- normalised action taken by the rl agent.

        Return:
            observation -- array consisting of four variables
            reward      -- immediate reward of the environment
            terminated  -- whether environment is in a terminal state
            truncated   -- whether the episode was truncated, always false
            info        -- additional information [not used]
        """
        u = self.action_to_control(action)

        # transition state next state given action and observe environment
        self.x = self.F(self.x, u, self.d[:, self.timestep], self.p)
        y = self.get_y()
        self.timestep += 1

        # terminate if state is terminal or crops died
        if self.terminal_state():
            self.terminated = True

        # calculate reward
        reward = self._get_reward(y, u)

        self.prev_u = np.copy(u)
        self.prev_dw = np.copy(self.x[0])

        info = self.get_info()

        return (
            self._get_obs(y),
            reward, 
            self.terminated, 
            False,
            info
            )

    def step2(self, action: np.ndarray):
        """
        Step function that simulates one timestep into the future given action input.
        In this case the action simply is the control input.
        Used for testing system dynamics with control inputs generated by different controller.

        Args:
            actions -- normalised action taken by the rl agent.

        Return:
            observation -- array consisting of four variables
            reward      -- immediate reward of the environment
            terminated  -- whether environment is in a terminal state
            truncated   -- whether the episode was truncated, always false
            info        -- additional information [not used]
        """
        u = action
        np.set_printoptions(suppress=True)
        print(u)

        # transition state next state given action and observe environment

        self.x = self.F(self.x, u, self.d[:,self.timestep], self.p)
        print(self.x)
        y = self.get_y()
        self.timestep += 1

        # terminate if state is terminal or crops died
        if self.terminal_state():
            self.terminated = True

        # calculate reward
        reward = self._get_reward(y, u)

        self.prev_u = np.copy(u)
        self.prev_dw = np.copy(self.x[0])

        info = self.get_info()

        return (
            self._get_obs(y),
            reward, 
            self.terminated, 
            False,
            info
            )

    def _get_obs(self, y: np.ndarray) -> np.ndarray:
        """
        Function that returns the observation space of the environment.
        """
        return self.observation_module.compute_obs(y, self.prev_dw, self.prev_u, self.d[:, self.timestep], self.timestep)

    def get_info(self):
        info =  {"econ_rewards": self.econ_rewards}
        return info

    def _get_reward(self, y, u):
        """
        Reward function that calculates the reward of the environment.
        """
        delta_dw  = self.x[0] - self.prev_dw 
        self.econ_rewards = float(compute_economic_reward(delta_dw, self.p, self.h, u))
        penalties = self._compute_penalty(y)
        return self.econ_rewards - penalties

    def _compute_penalty(self, y: np.ndarray) -> np.ndarray:
        """
        Penalty function that calculates the penalty of the environment.
        """
        lb_violation, ub_violation = self.constraint_violation(y)
        penalty = np.dot(self.lb_pen_w, lb_violation) + np.dot(self.ub_pen_w, ub_violation)
        return np.sum(penalty)

    def constraint_violation(self, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Function that computes the absolute penalties for violating system constraints.
        System constraints are currently non-dynamical, and based on observation bounds of gym environment.
        We do not look at dry mass bounds, since those are non-existent in real greenhouse.
        """
        lowerbound = self.obs_low[:] - y[1:]
        lowerbound[lowerbound < 0] = 0
        upperbound = y[1:] - self.obs_high[:]
        upperbound[upperbound < 0] = 0
        return lowerbound, upperbound

    def terminal_state(self)-> bool:
        """
        Function that checks whether the environment is in a terminal state.
        If the crop is dead, the temperature is too high or we reached the end of the simulation, the environment is in a terminal state.
        """
        if self.x[0] < 0 or self.timestep >= self.N:
            return True
        return False

    def action_to_control(self, action: np.ndarray) -> np.ndarray:
        """
        Function that converts the action to control inputs.
        """
        return np.clip(self.prev_u + action*self.delta_u, self.min_u, self.max_u) 


    def initialise_action(self):
        """
        Sets the first previous action. Important when generating random trajectories.
        Starts at doing nothing for now.
        """
        return np.zeros((self.nu, ))

    def reset(self, seed: int = 666):
        """
        Resets environment to starting state.
        Args:
            seed    -- random seed
        Returns:
            observation -- environment state
        """
        super().reset(seed=seed)
        self.timestep = 0        
        self.terminated = False
        self.prev_u = self.initialise_action()
        self.x = np.array(self.x0)
        self.prev_dw = np.copy(self.x[0])
        self.econ_rewards = 0
        self.d = load_disturbances(self.weather_filename, self.L,self.start_day, self.h , self.Np, self.nd)
        y = self.get_y()

        # self.profits = np.zeros((self.N, ))
        # self.revenues = np.zeros((self.N, ))
        # self.co2_costs = np.zeros((self.N, ))
        # self.heating_costs = np.zeros((self.N, ))
        # self.penalties = np.zeros((self.N, len(self.penalty_coefs)))

        return self._get_obs(y), self.get_info()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_name", type=str)
    parser.add_argument("--env_id", type=str, default="LettuceGreenhouse")
    parser.add_argument("--save_name", type=str)
    parser.add_argument("--weather_filename", type=str)
    args = parser.parse_args()

    with open(f"configs/envs/{args.env_id}.yml") as f:
        env_params = yaml.safe_load(f)

    with open(f"configs/models/ppo.yml") as f:
        rl_env_params = yaml.safe_load(f)

    env = LettuceGreenhouse(
        weather_filename=args.weather_filename,
        **env_params, **rl_env_params[args.env_id]
    )

    # Load and test MPC policy with 6hr control interval
    mpc_data = pd.read_csv('data/matching-mpc/mpc-6hr.csv')
    control_colids = ["u_{0}".format(i) for i in range(env.nu)]
    controls = mpc_data[control_colids].values

    # Test MPC policy
    env.reset()

    for i in range(controls.shape[0]):
        obs, rew, terminated, truncated, info = env.step2(controls[i])
        print(f"Step {i+1} - Economic reward: {info['econ_rewards']}")
        if env.terminated:
            break
        