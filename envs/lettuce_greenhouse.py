"""
Greenhouse dyanmics modelled in OpenAI Gym environment.
The controller for this environment control the valves of the greenhouse.
That regulates amount of heating (W/m2) and CO_2 into the greenhouse.

For now used modelled as a temperature reference tracking problem.
""" 
from typing import Dict, List, Optional, Tuple
from copy import copy, deepcopy
import argparse
import yaml
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from envs.observations import StandardObservations
from common.noise import parametric_uncertainty
from common.utils import *

observations = {"StandardObservations": StandardObservations}

class LettuceGreenhouse(gym.Env):
    """
    This class implements the original lettuce greenhouse model by Eldert van Henten (1994),
    in a gymnasium environment. Such that off-the-shelf RL libraries can be applied to control this model.
    It solves the model equations using the numerical Runge-Kutta-4 method.

    Arguments:
        nx                  -- number of greenhouse states
        ny                  -- number of greenhouse measurements
        nd                  -- number of disturbance (weather variables)
        nu                  -- number of control inputs
        h                   -- control rate of the system in minutes
        n_days              -- number of simulation days
        Np                  -- number of future weather predictions to use
        start_day           -- start day of simulation
        x0                  -- initial state of the greenhouse
        u0                  -- initial control inputs
        constraints         -- constraints of the environment
        weather_filename    -- path to the weather data
        lb_pen_w            -- lower bound penalty weights
        ub_pen_w            -- upper bound penalty weights
        obs_module          -- observation module
        obs_names           -- observation names
    """
    def __init__(self,
        nx: int,                # number of greenhouse states
        ny: int,                # number of greenhouse measurements
        nd: int,                # number of disturbance (weather variables)
        nu: int,                # number of control inputs
        dt: float,              # control rate of the system in minutes
        n_days: int,            # simulation days
        Np: int,                # number of future weather predictions to use
        start_day: int,         # start day of simulation
        x0: List[float],        # initial state of the greenhouse
        u0: List[float],        # initial control inputs
        constraints: Dict[str, float], # constraints of the environment
        weather_filename: str, # path to the weather data
        lb_pen_w: List[float],
        ub_pen_w: List[float],
        obs_module: str,
        obs_names: List[str],
        uncertainty_scale: float = 0.0
        ):
        super(LettuceGreenhouse, self).__init__()
        self.weather_filename = weather_filename

        # simulation parameters
        self.x0 = x0
        self.u0 = u0
        self.c = 86400                  # conversion between days and seconds               
        self.nx = nx
        self.nu = nu
        self.ny = ny
        self.nd = nd
        self.dt = dt
        self.n_days = n_days
        self.L = n_days * self.c
        self.start_day = start_day
        self.N = int(self.L//self.dt)     # number of steps to take during episode
        # prediction horizon for weather forecast for observation space
        self.Np = Np
        
        self.uncertainty_scale = uncertainty_scale

        # greenhouse model parameters
        self.p = get_parameters()

        # define the dynamical greenhouse model:
        self.F, self.g = define_model(self.dt)

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

        self.x_min = np.array(
            [
                constraints["W_min"],
                constraints["state_co2_min"],
                constraints["state_temp_min"],
                constraints["state_vp_min"]
            ],
            dtype=np.float32
        )
        self.x_max = np.array(
            [
                constraints["W_max"],
                constraints["state_co2_max"],
                constraints["state_temp_max"],
                constraints["state_vp_max"]
            ],
            dtype=np.float32
        )

        # min and max control inputs and 
        self.u_min = np.array(
            [constraints["co2_supply_min"],
            constraints["vent_min"],
            constraints["heat_min"]], 
            dtype=np.float32
        )
        self.u_max = np.array(
            [constraints["co2_supply_max"],
            constraints["vent_max"],
            constraints["heat_max"],], 
            dtype=np.float32
        )
        # previous control and allowed change control
        self.delta_u = self.u_max/10

        # weights for penalty function
        self.lb_pen_w = np.expand_dims(lb_pen_w, axis=0)
        self.ub_pen_w = np.expand_dims(ub_pen_w, axis=0)

    def seed(self, seed):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def get_state(self):
        return self.x.toarray().flatten()

    def get_numpy_state(self):
        """
        The first state of the environment is not a Casadi DM object.
        Therefore, we don't need to convert it to a numpy array.
        """
        return self.x

    def get_y(self):
        return self.g(np.copy(self.x)).toarray().flatten()

    def get_d(self):
        return transform_disturbances(np.copy(self.d[:, self.timestep]))

    def set_env_state(self, x, x_prev, u, timestep):
        self.x = np.copy(x)
        self.x_prev = np.copy(x_prev)
        self.y = self.g(x).toarray().ravel()
        self.y_prev = self.g(x_prev).toarray().ravel()
        y = np.copy(self.y)

        # if self.use_growth_dif:
        #     y[0] = y[0]  - self.y_prev[0] 

        self.timestep = timestep
        self.u = np.copy(u).ravel()
        d = self.get_d().ravel()     
        self.obs = np.concatenate([y, self.u, [self.timestep], d], dtype=np.float32)


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
        self.u = self.action_to_control(action)

        self.x_prev = np.copy(self.x)
        self.y_prev = np.copy(self.y) # Store previous system outputs

        params = parametric_uncertainty(self.p, self.uncertainty_scale, self._np_random)

        # transition state next state given action and observe environment
        self.x = self.F(self.x, self.u, self.d[:, self.timestep], params)
        self.y = self.get_y()
        self.timestep += 1

        # terminate if state is terminal or crops died
        if self.terminal_state():
            self.done = True

        # calculate reward
        reward = self._get_reward(self.y, self.u)

        self.prev_dw = np.copy(self.x[0])

        info = self.get_info(self.u)
        self.obs = self._get_obs()
        return (
            self.obs,
            reward, 
            self.done, 
            False,
            info
            )

    def _get_obs(self) -> np.ndarray:
        """
        Function that returns the observation space of the environment.
        """
        return self.observation_module.compute_obs(self.y, self.u, self.d[:, self.timestep], self.timestep)

    def get_info(self, u):
        info =  {"econ_rewards": self.econ_rewards}
        # info = {""}
        info["EPI"] = self.econ_rewards
        info["temp_violation"] = self.temp_violation
        info["rh_violation"] = self.rh_violation
        info["co2_violation"] = self.co2_violation
        info["controls"] = u
        return info

    def _get_reward(self, y, u):
        """
        Reward function that calculates the reward of the environment.
        """
        delta_dw  = self.x[0] - self.prev_dw
        self.econ_rewards = float(compute_economic_reward(delta_dw, self.p, self.dt, u))
        penalties = self._compute_penalty(self.y)
        return self.econ_rewards - penalties

    def _compute_penalty(self, y: np.ndarray) -> np.ndarray:
        """
        Penalty function that calculates the penalty of the environment.
        """
        lb_violation, ub_violation = self.constraint_violation(self.y)
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
        self.co2_violation = lowerbound[0] + upperbound[0]
        self.temp_violation = lowerbound[1] + upperbound[1]
        self.rh_violation = lowerbound[2] + upperbound[2]
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
        return np.clip(self.u + action*self.delta_u, self.u_min, self.u_max) 


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
        self.done = False

        self.u = np.copy(self.u0)
        self.x = np.array(self.x0)
        self.x_prev = np.copy(self.x0)
        self.prev_dw = np.copy(self.x[0])

        self.d = load_disturbances(self.weather_filename, self.L, self.start_day, self.dt , self.Np*2, self.nd)
        self.y = self.get_y()
        self.y_prev = np.copy(self.y)

        self.econ_rewards = 0
        self.co2_violation = 0.
        self.temp_violation = 0.
        self.rh_violation = 0.
        self.obs = self._get_obs()
        return self.obs, self.get_info(self.u)

    def sample_obs(self, std=0.5):
        """agent_state = (y1,y2,y3,y4,u1,u2,u3,k, d1,d2,d3,d4)
        """
        d = self.get_d()

        random_x = np.zeros(self.nx,)    
        random_u = np.zeros(self.nu,)

        # Drymass   
        # random_x[0] =  np.random.normal(loc=self.x.ravel()[0], scale = std*self.x.ravel()[0])
        bounds_min = self.x.ravel()[0]*(1-0.8) - 0.01
        bounds_max = self.x.ravel()[0]*(1+0.7) + 0.01
        
        # bounds_min = self.x.ravel()[0]*(1-0.1) - 0.01
        # bounds_max = self.x.ravel()[0]*(1+0.1) + 0.01
        bounds_min = np.maximum(bounds_min,self.x_min[0])
        bounds_max = np.minimum(bounds_max,self.x_max[0])
        random_x[0] =  np.random.uniform(bounds_min,bounds_max)

        # C02
        random_x[1] =  np.random.uniform(co2ppm2dens(random_x[2],400),co2ppm2dens(random_x[2],1800))
    
        # Temp
        max_temp_traj = 30
        min_temp_traj = 7
        random_x[2] =  np.random.uniform(min_temp_traj,max_temp_traj)      
        
        # Hum
        random_x[3] =  np.random.uniform(rh2vaporDens(random_x[2],50),rh2vaporDens(random_x[2],100))
          
        random_x = np.clip(random_x, self.x_min, self.x_max)
        random_y = self.g(random_x).toarray().ravel() 

        bounds_min = self.u.ravel()[0] - (std)*(self.u_max[0]-self.u_min[0])/2
        bounds_max = self.u.ravel()[0] + (std)*(self.u_max[0]-self.u_min[0])/2
        bounds_min = np.maximum(bounds_min, self.u_min[0])
        bounds_max = np.minimum(bounds_max, self.u_max[0])  
        random_u[0] = np.random.uniform(bounds_min, bounds_max)
    
        bounds_min = self.u.ravel()[1]- (std)*(self.u_max[1]-self.u_min[1])/2
        bounds_max = self.u.ravel()[1]+ (std)*(self.u_max[1]-self.u_min[1])/2
        bounds_min = np.maximum(bounds_min, self.u_min[1])
        bounds_max = np.minimum(bounds_max, self.u_max[1])  
        random_u[1] = np.random.uniform(bounds_min, bounds_max)

        # WHY IS THIS PIECE OF CODE COMMENTED OUT?
        # bounds_min = self.u.ravel()[2]- (std)*(self.u_max[2]-self.u_min[2])/2
        # bounds_max = self.u.ravel()[2]+ (std)*(self.u_max[2]-self.u_min[2])/2
        # bounds_min = np.maximum(bounds_min, self.u_min[2])
        # bounds_max = np.minimum(bounds_max, self.u_max[2])  
        # random_u[2] = np.random.uniform(bounds_min,bounds_max)

        random_u = np.random.uniform(self.u_min, self.u_max)
        # random_u = np.random.uniform(loc = self.u.ravel(), scale = std*self.u.ravel())
        random_u = np.clip(random_u, self.u_min, self.u_max)

        # THIS WAS TESTED AS FEATURE FOR THE VALUE FUNCTION
        # if self.use_growth_dif:
        #     random_y[0] = random_y[0]  - self.y_prev[0]  
        
        random_obs =  np.concatenate([random_y, random_u, [self.timestep], d.ravel()], dtype=np.float32)

        # WHY ARE THE STATE VARIABLES COPIED HERE?
        self.x = np.copy(random_x)
        self.y = self.g(random_x).toarray().ravel()  
        self.obs = np.copy(random_obs)
        self.u = np.copy(random_u)

        return random_obs, random_x, random_u

    def freeze(self):
        self.freeze_k           = deepcopy(self.timestep)
        self.freeze_x           = deepcopy(self.x)
        self.freeze_observation = deepcopy(self.obs)
        self.freeze_y           = deepcopy(self.y)
        self.freeze_x_prev      = deepcopy(self.x_prev)
        self.freeze_y_prev      = deepcopy(self.y_prev)
        self.freeze_done        = deepcopy(self.done)
        self.freeze_u           = deepcopy(self.u)
        
    def unfreeze(self):
        self.timestep       = deepcopy(self.freeze_k)
        self.x              = deepcopy(self.freeze_x)
        self.x_prev         = deepcopy(self.freeze_x_prev)
        self.obs            = deepcopy(self.freeze_observation)
        self.y              = deepcopy(self.freeze_y)
        self.y_prev         = deepcopy(self.freeze_y_prev)
        self.done           = deepcopy(self.freeze_done)
        self.u              = deepcopy(self.freeze_u)


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--project_name", type=str)
#     parser.add_argument("--env_id", type=str, default="LettuceGreenhouse")
#     parser.add_argument("--save_name", type=str)
#     parser.add_argument("--weather_filename", type=str)
#     args = parser.parse_args()

#     with open(f"configs/envs/{args.env_id}.yml") as f:
#         env_params = yaml.safe_load(f)

#     with open(f"configs/models/ppo.yml") as f:
#         rl_env_params = yaml.safe_load(f)

#     env = LettuceGreenhouse(
#         weather_filename=args.weather_filename,
#         **env_params, **rl_env_params[args.env_id]
#     )

#     # Load and test MPC policy with 6hr control interval
#     mpc_data = pd.read_csv("data/matching-mpc/mpc-6hr.csv")
#     control_colids = ["u_{0}".format(i) for i in range(env.nu)]
#     controls = mpc_data[control_colids].values

#     # Test MPC policy
#     env.reset()

#     for i in range(controls.shape[0]):
#         obs, rew, terminated, truncated, info = env.step2(controls[i])
#         print(f"Step {i+1} - Economic reward: {info["econ_rewards"]}")
#         if env.terminated:
#             break
