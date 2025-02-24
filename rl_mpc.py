import os
import argparse
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from tqdm import tqdm
import torch as th
import casadi as ca
import l4casadi as l4c


from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from mpc import MPC
from envs.lettuce_greenhouse import LettuceGreenhouse
from RL.rl_func_approximators import qvalue_fn, actor_fn
from common.noise import parametric_uncertainty
from common.rl_utils import load_rl_params
from common.utils import (
    load_disturbances, 
    compute_economic_reward,
    get_parameters,
    load_env_params,
    load_mpc_params,
    co2dens2ppm,
    vaporDens2rh
)

# Setting the discount factor to 1.0
# Make sure the loaded agents has the same discount factor.
GAMMA = 1.0
ALGS = {
    "ppo": PPO,
    "sac": SAC,
}

class RLMPC(MPC):
    def __init__(
            self,
            env_params: Dict[str, Any],
            mpc_params: Dict[str, Any],
            rl_env_params: Dict[str, Any],
            algorithm: str,
            env_path: str,
            rl_model_path: str,
            vf_path,
            use_trained_vf: bool,
            run: int
            ) -> None:
        super().__init__(**env_params, **mpc_params)
        env_norm = LettuceGreenhouse(**rl_env_params)
        env_norm = DummyVecEnv([lambda: env_norm])
        env_norm = VecNormalize(
            env_norm, 
            norm_obs = True, 
            norm_reward = False, 
            clip_obs = 10.,
            gamma=GAMMA,
        )
        env_norm = env_norm.load(env_path, env_norm)
        env_norm.training = False

        self.variance = env_norm.obs_rms.var
        self.mean = env_norm.obs_rms.mean

        # Unnormalized Environment, used for evaluation
        self.eval_env = LettuceGreenhouse(**rl_env_params)
        self.model = ALGS[algorithm].load(rl_model_path, env=self.eval_env)

        self.trained_vf = th.load(vf_path)
        self.trained_vf.eval()
        self.use_trained_vf = use_trained_vf
        self.init_l4casadi(run)

    def init_l4casadi(self, run):
        # --------------------------------
        # --- Casadi NNs and Functions ---
        # --------------------------------

        x = ca.MX.sym("x")
        mu = ca.MX.sym("mu")
        sigma_sq = ca.MX.sym("sigma_sq")
        normalized_obs = (x - mu) / ca.sqrt(sigma_sq + 1e-08)
        self.norm_obs_agent = ca.Function("normalized_obs", [x, mu, sigma_sq], [normalized_obs])
        observation = ca.MX.sym("observation")

        min_val = ca.MX.sym("min_val")
        max_val = ca.MX.sym("max_val")
        obs_norm = 10 * ((observation - min_val) / (max_val - min_val)) # TODO: Should it use parameters from SHARED.params instead of 10 and 5?
        normalizeObs_casadi = ca.Function("normalizeObs", [observation, min_val, max_val], [obs_norm])
        state_norm = 10 * ((observation - min_val) / (max_val - min_val)) - 5 # TODO: Should it use parameters from SHARED.params instead of 10 and 5?
        self.normalizeState_casadi = ca.Function("normalizeObs", [observation, min_val, max_val], [state_norm])

        # Creating casadi verions of the value functions
        vf_casadi_model = l4c.L4CasADi(self.trained_vf, device="cpu", name=f"vf_{run}")
        obs_sym_vf = ca.MX.sym("obs", 2, 1)
        # TODO: I need to transpose the observation to match the shape of the trained value function
        # since observation: (batch_size, N_features) and trained value function: (N_features, output_size)
        # vf_out = vf_casadi_model(obs_sym_vf)
        vf_out = vf_casadi_model(obs_sym_vf.T)
        vf_function = ca.Function("vf", [obs_sym_vf], [vf_out])

        # Approximated Model
        self.vf_casadi_model_approx = l4c.realtime.RealTimeL4CasADi(self.trained_vf, approximation_order=1)
        vf_casadi_approx_sym_out = self.vf_casadi_model_approx(obs_sym_vf)
        self.vf_casadi_approx_func =  ca.Function("vf_approx",[obs_sym_vf,self.vf_casadi_model_approx.get_sym_params()],[vf_casadi_approx_sym_out])

        # Qf from agent
        qf_casadi_model = l4c.L4CasADi(qvalue_fn(self.model.critic.q_networks[0]), device="cpu", name=f"qf_{run}") # Q: can we use "cuda" device? 
        obs_and_action_sym = ca.MX.sym("obs_and_a", 15, 1)
        qf_out = qf_casadi_model(obs_and_action_sym.T)
        self.qf_function = ca.Function("qf", [obs_and_action_sym], [qf_out])

        # Creating casadi version of the actor
        actor_casadi_model = l4c.L4CasADi(actor_fn(self.model.actor.latent_pi, self.model.actor.mu), device="cpu", name=f"actor_{run}")
        obs_sym = ca.MX.sym("obs", 12, 1)
        action_out = actor_casadi_model(obs_sym.T)
        self.actor_function = ca.Function("action", [obs_sym], [action_out])

        obs, _ = self.eval_env.reset()

        logs = self.unroll_actor(horizon=self.Np)

        self.rl_guess_xs = logs["x"]
        self.rl_guess_us = logs["u"]
        casadi_vf_approx_param = self.vf_casadi_model_approx.get_params(np.zeros(2))
        self.coef_size = casadi_vf_approx_param.shape[0]

    def unroll_actor(self, horizon=1, freeze=True):
        """
        Unrolls the actor function over the environment for a specified number of steps.
        Parameters:
            arg_actor_function (function): Actor function to determine actions based on normalized observations.
            freeze (bool): If True, freezes the environment"s state during the unrolling process. Default is True.
        Returns:
        dict: A dictionary containing logs of observations, states, actions, normalized observations, total reward, and reward log.
            - "obs" (numpy.ndarray): Transposed array of logged observations.
            - "x" (numpy.ndarray): Transposed array of logged states.
            - "u" (numpy.ndarray): Transposed array of logged actions.
            - "obs_norm" (numpy.ndarray): Transposed array of logged normalized observations.
            - "total_reward" (float): Total accumulated reward.
            - "reward_log" (numpy.ndarray): Array of logged rewards.
        """

        # Import

        # Define empty log variables
        log = {
            "obs":[],
            "x":[],
            "u":[],
        }
        obs_log, obs_norm_log, x_log, u_log = [],[],[],[]
        total_cost = 0
        rewards_log = []
        
        
        obs = self.eval_env._get_obs()
        # obs[0] *= 1e-3
        obs_log.append(obs)
        x_log.append(self.eval_env.get_numpy_state().ravel())

        # Freeze environment 
        if freeze:
            self.eval_env.freeze() # freeze variables

        done = False
        for i in range (0, horizon):

            # obs_norm = norm_obs(obs).toarray().squeeze(-1)
            obs_norm = self.norm_obs_agent(obs, self.mean, self.variance).toarray().ravel()
            action = self.actor_function(obs_norm).toarray().ravel()
            obs, reward, done, _,info = self.eval_env.step(action)
            # obs[0] *= 1e-3
            x = self.eval_env.get_state()
            
            total_cost += reward
            rewards_log.append(reward)
            obs_log.append(obs)
            obs_norm_log.append(self.norm_obs_agent(obs, self.mean, self.variance).toarray().ravel())
            x_log.append(x)
            u_log.append(obs[4:7])
            
        # Unfreeze environment
        if freeze:
            self.eval_env.unfreeze()

        # Store data
        log["obs"] = np.vstack(obs_log).transpose()

        log["x"] = np.vstack(x_log).transpose()
        log["u"] = np.vstack(u_log).transpose()
        log["obs_norm"] = np.vstack(obs_norm_log).transpose()
        log["total_reward"] = total_cost
        log["reward_log"] = np.array(rewards_log)

        return log

    def define_nlp(self, p):
        self.opti = ca.Opti()
        # mpc_reward = partial(reward_function, return_type = "DM")
        # get_d = partial(get_disturbance, weather_data=weather_data, start_time=start_time, Np=N+1, dt=dT)
        xs = self.opti.variable(self.nx, self.Np+1) # State Variables
        ys = self.opti.variable(self.ny, self.Np) # Output Variables
        us = self.opti.variable(self.nu, self.Np) # Control Variables
        P = self.opti.variable(6, self.Np) # Penalties for temp, CO2 and humidity. Upper and lower bounds.

        # TODO: What are these three lines of code for?
        TAYLOR_COEFS = self.opti.parameter(self.coef_size)
        A = self.opti.variable(3)
        self.opti.subject_to(-1<=(A<=1))

        time_step = self.opti.parameter(1,1)

        # Initial parameter values
        x0 = self.opti.parameter(self.nx, 1)  # Initial state
        init_u = self.opti.parameter(self.nu, 1)  # Initial control input
        ds = self.opti.parameter(self.nd, self.Np+1) # Disturbance Variables

        self.opti.set_initial(xs, self.rl_guess_xs)
        self.opti.set_initial(us, self.rl_guess_us)

        # Terminal constraints
        terminal_x = self.opti.parameter(self.nx, 1)
        terminal_u = self.opti.parameter(self.nu, 1)

        # Set parameters
        # self.opti.set_value(ds, get_d(0)) # TODO: DO we need to set ds to the disturbance values? (in mpc_opti.py we didn"t require this.)
        # self.opti.set_value(ds, ca.DM.zeros(ds.shape))
        # self.opti.set_value(x0, self.x_initial)
        # self.opti.set_value(init_u, self.u_initial)
        # self.opti.set_value(time_step, 0)
        # self.opti.set_value(terminal_x, self.rl_guess_xs[:,-1])
        # self.opti.set_value (terminal_u, self.rl_guess_us[:,-1])
        # self.opti.set_value(TAYLOR_COEFS, self.vf_casadi_model_approx.get_params(np.zeros(2)))

        self.opti.subject_to(0.95*terminal_x <= (xs[:,-1]  <= 1.05*terminal_x))
        self.opti.subject_to(0.95*terminal_u <= (us[:,-1]  <= 1.05*terminal_u))

        OBS = ca.vertcat(ys[0, -1], time_step+self.Np)
        OBS*= 1e-3
        OBS_NORM = self.opti.variable(2)
        self.opti.subject_to(
            OBS_NORM == self.normalizeState_casadi(
                OBS,
                np.array([self.x_min[0], 0]),
                np.array([self.x_max[0], self.N])
            )
        )

        # Define cost function
        J = 0

        # Set Constraints and Cost Function
        for ll in range(0, self.Np):

            # System dynamics and input constraints
            self.opti.subject_to(xs[:, ll+1] == self.F(xs[:, ll], us[:, ll], ds[:, ll], p))
            self.opti.subject_to(ys[:, ll] == self.g(xs[:, ll+1]))
            self.opti.subject_to(self.u_min <= (us[:, ll] <= self.u_max))

            # Linear penalty functions
            self.opti.subject_to(P[:, ll] >= 0)
            self.opti.subject_to(P[0, ll] >= self.lb_pen_w[0,0] * (self.y_min[1] - ys[1, ll]))
            self.opti.subject_to(P[1, ll] >= self.ub_pen_w[0,0] * (ys[1, ll] - self.y_max[1]))
            self.opti.subject_to(P[2, ll] >= self.lb_pen_w[0,1] * (self.y_min[2] - ys[2, ll]))
            self.opti.subject_to(P[3, ll] >= self.ub_pen_w[0,1] * (ys[2, ll] - self.y_max[2]))
            self.opti.subject_to(P[4, ll] >= self.lb_pen_w[0,2] * (self.y_min[3] - ys[3, ll]))
            self.opti.subject_to(P[5, ll] >= self.ub_pen_w[0,2] * (ys[3, ll] - self.y_max[3]))

            # COST FUNCTION WITH PENALTIES
            delta_dw = xs[0, ll+1] - xs[0, ll]
            J -= compute_economic_reward(delta_dw, p, self.dt, us[:,ll])
            J += (P[0, ll]+ P[1, ll]+P[2, ll]+P[3, ll]+P[4, ll]+P[5, ll])

            # Input rate constraint
            if ll < self.Np-1:
                self.opti.subject_to(-self.du_max<=(us[:,ll+1] - us[:,ll]<=self.du_max))

        # Value Function insertion
        if self.use_trained_vf:
            print("Using self trained vf")
            J_terminal = self.vf_casadi_approx_func(OBS_NORM, TAYLOR_COEFS)
        else:
            # pass
            print("using QF")
            self.opti.subject_to(OBS[0] - ys[0,-1] == 0)
            self.opti.subject_to(OBS[1:4] - ys[1:,-1] == 0)
            self.opti.subject_to(OBS[4:7] - us[:,-1] == 0)
            self.opti.subject_to(OBS[7] - time_step + self.Np == 0)
            self.opti.subject_to(OBS[8:] - ds[0:,-1] == 0)
            self.opti.subject_to(OBS_NORM == self.norm_obs_agent(OBS, self.mean, self.variance))
            J_terminal = self.qf_function(ca.vertcat(OBS_NORM, A))  
        J -= J_terminal

        # Constraints on intial state and input
        self.opti.subject_to(-self.du_max <= (us[:,0] - init_u <= self.du_max))  
        self.opti.subject_to(xs[:,0] == x0)
        self.opti.minimize(J)
        self.opti.solver('ipopt', self.nlp_opts)

    # Create the parametric solution function
        self.MPC_func = self.opti.to_function(
            "MPC_func",
            [x0, ds, init_u, time_step, terminal_x, terminal_u, xs, us, TAYLOR_COEFS],
            [us, xs, J, J_terminal, OBS_NORM],
            ["x0", "ds", "init_u", "time_step", "terminal_x", "terminal_u", "initial_X", "initial_U", "taylor_coefs"],
            ["us_opt", "xs_opt", "J", "J0", "terminal_obs"]
        )

        # Evaluate the initial cost using the MPC function

class Experiment:
    """Experiment manager to test the closed loop performance of MPC.

    Attributes:
        project_name (str): The name of the project.
        save_name (str): The name under which results will be saved.
        mpc (MPC): The MPC controller instance.
        x (np.ndarray): State trajectory.
        y (np.ndarray): Output trajectory.
        u (np.ndarray): Control input trajectory.
        d (np.ndarray): Disturbances.
        uopt (np.ndarray): Optimal control inputs.
        J (np.ndarray): Cost values.
        dJdu (np.ndarray): Gradient of the cost function.
        output (list): Optimization output.
        gradJ (np.ndarray): Gradient of the cost function.
        H (np.ndarray): Hessian of the cost function.
        step (int): Current step in the experiment.

    Methods:
        solve_nmpc(p: Dict[str, Any]) -> None:
            Solve the nonlinear MPC problem.

        update_results(us_opt: np.ndarray, Js_opt: float, sol: Dict[str, Any], step: int) -> None:
            Update the results after solving the MPC problem for a step.

        save_results() -> None:
            Save the results of the experiment to a CSV file.
    """
    def __init__(
        self,
        mpc: MPC,
        save_name: str,
        project_name: str,
        weather_filename: str,
        uncertainty_value: float,
        rng,
    ) -> None:

        self.project_name = project_name
        self.save_name = save_name
        self.mpc = mpc
        self.uncertainty_value = uncertainty_value
        self.rng = rng

        self.x = np.zeros((self.mpc.nx, self.mpc.N+1))
        self.y = np.zeros((self.mpc.nx, self.mpc.N+1))
        self.x[:, 0] = np.array(mpc.x_initial)
        self.y[:, 0] = mpc.g(self.x[:, 0]).toarray().ravel()
        self.u = np.zeros((self.mpc.nu, self.mpc.N+1))
        self.u[:, 0] = np.array(mpc.u_initial)
        self.d = load_disturbances(
            weather_filename,
            self.mpc.L,
            self.mpc.start_day,
            self.mpc.dt,
            self.mpc.Np+1,
            self.mpc.nd,
        )

        self.uopt = np.zeros((mpc.nu, mpc.Np, mpc.N+1))
        self.J = np.zeros((1, mpc.N))
        self.dJdu = np.zeros((mpc.nu*mpc.Np, mpc.N))
        self.output = []
        self.rewards = np.zeros((1, mpc.N))
        self.penalties = np.zeros((1, mpc.N))
        self.econ_rewards = np.zeros((1, mpc.N))

    def solve_nmpc(self, p) -> None:
        """Solve the nonlinear MPC problem.

        Args:
            p (Dict[str, Any]): the model parameters

        Returns:
            np.ndarray: the optimal control inputs
            float: the cost value
            np.ndarray: the constraints
            Dict[str, Any]: the optimization output
            np.ndarray: the control input changes
            np.ndarray: the gradient of the cost function
            np.ndarray: the hessian of the cost function
        """
        obs, _ = self.mpc.eval_env.reset()

        logs = self.mpc.unroll_actor(horizon=self.mpc.Np)
        casadi_vf_approx_param = self.mpc.vf_casadi_model_approx.get_params(np.zeros(2))
        coef_size = casadi_vf_approx_param.shape[0]

        us_opt, xs_opt, J_mpc_1, Jt_mpc_1, terminal_obs_1 = self.mpc.MPC_func(
            self.x[:, 0], 
            self.d[:, 0:0+self.mpc.Np+1], 
            self.mpc.u_initial,
            0,
            self.mpc.rl_guess_xs[:,-1],
            self.mpc.rl_guess_us[:,-1], 
            self.mpc.rl_guess_xs, 
            self.mpc.rl_guess_us,
            self.mpc.vf_casadi_model_approx.get_params(np.zeros(2))
        )
        self.mpc.eval_env.reset()
        for ll in range(self.mpc.N):
            if ll == 0:
                # Get very first guess and terminal constraint
                logs = self.mpc.unroll_actor(horizon=self.mpc.Np)
                rl_guess_xs = logs["x"]
                rl_guess_us = logs["u"]
                rl_guess_obs_norm = logs["obs_norm"]

                # Initial guesses for MPC
                x_guess_1 = np.copy(rl_guess_xs)
                u_guess_1 = np.copy(rl_guess_us)
                x_guess_2 = np.copy(rl_guess_xs)
                u_guess_2 = np.copy(rl_guess_us)

                TERM_POINT_1 = np.copy(logs['obs'][:,-1])
                TERM_POINT_1 = np.array([TERM_POINT_1[0], TERM_POINT_1[7]])
                TERM_POINT_1 = self.mpc.normalizeState_casadi(
                    TERM_POINT_1,
                    np.array([self.mpc.x_min[0], 0]),
                    np.array([self.mpc.x_max[0], self.mpc.N])
                )

                TERM_POINT_2 = np.copy(logs['obs'][:,-1]) 
                TERM_POINT_2 = np.array([TERM_POINT_2[0],TERM_POINT_2[7]])
                TERM_POINT_2 = self.mpc.normalizeState_casadi(
                    TERM_POINT_2,
                    np.array([self.mpc.x_min[0], 0]),
                    np.array([self.mpc.x_max[0], self.mpc.N])
                )
            else:
                # Get the optimal control value
                end_u = np.copy(us_opt[:, -1])
                end_x = np.copy(xs_opt[:, -1])
                end_xx = np.copy(xs_opt[:, -2])

                # Set environment to this state
                self.mpc.eval_env.set_env_state(end_x, end_xx, end_u, ll+self.mpc.Np-1)
                logs_1 = self.mpc.unroll_actor(horizon=1)   # TODO: why is this horizon=1?
                # Extract Trajectories from agent
                xx = logs_1['x']
                uu = logs_1['u']
                oo = logs_1['obs_norm']

                u_guess_1 = np.roll(us_opt, shift=-1, axis=1)
                u_guess_1[:,-1] = np.copy(uu[:,-1])

                x_guess_1 = np.roll(xs_opt, shift=-1,axis=1)
                x_guess_1[:,-1] = np.copy(xx[:,-1])
                breakpoint()
                TERM_POINT_1 = np.copy(logs_1['obs'][:,-1])
                TERM_POINT_1 = np.array([TERM_POINT_1[0],TERM_POINT_1[7]])
                TERM_POINT_1 = self.mpc.normalizeState_casadi(
                    TERM_POINT_1,
                    np.array([self.mpc.x_min[0], 0]),
                    np.array([self.mpc.x_max[0],
                              self.mpc.N])
                )

                # Unrolling actor from current state
                self.mpc.eval_env.set_env_state(self.x[:, ll], self.x[:, ll-1], self.u[:, ll], ll)
                logs_2 = self.mpc.unroll_actor(horizon=self.mpc.Np)

                x_guess_2 = np.copy(logs_2['x'])
                u_guess_2 = np.copy(logs_2['u'])

                TERM_POINT_2 = np.copy(logs_2['obs'][:,-1])
                TERM_POINT_2 = np.array([TERM_POINT_2[0], TERM_POINT_2[7]])
                TERM_POINT_2 = self.mpc.normalizeState_casadi(
                    TERM_POINT_2, 
                    np.array([self.mpc.x_min[0], 0]),
                    np.array([self.mpc.x_max[0], self.mpc.N])
                )

            # Getting Optimal Control Value
            coefs_1 = self.mpc.vf_casadi_model_approx.get_params(TERM_POINT_1.toarray().ravel())
            us_opt, xs_opt, J_mpc_1, Jt_mpc_1, terminal_obs_1 = self.mpc.MPC_func(
                self.x[:, ll], 
                self.d[:, ll:ll+self.mpc.Np+1], 
                self.u[:, ll],
                ll,
                x_guess_1[:,-1], 
                u_guess_1[:,-1], 
                x_guess_1, 
                u_guess_1,
                coefs_1
            )
            self.u[:, ll+1] = us_opt[:, 0].toarray().ravel()

            # Evolve State
            params = parametric_uncertainty(p, self.uncertainty_value, self.rng)

            self.x[:, ll+1] = self.mpc.F(self.x[:, ll], self.u[:, ll+1], self.d[:, ll], params).toarray().ravel()
            self.y[:, ll+1] = self.mpc.g(self.x[:, ll+1]).toarray().ravel()

            delta_dw = self.x[0, ll+1] - self.x[0, ll]
            econ_rew = compute_economic_reward(delta_dw, get_parameters(), self.mpc.dt, self.u[:, ll+1])
            penalties = self.mpc.compute_penalties(self.y[:, ll+1])
            self.update_results(us_opt, J_mpc_1, [], econ_rew, penalties, ll)


    def update_results(self, us_opt, Js_opt, sol, eco_rew, penalties, ll):
        """
        Args:
            uopt (_type_): _description_
            J (_type_): _description_
            output (_type_): _description_
            ll (_type_): _description_
        """
        self.uopt[:,:, ll] = us_opt
        self.J[:, ll] = Js_opt
        self.output.append(sol)
        self.econ_rewards[:, ll] = eco_rew
        self.penalties[:, ll] = penalties
        self.rewards[:, ll] = eco_rew - penalties


    def get_results(self, run):
        # Transform weather variables to the right units 
        self.d[1, :] = co2dens2ppm(self.d[2, :], self.d[1, :])
        self.d[3, :] = vaporDens2rh(self.d[2, :], self.d[3, :])

        # Create list of arrays to stack
        arrays = []

        # Time array
        arrays.append(self.mpc.t / 86400)
        # State arrays
        for i in range(self.x.shape[0]):
            arrays.append(self.x[i, :self.mpc.N])
            
        # Output arrays    
        for i in range(self.y.shape[0]):
            arrays.append(self.y[i, :self.mpc.N])
            
        # Input arrays
        for i in range(self.u.shape[0]):
            arrays.append(self.u[i, 1:])
            
        # Disturbance arrays
        for i in range(self.d.shape[0]):
            arrays.append(self.d[i, :self.mpc.N])
        
        # Cost and reward arrays
        arrays.append(self.J.flatten())
        arrays.append(self.econ_rewards.flatten())
        arrays.append(self.penalties.flatten()) 
        arrays.append(self.rewards.flatten())
        arrays.append(np.ones(self.mpc.N) * run)

        # Stack all arrays vertically
        return np.vstack(arrays).T

    def save_results(self, save_path):
        """
        """
        data = {}
        # transform the weather variables to the right units
        self.d[1, :] = co2dens2ppm(self.d[2, :], self.d[1, :])
        self.d[3, :] = vaporDens2rh(self.d[2, :], self.d[3, :])
        data["time"] = self.mpc.t / 86400
        for i in range(self.x.shape[0]):
            data[f"x_{i}"] = self.x[i, :self.mpc.N]
        for i in range(self.y.shape[0]):
            data[f"y_{i}"] = self.y[i, :self.mpc.N]
        for i in range(self.u.shape[0]):
            data[f"u_{i}"] = self.u[i, 1:]
        for i in range(self.d.shape[0]):
            data[f"d_{i}"] = self.d[i, :self.mpc.N]
        data["J"] = self.J.flatten()
        data["econ_rewards"] = self.econ_rewards.flatten()
        data["penalties"] = self.penalties.flatten()
        data["rewards"] = self.rewards.flatten()    

        df = pd.DataFrame(data, columns=data.keys())
        df.to_csv(f"{save_path}/{self.save_name}.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str, default="matching-thesis")
    parser.add_argument("--env_id", type=str, default="LettuceGreenhouse")
    parser.add_argument("--save_name", type=str)
    parser.add_argument("--weather_filename", default="outdoorWeatherWurGlas2014.csv", type=str)
    parser.add_argument("--algorithm", type=str, default="sac")
    parser.add_argument("--model_name", type=str, default="thesis-agent")
    parser.add_argument("--use_trained_vf", action="store_true")
    parser.add_argument("--stochastic", action="store_true")
    args = parser.parse_args()
    if args.stochastic:
        load_path = f"train_data/{args.project}/{args.algorithm}/stochastic"
    else:
        load_path = f"train_data/{args.project}/{args.algorithm}/deterministic"
    save_path = f"data/{args.project}/rlmpc"
    os.makedirs(save_path, exist_ok=True)

    rl_model_path = f"{load_path}/models/{args.model_name}/best_model.zip"
    vf_path = f"{load_path}/models/{args.model_name}/vf.zip"
    env_path = f"{load_path}/envs/{args.model_name}/best_vecnormalize.pkl"

    uncertainty_value = 0.05

    # load the config file
    env_params = load_env_params(args.env_id)
    mpc_params = load_mpc_params(args.env_id)
    mpc_params["uncertainty_value"] = uncertainty_value

    # load the RL parameters
    hyperparameters, rl_env_params = load_rl_params(args.env_id, args.algorithm)
    rl_env_params.update(env_params)
    rl_env_params["uncertainty_value"] = uncertainty_value

    # define the RL-MPC
    p = get_parameters()
    rl_mpc = RLMPC(
        env_params, 
        mpc_params, 
        rl_env_params,
        args.algorithm,
        env_path,
        rl_model_path,
        vf_path,
        use_trained_vf=args.use_trained_vf,
        run=0
    )
    rl_mpc.define_nlp(p)

    # run the experiment
    rng = np.random.default_rng(0)
    exp = Experiment(rl_mpc, args.save_name, args.project, args.weather_filename, uncertainty_value, rng)
    exp.solve_nmpc(p)
    exp.save_results(save_path)
