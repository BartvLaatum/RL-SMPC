import os
import argparse
from typing import Any, Dict, List, Tuple
from tqdm import tqdm
from time import time

import numpy as np
import pandas as pd
from tqdm import tqdm
import torch as th
import casadi as ca
import l4casadi as l4c


from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from smpc import SMPC
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

class RLSMPC(SMPC):
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
        # obs_norm = 10 * ((observation - min_val) / (max_val - min_val)) # TODO: Should it use parameters from SHARED.params instead of 10 and 5?
        # normalizeObs_casadi = ca.Function("normalizeObs", [observation, min_val, max_val], [obs_norm])
        state_norm = 10 * ((observation - min_val) / (max_val - min_val)) - 5 # TODO: Should it use parameters from SHARED.params instead of 10 and 5?
        self.normalizeState_casadi = ca.Function("normalizeObs", [observation, min_val, max_val], [state_norm])

        # Creating casadi verions of the value functions
        # vf_casadi_model = l4c.L4CasADi(self.trained_vf, device="cpu", name=f"vf_{run}")
        obs_sym_vf = ca.MX.sym("obs", 2, 1)

        # TODO: I need to transpose the observation to match the shape of the trained value function
        # since observation: (batch_size, N_features) and trained value function: (N_features, output_size)
        # vf_out = vf_casadi_model(obs_sym_vf)
        # vf_out = vf_casadi_model(obs_sym_vf.T)
        # vf_function = ca.Function("vf", [obs_sym_vf], [vf_out])

        # Create taylor approximation model of the value function
        self.vf_casadi_model_approx = l4c.realtime.RealTimeL4CasADi(self.trained_vf, approximation_order=1)
        vf_casadi_approx_sym_out = self.vf_casadi_model_approx(obs_sym_vf)
        self.vf_casadi_approx_func =  ca.Function(
            "vf_approx",
            [obs_sym_vf, self.vf_casadi_model_approx.get_sym_params()],
            [vf_casadi_approx_sym_out]
        )

        # Qf from agent
        # qf_casadi_model = l4c.L4CasADi(qvalue_fn(self.model.critic.q_networks[0]), device="cpu", name=f"qf_{run}") # Q: can we use "cuda" device? 
        # obs_and_action_sym = ca.MX.sym("obs_and_a", 15, 1)
        # qf_out = qf_casadi_model(obs_and_action_sym.T)
        # # NOTE: THIS ONE IS ONLY USED WHEN WE USE Q-VALUE FUNCTION RATHER THAN THE VF-FUNCTION (ie. when PPO is used)
        # self.qf_function = ca.Function("qf", [obs_and_action_sym], [qf_out])

        # Creating casadi version of the actor
        actor_casadi_model = l4c.L4CasADi(actor_fn(self.model.actor.latent_pi, self.model.actor.mu), device="cpu", name=f"actor_{run}")
        obs_sym = ca.MX.sym("obs_sym", 12, 1)
        action_out = actor_casadi_model(obs_sym.T)
        self.actor_function = ca.Function(
            "action",
            [obs_sym],
            [action_out]
        )

        # Define the Jacobian of action_out wrt first 4 obs components (the system output variables)
        y_sym = ca.MX.sym("y_sym", 4, 1)
        u_sym = ca.MX.sym("u_sym", 3, 1)
        timestep = ca.MX.sym("timestep", 1)
        d_sym = ca.MX.sym("d_sym", 4, 1)

        # required for the first-order implementation
        obs_sym_chain_rule = self.h(y_sym, u_sym, timestep, d_sym)

        # Actor output
        action_out_chain_rule = actor_casadi_model(obs_sym_chain_rule.T)

        # Build the action Function with *all* relevant inputs
        self.actor_function_chain_rule = ca.Function(
            "action",
            [y_sym, u_sym, timestep, d_sym],
            [action_out_chain_rule]
        )

        # Jacobian wrt y_sym:
        J_x = ca.jacobian(action_out_chain_rule, y_sym)

        self.jac_actor_wrt_state = ca.Function(
            "jac_actor_wrt_state",
            [y_sym, u_sym, timestep, d_sym],
            [J_x]
        )

        # Jacobian wrt u_sym:
        J_u = ca.jacobian(action_out_chain_rule, u_sym)
        self.jac_actor_wrt_input = ca.Function(
            "jac_actor_wrt_state",
            [y_sym, u_sym, timestep, d_sym],
            [J_u]
        )

        casadi_vf_approx_param = self.vf_casadi_model_approx.get_params(np.zeros(2))
        self.coef_size = casadi_vf_approx_param.shape[0]

    def h(self, x, u, d, timestep):
        return ca.vertcat(x, u, timestep, d)
 

    def unroll_actor(self, p_i_samples=None, horizon=1, freeze=True,):
        """
        Unrolls the actor function over the environment for a specified number of steps.
        Parameters:
            p_i_samples (np.ndarray): One sampled trajectory (i) of the parameteric uncertainty.
            freeze (bool): If True, freezes the environment"s state during the unrolling process. Default is True.
            horizon (int): Number of steps to unroll the actor function. Default is 1.

        Returns:
        dict: A dictionary containing logs of observations, states, actions, normalized observations, total reward, and reward log.
            - "obs" (numpy.ndarray): Transposed array of logged observations.
            - "x" (numpy.ndarray): Transposed array of logged states.
            - "u" (numpy.ndarray): Transposed array of logged actions.
            - "obs_norm" (numpy.ndarray): Transposed array of logged normalized observations.
            - "total_reward" (float): Total accumulated reward.
            - "reward_log" (numpy.ndarray): Array of logged rewards.
        """
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
        obs_log.append(obs)
        obs_norm_log.append(self.norm_obs_agent(obs, self.mean, self.variance).toarray().ravel())

        x_log.append(self.eval_env.get_numpy_state().ravel())

        # Freeze curren state of the environment 
        if freeze:
            self.eval_env.freeze()


        done = False
        for i in range (0, horizon):
            pk = p_i_samples[i] if p_i_samples is not None else None
            obs_norm = self.norm_obs_agent(obs, self.mean, self.variance).toarray().ravel()

            action = self.actor_function(obs_norm).toarray().ravel()
            obs, reward, done, _,info = self.eval_env.step(action, pk)
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
        log["obs"] = np.vstack(obs_log).T

        log["x"] = np.vstack(x_log).T
        log["u"] = np.vstack(u_log).T
        log["obs_norm"] = np.vstack(obs_norm_log).T
        log["total_reward"] = total_cost
        log["reward_log"] = np.array(rewards_log)

        return log

    def solve_ocp(
            self, 
            x0, 
            u0, 
            ds, 
            timestep,
            theta_init,
            xk_samples, 
            terminal_xs, 
            uk_samples, 
            p_samples, 
            taylor_coefficients
        ):
        """
        Sets solver values for the OCP.
        This function initializes various parameters and values required for the scenario-based
        Stochastic Model Predictive Control (MPC) solver, including initial states, control inputs,
        scenario trajectories, and Taylor expansion coefficients.
        Additionally, it provides an initial guess for the optimization variables.
        Args:
            x0 (numpy.ndarray): Initial state vector
            u0 (numpy.ndarray): Initial control input vector
            ds (float): Step size or discretization parameter
            k (int): Current timestep
            xk_samples (list): List of state trajectories for each scenario
            terminal_xs (list): List of terminal states for each scenario
            uk_samples (list): List of control input trajectories for each scenario
            p_samples (list): List of parameter samples for each scenario
            taylor_coefficients (list): List of Taylor expansion coefficients for each scenario
        Returns:
            (xs_opt, ys_opt, theta_opt, J, exit_message)
        Note:
            This function assumes that the OCP (self.opti) has been properly
            initialized with the corresponding decision variables (xs_list, terminal_xs,
            u_samples, p_samples, TAYLOR_COEFS_samples).
        """
        # Set the OCP parameter values
        self.opti.set_value(self.x0, x0)        
        self.opti.set_value(self.init_u, u0)
        self.opti.set_value(self.timestep, timestep)
        self.opti.set_value(self.ds, ds)
        
        # Set initial guess for state trajectories for each scenario
        for i, xs in enumerate(self.xs_list):
            self.opti.set_initial(xs, xk_samples[i])
        
        self.opti.set_initial(self.theta, theta_init)

        # Set terminal state values for each scenario
        for i, xterminal in enumerate(self.terminal_xs):
            self.opti.set_value(xterminal, terminal_xs[i])
        
        # Set sampled RL control input values for each scenario
        for i, us in enumerate(self.u_samples):
            self.opti.set_value(us, uk_samples[i])
        
        # Set parametric uncertainty values for each scenario
        for i, p_sample in enumerate(self.p_samples):
            self.opti.set_value(p_sample, p_samples[i].T)
        
        # Set Taylor expansion coefficients for each scenario
        for i, taylor_coeffs in enumerate(self.TAYLOR_COEFS_samples):
            self.opti.set_value(taylor_coeffs, taylor_coefficients[i])

        start_time = time()
        # Solve the OCP for the given scenario
        try:
            solution = self.opti.solve()
        except RuntimeError as err:
            # Recover from the failed solve: you might use the last iterate available via opti.debug
            print("Solver failed with error:", err)
            solution = self.opti.debug  # Returns the current iterate even if not converged
        solver_time = time() - start_time

        stats = solution.stats()
        exit_message = stats['return_status']
        xs_opt = [solution.value(self.xs_list[i]) for i in range(self.Ns)]
        ys_opt = [solution.value(self.ys_list[i]) for i in range(self.Ns)]
        theta_opt = solution.value(self.theta)
        J = solution.value(self.opti.f)

        return xs_opt, ys_opt, theta_opt, J, solver_time, exit_message

    def define_zero_order_snlp(self, p: np.ndarray) -> None:
        """
        Define the zero-order stochastic nonlinear programming (SNLP) problem.

        This function sets up the optimization problem for the zero-order stochastic MPC 
        using CasADi's Opti stack. It defines decision variables, constraints, and the 
        cost function for multiple scenarios.

        The zero-order approach uses direct perturbation of the nominal control input 
        generated by the RL policy, without considering gradient information.

        Args:
            p (np.ndarray): Array of model parameters used in the system dynamics 
                and cost function calculations.

        Returns:
            None: The function updates the class attributes:
                - self.opti: CasADi optimization problem
                - self.MPC_func: CasADi function for solving the MPC problem

        Note:
            The function creates an optimization problem that:
            - Minimizes the average cost across all scenarios
            - Enforces system dynamics and input constraints
            - Includes terminal constraints and value function approximation
            - Uses the RL policy to generate nominal trajectories
        """
        self.opti = ca.Opti()
        num_penalties = 6

        # Decision Variables (Control inputs, slack variables, states, outputs)
        self.theta = self.opti.variable(self.nu, self.Np)  # Control inputs (nu x Np)
        self.xs_list = [self.opti.variable(self.nx, self.Np+1) for _ in range(self.Ns)]
        self.ys_list = [self.opti.variable(self.ny, self.Np) for _ in range(self.Ns)]
        Ps = [self.opti.variable(num_penalties, self.Np) for _ in range(self.Ns)]

        self.TAYLOR_COEFS_samples = [self.opti.parameter(self.coef_size) for _ in range(self.Ns)]
        # A = self.opti.variable(3)         # NOTE these are only necessary when using PPO as value function
        # self.opti.subject_to(-1<=(A<=1))

        self.timestep = self.opti.parameter(1,1)
        self.u_samples = [self.opti.parameter(self.nu, self.Np) for _ in range(self.Ns)]
        self.p_samples = [self.opti.parameter(p.shape[0], self.Np) for _ in range(self.Ns)]

        # Initial parameter values
        self.x0 = self.opti.parameter(self.nx, 1)  # Initial state
        self.init_u = self.opti.parameter(self.nu, 1)  # Initial control input
        self.ds = self.opti.parameter(self.nd, self.Np+1) # Disturbance Variables

        # Terminal constraint for the state
        self.terminal_xs = [self.opti.parameter(self.nx, 1) for _ in range(self.Ns)]

        for i, xs in enumerate(self.xs_list):
            self.opti.subject_to(0.95*self.terminal_xs[i] <= (xs[:,-1]  <= 1.05*self.terminal_xs[i]))


        # Define cost function
        J = 0

        # Set Constraints and Cost Function
        for i in range(self.Ns):
            xs = self.xs_list[i]
            ys = self.ys_list[i]
            ps = self.p_samples[i]
            us = self.u_samples[i]
            TAYLOR_COEFS = self.TAYLOR_COEFS_samples[i]
            P = Ps[i]

            self.opti.subject_to(xs[:,0] == self.x0)
            
            OBS = ca.vertcat(ys[0, -1], self.timestep+self.Np)
            OBS_NORM = self.opti.variable(2)
            self.opti.subject_to(
                OBS_NORM == self.normalizeState_casadi(
                    OBS,
                    np.array([self.x_min[0], 0]),
                    np.array([self.x_max[0], self.N])
                )
            )

            for ll in range(0, self.Np):
                pk = ps[:, ll]
                uk = us[:, ll] + self.theta[:, ll]

                # System dynamics and input constraints
                self.opti.subject_to(xs[:, ll+1] == self.F(xs[:, ll], uk, self.ds[:, ll], pk))
                self.opti.subject_to(ys[:, ll] == self.g(xs[:, ll+1]))
                self.opti.subject_to(self.u_min <= (uk <= self.u_max))

                # Linear penalty functions
                self.opti.subject_to(P[:, ll] >= 0)
                self.opti.subject_to(P[0, ll] >= self.lb_pen_w[0,0] * (self.y_min[1] - ys[1, ll]))
                self.opti.subject_to(P[1, ll] >= self.ub_pen_w[0,0] * (ys[1, ll] - self.y_max[1]))
                self.opti.subject_to(P[2, ll] >= self.lb_pen_w[0,1] * (self.y_min[2] - ys[2, ll]))
                self.opti.subject_to(P[3, ll] >= self.ub_pen_w[0,1] * (ys[2, ll] - self.y_max[2]))
                self.opti.subject_to(P[4, ll] >= self.lb_pen_w[0,2] * (self.y_min[3] - ys[3, ll]))
                self.opti.subject_to(P[5, ll] >= self.ub_pen_w[0,2] * (ys[3, ll] - (self.y_max[3]-2.0)))

                # COST FUNCTION WITH PENALTIES
                delta_dw = xs[0, ll+1] - xs[0, ll]
                J -= compute_economic_reward(delta_dw, p, self.dt, uk)
                J += (P[0, ll]+ P[1, ll]+P[2, ll]+P[3, ll]+P[4, ll]+P[5, ll])

                # Input rate constraint
                if ll < self.Np-1:
                    self.opti.subject_to(-self.du_max <= ((us[:, ll+1] + self.theta[:, ll+1]) - uk <=self.du_max))

            # Value Function insertion
            if self.use_trained_vf:
                J_terminal = self.vf_casadi_approx_func(OBS_NORM, TAYLOR_COEFS)
            else:
                J_terminal = 0
            #     # pass
            #     print("using QF")
            #     self.opti.subject_to(OBS[0] - ys[0,-1] == 0)
            #     self.opti.subject_to(OBS[1:4] - ys[1:,-1] == 0)
            #     self.opti.subject_to(OBS[4:7] - us[:,-1] == 0)
            #     self.opti.subject_to(OBS[7] - self.timestep + self.Np == 0)
            #     self.opti.subject_to(OBS[8:] - self.ds[0:,-1] == 0)
            #     self.opti.subject_to(OBS_NORM == self.norm_obs_agent(OBS, self.mean, self.variance))
            #     J_terminal = self.qf_function(ca.vertcat(OBS_NORM, A))  
            J -= J_terminal

            # Constraints on intial state and input
            self.opti.subject_to(-self.du_max <= ((us[:, 0] + self.theta[:, 0]) - self.init_u <= self.du_max))  

        J = J / self.Ns

        self.opti.minimize(J)
        self.opti.solver('ipopt', self.nlp_opts)

    def compute_control_input(self, xs, ll, us, os_y, os_u, theta, jac_y_full, jac_input_full, u_prev):
        """
        Compute the control input using first-order Taylor expansion of the RL policy.

        This function computes the control input by combining the nominal input from the 
        RL policy with corrections based on the first-order Taylor expansion around the 
        nominal trajectory. It normalizes observations and applies the Jacobian matrices 
        to compute state and input-dependent corrections.

        Args:
            xs (casadi.MX): State trajectory matrix of shape (nx, Np+1)
            ll (int): Current time step in the prediction horizon
            us (casadi.MX): Nominal control inputs from RL policy of shape (nu, Np)
            os_y (casadi.MX): Normalized output observations of shape (ny, Np)
            os_u (casadi.MX): Normalized input observations of shape (nu, Np)
            theta (casadi.MX): Optimization variables for control correction of shape (nu, Np)
            jac_y_full (casadi.MX): Full Jacobian matrix w.r.t. outputs of shape (nu, ny*Np)
            jac_input_full (casadi.MX): Full Jacobian matrix w.r.t. inputs of shape (nu, nu*Np)
            u_prev (casadi.MX): Previous control input of shape (nu, 1)

        Returns:
            casadi.MX: Computed control input of shape (nu, 1) combining nominal input 
                and first-order corrections

        Note:
            The function performs these steps:
            1. Normalizes current output and input observations
            2. Extracts relevant portions of the Jacobian matrices
            3. Computes control input using first-order Taylor expansion
        """
        # Get the full observation vector from the state
        obs_y = self.g(xs[:, ll])
        obs_norm_y = self.opti.variable(self.ny)
        self.opti.subject_to(
            obs_norm_y == self.norm_obs_agent(obs_y, self.mean[:self.ny], self.variance[:self.ny])
        )

        obs_norm_u = self.opti.variable(self.nu)
        self.opti.subject_to(
            obs_norm_u == self.norm_obs_agent(u_prev, self.mean[self.ny:self.ny+self.nu], self.variance[self.ny:self.ny+self.nu])
        )

        # Make sure the jacobian matrix has the right dimensions
        jac_y_matrix = jac_y_full[:, self.ny*ll:self.ny*(ll+1)]
        jac_input_matrix = jac_input_full[:, self.nu*ll:self.nu*(ll+1)]

        # Compute control input
        uk = us[:, ll] + ca.mtimes(jac_y_matrix, (os_y[:, ll] - obs_norm_y)) + \
            ca.mtimes(jac_input_matrix, os_u[:, ll] - obs_norm_u) + theta[:,ll]
        
        return uk

    def define_first_order_snlp(self, p: np.ndarray) -> None:
        """
        Define the first-order stochastic nonlinear programming (SNLP) problem.

        This function sets up the optimization problem for the first-order stochastic MPC 
        using CasADi's Opti stack. It defines decision variables, constraints, and the 
        cost function for multiple scenarios, incorporating first-order Taylor expansions 
        to adjust control inputs based on state and input observations.

        The first-order approach uses gradient information to refine the nominal control 
        input generated by the RL policy, allowing for more precise adjustments.

        Args:
            p (np.ndarray): Array of model parameters used in the system dynamics 
                and cost function calculations.

        Returns:
            None: The function updates the class attributes:
                - self.opti: CasADi optimization problem
                - self.MPC_func: CasADi function for solving the MPC problem

        Note:
            The function creates an optimization problem that:
            - Minimizes the average cost across all scenarios
            - Enforces system dynamics and input constraints
            - Includes terminal constraints and value function approximation
            - Uses the RL policy to generate nominal trajectories
            - Incorporates first-order corrections to control inputs
        """
        self.opti = ca.Opti()
        num_penalties = 6

        # Decision Variables (Theta, slack variables, states, outputs)
        theta = self.opti.variable(self.nu, self.Np)  # theta (nu x Np)
        xs_list = [self.opti.variable(self.nx, self.Np+1) for _ in range(self.Ns)]
        ys_list = [self.opti.variable(self.ny, self.Np) for _ in range(self.Ns)]
        Ps = [self.opti.variable(num_penalties, self.Np) for _ in range(self.Ns)]
        TAYLOR_COEFS_samples = [self.opti.parameter(self.coef_size) for _ in range(self.Ns)]

        # A = self.opti.variable(3)
        # self.opti.subject_to(-1<=(A<=1))

        # Parameters
        timestep = self.opti.parameter(1,1)
        u_samples = [self.opti.parameter(self.nu, self.Np) for _ in range(self.Ns)]
        p_samples = [self.opti.parameter(p.shape[0], self.Np) for _ in range(self.Ns)]
        obs_norm_y_samples = [self.opti.parameter(self.ny, self.Np) for _ in range(self.Ns)]
        obs_norm_input_samples = [self.opti.parameter(self.nu, self.Np) for _ in range(self.Ns)]
        jac_obs_y_samples = [self.opti.parameter(self.nu, self.ny * self.Np) for _ in range(self.Ns)]
        jac_obs_input_samples = [self.opti.parameter(self.nu, self.nu * self.Np) for _ in range(self.Ns)]

        # Initial parameter values
        x0 = self.opti.parameter(self.nx, 1)  # Initial state
        init_u = self.opti.parameter(self.nu, 1)  # Initial control input
        ds = self.opti.parameter(self.nd, self.Np+1) # Disturbance Variables

        # Terminal constraint fpr the state
        terminal_xs = [self.opti.parameter(self.nx, 1) for _ in range(self.Ns)]

        for i, xs in enumerate(xs_list):
            self.opti.subject_to(0.95*terminal_xs[i] <= (xs[:,-1]  <= 1.05*terminal_xs[i]))

        # Initialise cost function
        J = 0

        # Set Constraints and Cost Function
        for i in range(self.Ns):
            xs = xs_list[i]
            ys = ys_list[i]
            ps = p_samples[i]
            us = u_samples[i]
            os_y = obs_norm_y_samples[i]
            os_u = obs_norm_input_samples[i]
            jac_y_full = jac_obs_y_samples[i]  # jacobian: shape (3, ny*Np)
            jac_input_full = jac_obs_input_samples[i]  # jacobian: shape (3, nu*Np)
            P = Ps[i]
            TAYLOR_COEFS = TAYLOR_COEFS_samples[i]

            u_prev = init_u
            
            self.opti.subject_to(xs[:,0] == x0)

            OBS = ca.vertcat(ys[0, -1], timestep+self.Np)
            OBS_NORM = self.opti.variable(2)
            self.opti.subject_to(
                OBS_NORM == self.normalizeState_casadi(
                    OBS,
                    np.array([self.x_min[0], 0]),
                    np.array([self.x_max[0], self.N])
                )
            )

            for ll in range(0, self.Np):
                pk = ps[:, ll]

                uk = self.compute_control_input(xs, ll, us, os_y, os_u, theta, jac_y_full, jac_input_full, u_prev)

                # System dynamics and input constraints
                self.opti.subject_to(xs[:, ll+1] == self.F(xs[:, ll], uk, ds[:, ll], pk))
                self.opti.subject_to(ys[:, ll] == self.g(xs[:, ll+1]))
                self.opti.subject_to(self.u_min <= (uk <= self.u_max))

                # Linear penalty functions
                self.opti.subject_to(P[:, ll] >= 0)
                self.opti.subject_to(P[0, ll] >= self.lb_pen_w[0,0] * (self.y_min[1] - ys[1, ll]))
                self.opti.subject_to(P[1, ll] >= self.ub_pen_w[0,0] * (ys[1, ll] - self.y_max[1]))
                self.opti.subject_to(P[2, ll] >= self.lb_pen_w[0,1] * (self.y_min[2] - ys[2, ll]))
                self.opti.subject_to(P[3, ll] >= self.ub_pen_w[0,1] * (ys[2, ll] - self.y_max[2]))
                self.opti.subject_to(P[4, ll] >= self.lb_pen_w[0,2] * (self.y_min[3] - ys[3, ll]))
                self.opti.subject_to(P[5, ll] >= self.ub_pen_w[0,2] * (ys[3, ll] - (self.y_max[3]-2.0)))

                # COST FUNCTION WITH PENALTIES
                delta_dw = xs[0, ll+1] - xs[0, ll]
                J -= compute_economic_reward(delta_dw, p, self.dt, uk)
                J += (P[0, ll]+ P[1, ll]+P[2, ll]+P[3, ll]+P[4, ll]+P[5, ll])
                
                # Input rate constraint on the variation from the previous control action)
                if ll < self.Np-1:
                    next_input = self.compute_control_input(xs, ll+1, us, os_y, os_u, theta, jac_y_full, jac_input_full, uk)
                    self.opti.subject_to(-self.du_max <= (next_input - uk <= self.du_max))
                u_prev = uk

            # Value Function insertion
            if self.use_trained_vf:
                J_terminal = self.vf_casadi_approx_func(OBS_NORM, TAYLOR_COEFS)
            else:
                J_terminal = 0
            #     print("using QF")
            #     self.opti.subject_to(OBS[0] - ys[0,-1] == 0)
            #     self.opti.subject_to(OBS[1:4] - ys[1:,-1] == 0)
            #     self.opti.subject_to(OBS[4:7] - us[:,-1] == 0)
            #     self.opti.subject_to(OBS[7] - timestep + self.Np == 0)
            #     self.opti.subject_to(OBS[8:] - ds[0:,-1] == 0)
            #     self.opti.subject_to(OBS_NORM == self.norm_obs_agent(OBS, self.mean, self.variance))
            #     J_terminal = self.qf_function(ca.vertcat(OBS_NORM, A))  
            J -= J_terminal


            # # Constraints on intial input
            uk = self.compute_control_input(xs, 0, us, os_y, os_u, theta, jac_y_full, jac_input_full, init_u)
            self.opti.subject_to(-self.du_max <= (uk - init_u <= self.du_max))  

        J = J / self.Ns

        self.opti.minimize(J)
        self.opti.solver('ipopt', self.nlp_opts)

        self.MPC_func = self.opti.to_function(
            "MPC_func",
            [
                x0, ds, init_u, timestep, theta, *xs_list, *terminal_xs, *u_samples, *obs_norm_y_samples,
                *obs_norm_input_samples, *jac_obs_y_samples, *jac_obs_input_samples, *p_samples, *TAYLOR_COEFS_samples
            ],                                                      # Function input

            [theta, ca.vertcat(*xs_list), ca.vertcat(*ys_list), J], # output

            ["x0", "ds", "init_u", "timestep", "theta_init"] +      # Function input
            [f"x_init_{i}" for i in range(self.Ns)] +               # xs initial guess
            [f"x_terminal_{i}" for i in range(self.Ns)] +           # terminal constraints for x
            [f"u_sample_{i}" for i in range(self.Ns)] +             # u samples
            [f"obs_norm_y_samples_{i}" for i in range(self.Ns)] +   # normalized y samples
            [f"obs_norm_input_samples_{i}" for i in range(self.Ns)] + # normalized input samples
            [f"jac_obs_y_sample_{i}" for i in range(self.Ns)] +     # policy jacobian wrt y
            [f"jac_obs_input_sample_{i}" for i in range(self.Ns)] + # policy jacobian wrt input 
            [f"p_sample_{i}" for i in range(self.Ns)] +             # p samples
            [f"taylor_coefs_{i}" for i in range(self.Ns)],          # taylor coefficients

            ["theta_opt", "xs_opt", "ys_opt", "J"]                  # output
        ) 


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
        mpc: SMPC,
        save_name: str,
        project_name: str,
        weather_filename: str,
        uncertainty_value: float,
        p,
        rng
    ) -> None:

        self.project_name = project_name
        self.save_name = save_name
        self.mpc = mpc
        self.uncertainty_value = uncertainty_value
        self.p = p
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
        self.p_samples_all = np.zeros((mpc.Ns, 34, mpc.Np, mpc.N))
        self.dJdu = np.zeros((mpc.nu*mpc.Np, mpc.N))
        self.xs_opt_all = np.zeros((mpc.Ns, mpc.nx, mpc.Np+1, mpc.N))
        self.ys_opt_all = np.zeros((mpc.Ns, mpc.ny, mpc.Np+1, mpc.N))
        self.output = []
        self.rewards = np.zeros((1, mpc.N))
        self.penalties = np.zeros((1, mpc.N))
        self.econ_rewards = np.zeros((1, mpc.N))
        self.solver_times = np.zeros((1, mpc.N))
        self.exit_message = np.zeros((1, mpc.N))

    def generate_psamples(self) -> List[np.ndarray]:
        """
        Generate parametric samples for the specified number of scenarios.

        This function creates a list of parametric samples, where each element 
        corresponds to a scenario and contains samples generated based on the 
        specified uncertainty parameters.

        The number of scenarios is determined by `self.mpc.Ns`, and for each scenario, 
        `self.mpc.Np` samples are generated using the `parametric_uncertainty` function.

        Returns:
            List[np.ndarray]: A list where each element is a 2D array of generated 
            parametric samples for a scenario.

        Example:
            p_sample_list = self.generate_psamples()
        """
        p_samples = []
        for i in range(self.mpc.Ns):
            scenario_samples = []
            for k in range(self.mpc.Np):
                # Use MPC's random number generator to sample parameters
                pk = parametric_uncertainty(self.p, self.uncertainty_value, self.mpc.rng)
                scenario_samples.append(pk)
            p_samples.append(np.vstack(scenario_samples))
        return p_samples

    def generate_samples(self, p_samples: List[np.ndarray]) -> Tuple[List[np.ndarray], ...]:
        """
        Generate state, input, and observation samples for each scenario.

        This function generates various samples and their derivatives for each scenario 
        by unrolling the RL policy (i.e., actor) on the environment with the provided parametric samples.

        Args:
            p_samples (List[np.ndarray]): List of parametric samples for each scenario.
                Each element is a 2D array of shape (Np, n_params).

        Returns:
            Tuple[List[np.ndarray], ...]: A tuple containing lists of samples for each scenario:
                - xk_samples: List of state trajectories
                - terminal_xs: List of terminal states
                - uk_samples: List of input trajectories
                - end_points: List of the terminal observations
                - obs_norm_y_samples: List of normalized output observations
                - obs_norm_input_samples: List of normalized input observations
                - jacobian_obs_state_samples: List of state Jacobians
                - jacobian_obs_input_samples: List of input Jacobians
        """
        xk_samples = []
        terminal_xs = []
        uk_samples = []
        end_points = []
        obs_norm_y_samples = []
        obs_norm_input_samples = []
        jacobian_obs_state_samples = []
        jacobian_obs_input_samples = []

        for i in range(self.mpc.Ns):
            p_i_samples = p_samples[i]
            log = self.mpc.unroll_actor(p_i_samples, horizon=self.mpc.Np)
            
            # Store input and state trajectories
            uk_samples.append(np.vstack(log["u"]))
            xk_samples.append(np.vstack(log["x"]))

            terminal_xs.append(log["x"][:, -1])

            # Extract normalized observations
            obs_norm = np.vstack(log["obs_norm"])
            obs_norm_y_samples.append(obs_norm[:self.mpc.ny, :-1])
            obs_norm_input_samples.append(obs_norm[self.mpc.ny:self.mpc.ny+self.mpc.nu, :-1])

            # extract the observation of the terminal observation:
            end_points.append(log["obs"][:, -1])

            # Extract components for Jacobian computation
            y = log["obs_norm"][:4]
            u = log["obs_norm"][4:7]
            timestep = log["obs_norm"][7]
            d = log["obs_norm"][8:]

            # Compute Jacobians for each timestep
            jacobian_obs_state = []
            jacobian_obs_input = []
            for t in range(self.mpc.Np):
                y_t = y[:, t]
                u_t = u[:, t]
                timestep_t = timestep[t]
                d_t = d[:, t]
                
                jac_t_state = self.mpc.jac_actor_wrt_state(y_t, u_t, timestep_t, d_t).toarray()
                jacobian_obs_state.append(jac_t_state)

                jac_t_input = self.mpc.jac_actor_wrt_input(y_t, u_t, timestep_t, d_t).toarray()
                jacobian_obs_input.append(jac_t_input)

            jacobian_obs_state_samples.append(np.hstack(jacobian_obs_state))
            jacobian_obs_input_samples.append(np.hstack(jacobian_obs_input))

        return (xk_samples, terminal_xs, uk_samples, end_points, obs_norm_y_samples, obs_norm_input_samples, 
                jacobian_obs_state_samples, jacobian_obs_input_samples)

    def get_taylor_coefficients(self, end_points: List[np.ndarray]) -> List[ca.DM]:
        """
        Extracts normalized terminal points from observations and computes Taylor coefficients.

        Args:
            end_points: List of Ns observations of shape (12,1)
        
        Returns:
            List of Taylor coefficients for each sample
        """
        taylor_coefficients = []

        for obs in end_points:
            # Extract relevant state information
            term_point = np.array([obs[0], obs[7]])

            # Normalize the state
            norm_term_point = self.mpc.normalizeState_casadi(
                term_point,
                np.array([self.mpc.y_min[0], 0]),
                np.array([self.mpc.y_max[0], self.mpc.N])
            )

            # Get Taylor coefficients for the value function approximation
            coefs = self.mpc.vf_casadi_model_approx.get_params(norm_term_point.toarray().ravel())
            taylor_coefficients.append(coefs)

        return taylor_coefficients

    def solve_nsmpc(self, order: str) -> None:
        """
        Solve the nonlinear Reinforcement Learning Stochastic Model Predictive Control (RL-SMPC) problem.

        This function executes the RL-SMPC algorithm over a specified time horizon, using either 
        zero-order or first-order approximation schemes to compute the optimal control inputs. 
        It iteratively updates the system state, computes economic rewards, and applies penalties 
        based on the control actions taken.

        Args:
            order (str): The order of the RL approximation scheme, either "zero" or "first".

        Returns:
            None: The function updates the class attributes with the results of the optimization:
                - self.u: Updated control inputs
                - self.x: Updated state trajectory
                - self.y: Updated output trajectory
                - self.J: Cost values associated with the control inputs
                - self.econ_rewards: Economic rewards computed for each step
                - self.penalties: Penalties applied for constraint violations

        Note:
            The function performs the following steps:
            - Resets the environment and initializes variables
            - Iteratively solves the MPC problem for each time step
            - Applies the computed control inputs to the system
            - Updates the state and output trajectories
            - Computes and logs economic rewards and penalties
        """
        self.mpc.eval_env.reset()
        theta_init = np.zeros((self.mpc.nu, self.mpc.Np))

        for ll in tqdm(range(self.mpc.N)):
            p_samples = self.generate_psamples()
            if ll == 0:
                self.mpc.eval_env.set_env_state(self.x[:, ll], self.x[:,ll], self.u[:,ll], ll)
            else:
                self.mpc.eval_env.set_env_state(self.x[:, ll], self.x[:,ll-1], self.u[:,ll], ll)
            (xk_samples, terminal_xs, uk_samples, end_points, obs_norm_y_samples, obs_norm_input_samples, 
                jacobian_obs_state_samples, jacobian_obs_input_samples) = \
                self.generate_samples(p_samples)

            # Get Taylor coefficients for all samples
            taylor_coefficients = self.get_taylor_coefficients(end_points)

            ds = self.d[:, ll:ll+self.mpc.Np+1]
            timestep = [ll]

            # we have to transpose p_samples since MPC_func expects matrix of shape (n_params, Np)
            p_sample_list = [p_samples[i].T for i in range(self.mpc.Ns)]

            # Call MPC function with all inputs as CasADi DM
            if order == "zero":
                xs_opt, ys_opt, theta_opt, J_mpc_1, solver_time, exit_message = \
                    self.mpc.solve_ocp(
                        self.x[:, ll], 
                        self.u[:,ll],
                        ds,
                        timestep,
                        theta_init,
                        xk_samples,
                        terminal_xs,
                        uk_samples,
                        p_samples,
                        taylor_coefficients
                    )

            elif order == "first":
                theta_opt, xs_opt, ys_opt, J_mpc_1 = self.mpc.MPC_func(
                    self.x[:, ll],          # initial state
                    ds,                     # disturbances
                    self.u[:, ll],          # initial input
                    timestep,               # current timestep
                    theta_init,             # initial guess for theta
                    *xk_samples,            # initial guess for the states
                    *terminal_xs,           # terminal state constraint
                    *uk_samples,            # input samples
                    *obs_norm_y_samples,    # observation y samples
                    *obs_norm_input_samples,        # observation input samples
                    *jacobian_obs_state_samples,    # jacobian evaluated at observation y samples
                    *jacobian_obs_input_samples,    # jacobian evaluated at observation input samples
                    *p_sample_list,                  # parameter samples
                    *taylor_coefficients            # taylor coefficients for value function approximation
                )

            # Since the first RL sample always depends on x0 all the samples input (u) at t=0 will the same;
            # Additionally, the first order approximation does not require to be used in the first iteration
            us_opt = uk_samples[0][:,0] + theta_opt[:, 0]
            self.u[:, ll+1] = us_opt

            theta_init = np.concatenate([theta_opt[:, 1:], ca.reshape(theta_opt[:, -1], (self.mpc.nu, 1))], axis=1)

            # Evolve State
            params = parametric_uncertainty(self.p, self.uncertainty_value, self.rng)

            self.x[:, ll+1] = self.mpc.F(self.x[:, ll], self.u[:, ll+1], self.d[:, ll], params).toarray().ravel()
            self.y[:, ll+1] = self.mpc.g(self.x[:, ll+1]).toarray().ravel()

            delta_dw = self.x[0, ll+1] - self.x[0, ll]
            econ_rew = compute_economic_reward(delta_dw, get_parameters(), self.mpc.dt, self.u[:, ll+1])
            penalties = self.mpc.compute_penalties(self.y[:, ll+1])
            self.update_results(us_opt.reshape(-1,1), J_mpc_1, [], econ_rew, penalties, ll, solver_time, exit_message=exit_message)

    def solve_smpc_OL_predictions(self, order) -> None:
        """Solve the nonlinear Stochastic MPC problem.

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
        self.mpc.eval_env.reset()
        theta_init = np.zeros((self.mpc.nu, self.mpc.Np))

        for ll in tqdm(range(1)):
            p_samples = self.generate_psamples()
            if ll == 0:
                self.mpc.eval_env.set_env_state(self.x[:, ll], self.x[:,ll], self.u[:,ll], ll)
            else:
                self.mpc.eval_env.set_env_state(self.x[:, ll], self.x[:,ll-1], self.u[:,ll], ll)
            (xk_samples, terminal_xs, uk_samples, end_points, obs_norm_y_samples, obs_norm_input_samples, 
                jacobian_obs_state_samples, jacobian_obs_input_samples) = \
                self.generate_samples(p_samples)

            # Get Taylor coefficients for all samples
            taylor_coefficients = self.get_taylor_coefficients(end_points)

            # Convert inputs to CasADi DM format; NOTE is this required??
            ds = self.d[:, ll:ll+self.mpc.Np+1]
            timestep = [ll]

            # we have to transpose p_samples since MPC_func expects matrix of shape (n_params, Np)
            p_sample_list = [p_samples[i].T for i in range(self.mpc.Ns)]

            # Call MPC function with all inputs as CasADi DM
            if order == "zero":
                xs_opt, ys_opt, theta_opt, J_mpc_1, exit_message = \
                    self.mpc.solve_ocp(
                        self.x[:, ll], 
                        self.u[:,ll],
                        ds,
                        timestep,
                        theta_init,
                        xk_samples,
                        terminal_xs,
                        uk_samples,
                        p_samples,
                        taylor_coefficients
                    )

            elif order == "first":
                theta_opt, xs_opt, ys_opt, J_mpc_1 = self.mpc.MPC_func(
                    self.x[:, ll],          # initial state
                    ds,                     # disturbances
                    self.u[:, ll],          # initial input
                    timestep,               # current timestep
                    theta_init,             # initial guess for theta
                    *xk_samples,            # initial guess for the states
                    *terminal_xs,           # terminal state constraint
                    *uk_samples,            # input samples
                    *obs_norm_y_samples,    # observation y samples
                    *obs_norm_input_samples,        # observation input samples
                    *jacobian_obs_state_samples,    # jacobian evaluated at observation y samples
                    *jacobian_obs_input_samples,    # jacobian evaluated at observation input samples
                    *p_sample_list,                  # parameter samples
                    *taylor_coefficients            # taylor coefficients for value function approximation
                )

            # Since the first RL sample always depends on x0 all the samples input (u) at t=0 will the same;
            if order == "zero":
                us_opt = uk_samples[0][:,0] + theta_opt[:, 0]

            elif order == "first":
                jac_y_matrix = jacobian_obs_state_samples[0][:,:self.mpc.ny]
                jac_input_matrix = jacobian_obs_input_samples[0][:,:self.mpc.nu]
                obs_y = self.mpc.g(self.x[:, ll])
                obs_norm_y = self.mpc.norm_obs_agent(obs_y, self.mpc.mean[:self.mpc.ny], self.mpc.variance[:self.mpc.ny])
                obs_u = self.u[:, ll]
                obs_norm_u = self.mpc.norm_obs_agent(obs_u, self.mpc.mean[self.mpc.ny:self.mpc.ny+self.mpc.nu], self.mpc.variance[self.mpc.ny:self.mpc.ny+self.mpc.nu])

                # Compute control input
                us_opt = uk_samples[0][:, 0] + ca.mtimes(jac_y_matrix, (obs_norm_y_samples[0][:, 0] - obs_norm_y)) + \
                    ca.mtimes(jac_input_matrix, obs_norm_input_samples[0][:, 0] - obs_norm_u) + theta_opt[:, 0]

            self.u[:, ll+1] = us_opt.toarray().ravel()

            theta_init = np.concatenate([theta_opt[:, 1:], ca.reshape(theta_opt[:, -1], (self.mpc.nu, 1))], axis=1)

            # Evolve State
            params = parametric_uncertainty(self.p, self.uncertainty_value, self.rng)

            self.x[:, ll+1] = self.mpc.F(self.x[:, ll], self.u[:, ll+1], self.d[:, ll], params).toarray().ravel()
            self.y[:, ll+1] = self.mpc.g(self.x[:, ll+1]).toarray().ravel()

            delta_dw = self.x[0, ll+1] - self.x[0, ll]
            econ_rew = compute_economic_reward(delta_dw, get_parameters(), self.mpc.dt, self.u[:, ll+1])
            penalties = self.mpc.compute_penalties(self.y[:, ll+1])

            xs_opt = xs_opt.toarray().reshape(self.mpc.Ns, self.mpc.nx, self.mpc.Np+1)
            ys_opt = ys_opt.toarray().reshape(self.mpc.Ns, self.mpc.ny, self.mpc.Np)

            # Insert the first index of the third dimension of xs_opt into ys_opt
            # We need to reshape ys_opt to include space for the additional timestep
            ys_opt_with_x0 = np.zeros((self.mpc.Ns, self.mpc.ny, self.mpc.Np + 1))
            
            # Copy existing ys_opt data
            ys_opt_with_x0[:, :, 1:] = ys_opt
            
            # For the first timestep, calculate outputs from the first state in xs_opt
            for s in range(self.mpc.Ns):
                ys_opt_with_x0[s, :, 0] = self.mpc.g(xs_opt[s, :, 0]).toarray().ravel()

            # Replace ys_opt with the new array that includes the initial output
            ys_opt = ys_opt_with_x0

            # Save the open-loop predictions for this timestep
            self.xs_opt_all[:, :, :, ll] = xs_opt
            self.ys_opt_all[:, :, :, ll] = ys_opt
            self.p_samples_all[:,:,:,ll] = np.array(p_sample_list)

            # self.update_results(us_opt, J_opt, [], econ_rew, penalties, ll)
            self.update_results(uk_samples[0][:,:] + theta_opt[:, :], J_mpc_1, [], econ_rew, penalties, ll)
        
        # Save the open-loop predictions to file after all timesteps
        self.save_open_loop_predictions()

    def save_open_loop_predictions(self):
        """
        Save the open-loop predictions (xs_opt_all and ys_opt_all) to a file.
        Using .npz format which is efficient for storing multiple numpy arrays.
        """
        save_path = os.path.join("data", self.project_name)
        os.makedirs(save_path, exist_ok=True)
        save_path = os.path.join(save_path,f"{self.save_name}-OL-predictions.npz")
        np.savez(
            save_path,
            xs_opt_all=self.xs_opt_all,
            ys_opt_all=self.ys_opt_all,
            us_opt=self.uopt,
            p_samples_all=self.p_samples_all,
            d=self.d,
            x=self.x,
            y=self.y,
            u=self.u,
            J=self.J,
            econ_rewards=self.econ_rewards,
            penalties=self.penalties,
            rewards=self.rewards
        )
        print(f"Open-loop predictions saved to {save_path}")


    def update_results(self, us_opt, Js_opt, sol, eco_rew, penalties, ll, solver_time, exit_message=None):
        """
        Args:
            uopt (_type_): _description_
            J (_type_): _description_
            output (_type_): _description_
            ll (_type_): _description_
        """
        if exit_message == "Solve_Succeeded" or exit_message == "Solved_To_Acceptable_Level":
            exit_message = 0
        else:
            exit_message = 1

        self.uopt[:,:, ll] = us_opt
        self.J[:, ll] = Js_opt
        self.output.append(sol)
        self.econ_rewards[:, ll] = eco_rew
        self.penalties[:, ll] = penalties
        self.rewards[:, ll] = eco_rew - penalties
        self.solver_times[:, ll] = solver_time
        self.exit_message[:, ll] = exit_message

    def get_results(self, run) -> np.ndarray:
        """
        Returns an array with the closed-loop trajectories of the simulation run.
        Adds the run ID to the last column of the array.
        Additionally saves performance metrics such as cost, rewards, and penalties.
        Args:
            run (int): The run ID for the simulation.
        Returns:
            np.ndarray: A 2D array containing the closed-loop trajectories and performance metrics.
        """
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
        arrays.append(self.solver_times.flatten())
        arrays.append(self.exit_message.flatten())
        arrays.append(np.ones(self.mpc.N) * run)

        # Stack all arrays vertically
        return np.vstack(arrays).T

    def retrieve_results(self, run=0) -> pd.DataFrame:
        """
        Creates a pandas dataframe with the closed-loop trajectories of the simulation run.
        Adds the run ID to the last column of the dataframe.
        Additionally saves performance metrics such as cost, rewards, and penalties.
        Args:
            run (int): The run ID for the simulation.
        Returns:
            pd.DataFrame: A dataframe containing the closed-loop trajectories and performance metrics.
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
        # for i in range(self.uopt.shape[0]):
        #     for j in range(self.uopt.shape[1]):
        #         data[f"uopt_{i}_{j}"] = self.uopt[i, j, :-1]
        data["J"] = self.J.flatten()
        data["econ_rewards"] = self.econ_rewards.flatten()
        data["penalties"] = self.penalties.flatten()
        data["rewards"] = self.rewards.flatten()
        data["solver_times"] = self.solver_times.flatten()
        data["solver_success"] = self.exit_message.flatten()

        df = pd.DataFrame(data, columns=data.keys())
        df['run'] = run
        return df

    def save_results(self, save_path):
        """
        Saves the results of a single run experiment into a CSV file.
        Args:
            save_path (str): The path where the results will be saved.
        """
        df = self.retrieve_results()
        df.to_csv(f"{save_path}/{self.save_name}", index=False)

def create_rl_smpc(
        h, 
        env_params, 
        mpc_params,
        rl_env_params, 
        algorithm,
        env_path, 
        rl_model_path, 
        vf_path,
        run,
        use_trained_vf=True,
        Ns=10,
    ):
    """Creates a Reinforcement Learning Stochastic Model Predictive Controller (RL-SMPC).

    Args:
        h (int): Prediction horizon in hours
        env_params (dict): Environment parameters for the greenhouse simulation
        mpc_params (dict): Parameters for the MPC controller
        rl_env_params (dict): Parameters for the RL environment
        args (Namespace): Command line arguments containing algorithm and value function settings
        env_path (str): Path to the normalized environment file
        rl_model_path (str): Path to the trained RL model
        vf_path (str): Path to the trained value function

    Returns:
        RLSMPC: An initialized RL-SMPC controller instance
    """
    mpc_rng = np.random.default_rng(42 + run)
    mpc_params["rng"] = mpc_rng    
    mpc_params["Ns"] = Ns
    mpc_params["Np"] = int(h * 3600 / env_params["dt"])

    return RLSMPC(
        env_params,
        mpc_params,
        rl_env_params,
        algorithm, 
        env_path,
        rl_model_path,
        use_trained_vf=use_trained_vf,
        vf_path=vf_path,
        run=run,
    )

def load_experiment_parameters(project, env_id, algorithm, mode, model_name, uncertainty_value):
    """Load and prepare all parameters needed for the RL-SMPC experiment.

    Args:
        project (str): Project name
        env_id (str): Environment identifier
        algorithm (str): RL algorithm name
        mode (str): Mode of operation ('deterministic' or 'stochastic')
        model_name (str): Name of the trained model
        uncertainty_value (float): Value for uncertainty parameter

    Returns:
        tuple: Contains environment parameters, MPC parameters, RL environment parameters,
                and paths to the trained models/environments
    """
    load_path = f"train_data/{project}/{algorithm}/{mode}"

    # load the environment parameters
    env_params = load_env_params(env_id)
    mpc_params = load_mpc_params(env_id)
    mpc_params["uncertainty_value"] = uncertainty_value

    # load the RL parameters
    hyperparameters, rl_env_params = load_rl_params(env_id, algorithm)
    rl_env_params.update(env_params)
    rl_env_params["uncertainty_value"] = uncertainty_value

    # the paths to the RL models and environment
    rl_model_path = f"{load_path}/models/{model_name}/best_model.zip"
    vf_path = f"{load_path}/models/{model_name}/vf.zip"
    env_path = f"{load_path}/envs/{model_name}/best_vecnormalize.pkl"

    return env_params, mpc_params, rl_env_params, env_path, rl_model_path, vf_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str, default="matching-thesis")
    parser.add_argument("--env_id", type=str, default="LettuceGreenhouse")
    parser.add_argument("--save_name", type=str)
    parser.add_argument("--weather_filename", default="outdoorWeatherWurGlas2014.csv", type=str)
    parser.add_argument("--algorithm", type=str, default="sac")
    parser.add_argument("--model_name", type=str, default="thesis-agent")
    parser.add_argument("--use_trained_vf", action="store_true")
    parser.add_argument("--uncertainty_value", type=float, required=True)
    parser.add_argument("--mode", type=str, choices=['deterministic', 'stochastic'], required=True)
    parser.add_argument("--order", type=str, choices=["zero", "first"], required=True)
    args = parser.parse_args()

    save_path = f"data/{args.project}/stochastic/rlsmpc"
    os.makedirs(save_path, exist_ok=True)

    env_params, mpc_params, rl_env_params, env_path, rl_model_path, vf_path = \
        load_experiment_parameters(args.project, args.env_id, args.algorithm, args.mode, args.model_name, args.uncertainty_value)

    # run the experiment
    H = [1, 2, 3, 4, 5, 6]
    for h in H:
        save_name = f"{args.model_name}-{args.save_name}-{h}H-{args.uncertainty_value}"
        p = get_parameters()
        exp_rng = np.random.default_rng(666) 

        rl_mpc = create_rl_smpc(
            h, 
            env_params, 
            mpc_params,
            rl_env_params,
            args.algorithm,
            env_path,
            rl_model_path,
            vf_path,
            args.used_trained_vf
        )

        if args.order == "zero":
            rl_mpc.define_zero_order_snlp(p)
        elif args.order == "first":
            rl_mpc.define_first_order_snlp(p)

        exp = Experiment(rl_mpc, save_name, args.project, args.weather_filename, args.uncertainty_value, p, exp_rng)
        exp.solve_nsmpc(args.order)
        exp.save_results(save_path)
