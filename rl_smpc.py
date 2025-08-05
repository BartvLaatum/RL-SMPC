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
    """
    Reinforcement Learning Stochastic Model Predictive Controller (RL-SMPC) for greenhouse systems.

    This class implements a hybrid control approach that combines Reinforcement Learning (RL) 
    with Stochastic Model Predictive Control (SMPC). The controller uses a trained RL policy 
    to generate nominal control trajectories and then optimizes corrections to these trajectories 
    using scenario-based stochastic optimization.

    The RL-SMPC approach provides several advantages:
    - Leverages the generalization capabilities of trained RL policies
    - Handles parametric uncertainty through scenario-based optimization
    - Maintains constraint satisfaction through explicit optimization
    - Balances exploration (RL policy) with exploitation (MPC optimization)

    The controller supports both zero-order and first-order approximation schemes:
    - Zero-order: Direct perturbation of nominal RL control inputs
    - First-order: Uses gradient information from the RL policy for refined corrections

    Attributes:
        eval_env (LettuceGreenhouse): Unnormalized environment for evaluation
        model (PPO/SAC): Trained RL model (PPO or SAC algorithm)
        trained_vf (torch.nn.Module): Trained value function for terminal cost
        use_trained_vf (bool): Whether to use the trained value function
        terminal (bool): Whether to enforce terminal state constraints
        rl_feedback (bool): Whether to use RL feedback in control computation
        variance (np.ndarray): Observation variance for normalization
        mean (np.ndarray): Observation mean for normalization
        norm_obs_agent (casadi.Function): CasADi function for observation normalization
        normalizeState_casadi (casadi.Function): CasADi function for state normalization
        vf_casadi_model_approx (RealTimeL4CasADi): CasADi approximation of value function
        vf_casadi_approx_func (casadi.Function): CasADi function for value function evaluation
        actor_function (casadi.Function): CasADi function for RL policy evaluation
        coef_size (int): Size of Taylor expansion coefficients

    Methods:
        init_l4casadi(run) -> None:
            Initialize CasADi neural network functions for RL policy and value function.
            
        h(x, u, d, timestep) -> casadi.MX:
            Observation function that constructs the full observation vector.
            
        unroll_actor(p_i_samples, horizon, freeze) -> dict:
            Unroll the RL policy over the environment for scenario generation.
            
        solve_ocp(x0, u0, ds, timestep, theta_init, xk_samples, terminal_xs, 
                 uk_samples, p_samples, taylor_coefficients) -> tuple:
            Solve the optimal control problem with scenario-based constraints.
            
        define_zero_order_snlp(p) -> None:
            Define the zero-order stochastic nonlinear programming problem.
            
        compute_control_input(xs, ll, us, os_y, os_u, theta, jac_y_full, 
                            jac_input_full, u_prev) -> casadi.MX:
            Compute control input using first-order Taylor expansion (first-order mode).

    Example:
        >>> env_params = load_env_params("LettuceGreenhouse")
        >>> mpc_params = load_mpc_params("LettuceGreenhouse")
        >>> rl_env_params = load_rl_params("LettuceGreenhouse", "sac")
        >>> rl_mpc = RLSMPC(env_params, mpc_params, rl_env_params, "sac",
        ...                 "env_path.pkl", "model.zip", "vf.zip", run=0)
        >>> rl_mpc.define_zero_order_snlp(model_parameters)

    Notes:
        - Inherits from SMPC class for base stochastic MPC functionality
        - Uses L4CasADi for efficient neural network integration with CasADi
        - Supports both PPO and SAC algorithms
        - Implements scenario-based uncertainty handling
        - Provides flexible terminal cost and constraint options
    """
    def __init__(
            self,
            env_params: Dict[str, Any],
            mpc_params: Dict[str, Any],
            rl_env_params: Dict[str, Any],
            algorithm: str,
            env_path: str,
            rl_model_path: str,
            vf_path,
            run: int,
            use_trained_vf: bool,
            terminal: bool = True,
            rl_feedback: bool = True
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
        self.terminal = terminal
        self.rl_feedback = rl_feedback
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
        state_norm = 10 * ((observation - min_val) / (max_val - min_val)) - 5 # TODO: Should it use parameters from SHARED.params instead of 10 and 5?
        self.normalizeState_casadi = ca.Function("normalizeObs", [observation, min_val, max_val], [state_norm])

        # Creating casadi verions of the value functions
        obs_sym_vf = ca.MX.sym("obs", 2, 1)

        # Create Taylor approximation model of the value function
        self.vf_casadi_model_approx = l4c.realtime.RealTimeL4CasADi(self.trained_vf, approximation_order=1)
        vf_casadi_approx_sym_out = self.vf_casadi_model_approx(obs_sym_vf)
        self.vf_casadi_approx_func =  ca.Function(
            "vf_approx",
            [obs_sym_vf, self.vf_casadi_model_approx.get_sym_params()],
            [vf_casadi_approx_sym_out]
        )

        # Creating casadi version of the actor
        actor_casadi_model = l4c.L4CasADi(actor_fn(self.model.actor.latent_pi, self.model.actor.mu), device="cpu", name=f"actor_{run}")
        obs_shape = self.eval_env.observation_space.shape
        obs_sym = ca.MX.sym("obs_sym", obs_shape[0], 1)
        action_out = actor_casadi_model(obs_sym.T)
        self.actor_function = ca.Function(
            "action",
            [obs_sym],
            [action_out]
        )

        # Coefficients for Taylor approximation of the terminal value function
        casadi_vf_approx_param = self.vf_casadi_model_approx.get_params(np.zeros(2))
        self.coef_size = casadi_vf_approx_param.shape[0] 

    def unroll_actor(self, p_i_samples=None, horizon=1, freeze=True,):
        """
        Unroll the RL policy over the environment for scenario generation.

        This method simulates the RL policy's behavior over a specified horizon by
        repeatedly applying the policy to the environment and collecting trajectories.
        It generates nominal control sequences that serve as starting points nonlinear feedback policy for
        the scenario-baed SMPC optimization

        Args:
            p_i_samples (np.ndarray, optional): Parametric uncertainty samples for 
                the scenario. Shape should be (horizon, n_params). Defaults to None.
            horizon (int, optional): Number of steps to unroll the policy. Defaults to 1.
            freeze (bool, optional): Whether to freeze the environment state during 
                unrolling. Defaults to True.

        Returns:
            dict: Dictionary containing the unrolled trajectory data:
                - "obs" (np.ndarray): Observation trajectories (n_obs, horizon+1)
                - "x" (np.ndarray): State trajectories (nx, horizon+1)
                - "u" (np.ndarray): Control input trajectories (nu, horizon)
                - "obs_norm" (np.ndarray): Normalized observation trajectories (n_obs, horizon+1)
                - "total_reward" (float): Total accumulated reward over horizon
                - "reward_log" (np.ndarray): Individual step rewards (horizon,)
        """
        # Define empty log variables
        log = {
            "obs":[],
            "x":[],
            "u":[],
        }

        # Lists to log the rolling actor
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

        # Perform rollout for the given horizon length
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
        Solve the optimal control problem.

        This method sets up and solves the scenario-based stochastic MPC optimization
        problem. It initializes all scenario parameters, sets initial guesses for
        optimization variables, and solves the resulting nonlinear programming problem.

        Args:
            x0 (np.ndarray): Initial state vector (nx,)
            u0 (np.ndarray): Initial control input vector (nu,)
            ds (np.ndarray): Disturbance trajectory (nd, Np+1)
            timestep (list): Current simulation timestep [k]
            theta_init (np.ndarray): Initial guess for control corrections (nu, Np)
            xk_samples (list): List of state trajectory guesses for each scenario
            terminal_xs (list): List of terminal state constraints for each scenario
            uk_samples (list): List of nominal control inputs for each scenario
            p_samples (list): List of parametric uncertainty samples for each scenario
            taylor_coefficients (list): List of Taylor expansion coefficients for value function

        Returns:
            tuple: Contains optimization results:
                - xs_opt (list): Optimal state trajectories for all scenarios
                - ys_opt (list): Optimal output trajectories for all scenarios
                - theta_opt (np.ndarray): Optimal control corrections (nu, Np)
                - J (float): Optimal cost value
                - solver_time (float): Time taken by the optimization solver
                - exit_message (str): Solver exit status message
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
        Define stochastic nonlinear programming with zero-order approximation of 
        the RL policy as feedback law. 
        Inlcuding terminal cost and constraints provided by the RL policy.

        This function sets up the optimization problem for the scenario-based stochastic MPC 
        using CasADi's Opti stack. It defines decision variables, constraints, and the 
        cost function for multiple scenarios.

        The zero-order approach uses direct perturbation of the nominal control input 
        generated by the RL policy.

        Args:
            p (np.ndarray): Array of model parameters used in the system dynamics 
                and cost function calculations.

        Returns:
            None: The function updates the class attributes:
                - self.opti: CasADi optimization problem
                - self.MPC_func: CasADi function for solving the MPC problem
        """
        self.opti = ca.Opti()
        num_penalties = 6

        # Decision Variables (Control inputs, slack variables, states, outputs)
        self.theta = self.opti.variable(self.nu, self.Np)  # Control inputs (nu x Np)
        self.xs_list = [self.opti.variable(self.nx, self.Np+1) for _ in range(self.Ns)]
        self.ys_list = [self.opti.variable(self.ny, self.Np) for _ in range(self.Ns)]
        Ps = [self.opti.variable(num_penalties, self.Np) for _ in range(self.Ns)]

        # Taylor expansion coefficients for the value function
        self.TAYLOR_COEFS_samples = [self.opti.parameter(self.coef_size) for _ in range(self.Ns)]

        # Parameters
        self.timestep = self.opti.parameter(1,1)
        self.u_samples = [self.opti.parameter(self.nu, self.Np) for _ in range(self.Ns)]
        self.p_samples = [self.opti.parameter(p.shape[0], self.Np) for _ in range(self.Ns)]

        # Initial parameter values
        self.x0 = self.opti.parameter(self.nx, 1)  # Initial state
        self.init_u = self.opti.parameter(self.nu, 1)  # Initial control input
        self.ds = self.opti.parameter(self.nd, self.Np+1) # Disturbance Variables
        self.terminal_xs = [self.opti.parameter(self.nx, 1) for _ in range(self.Ns)]

        # Setting terminal constraints for the states (for each scenario)
        for i, xs in enumerate(self.xs_list):
            if self.terminal:
                self.opti.subject_to(0.95*self.terminal_xs[i] <= (xs[:,-1]  <= 1.05*self.terminal_xs[i]))

        # Define cost function
        J = 0

        # Loop through all Ns scenarios 
        for i in range(self.Ns):
            # Select the current scenario trajectory
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
            # The finite-horizon optimization problem for the current scenario
            for ll in range(0, self.Np):
                # The current sample trajectory
                pk = ps[:, ll]
                if self.rl_feedback:
                    uk = us[:, ll] + self.theta[:, ll]
                else:
                    uk = self.theta[:, ll]
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
                self.opti.subject_to(P[5, ll] >= self.ub_pen_w[0,2] * (ys[3, ll] - self.y_max[3]))

                # Cost function with penalties for the soft constraints
                delta_dw = xs[0, ll+1] - xs[0, ll]
                J -= compute_economic_reward(delta_dw, p, self.dt, uk)
                J += (P[0, ll]+ P[1, ll]+P[2, ll]+P[3, ll]+P[4, ll]+P[5, ll])

                # Input rate constraint
                if ll < self.Np-1:
                    if self.rl_feedback:                    
                        self.opti.subject_to(-self.du_max <= ((us[:, ll+1] + self.theta[:, ll+1]) - uk <=self.du_max))
                    else:
                        self.opti.subject_to(-self.du_max <= (self.theta[:, ll+1] - uk <=self.du_max))

            # Value Function insertion
            if self.use_trained_vf:
                J_terminal = self.vf_casadi_approx_func(OBS_NORM, TAYLOR_COEFS)
            else:
                J_terminal = 0
            J -= J_terminal

            # Constraints on intial state and input
            if self.rl_feedback:
                self.opti.subject_to(-self.du_max <= ((us[:, 0] + self.theta[:, 0]) - self.init_u <= self.du_max))  
            else:
                self.opti.subject_to(-self.du_max <= (self.theta[:, 0] - self.init_u <= self.du_max))

        # Expected value of the cost function over several samples
        J = J / self.Ns

        self.opti.minimize(J)
        self.opti.solver('ipopt', self.nlp_opts)


class Experiment:
    """
    Experiment manager for RL-SMPC closed-loop simulation and performance evaluation.

    This class manages the closed-loop simulation of a Reinforcement Learning Stochastic 
    Model Predictive Controller (RL-SMPC). It handles scenario 
    generation, parametric uncertainty, Taylor coefficient computation, and comprehensive 
    result tracking for both standard and open-loop prediction modes.

    The experiment workflow includes:
    - Initialization with RL-SMPC controller and simulation parameters
    - Scenario-based parametric uncertainty generation
    - RL policy unrolling for nominal trajectory generation
    - Taylor coefficient computation for value function approximation
    - Closed-loop simulation with receding horizon optimization
    - Comprehensive result collection and analysis
    - Data export in multiple formats (numpy arrays, pandas DataFrames, .npz files)

    Attributes:
        project_name (str): Name of the project for result organization
        save_name (str): Filename for saving experiment results
        mpc (SMPC): The RL-SMPC controller instance
        uncertainty_value (float): Level of parametric uncertainty for robustness testing
        p (np.ndarray): Model parameters
        rng (np.random.Generator): Random number generator for uncertainty
        x (np.ndarray): State trajectory over simulation period (nx, N+1)
        y (np.ndarray): Output trajectory over simulation period (ny, N+1)
        u (np.ndarray): Control input trajectory (nu, N+1)
        d (np.ndarray): Disturbance trajectory (weather data)
        uopt (np.ndarray): Optimal control sequences from MPC (nu, Np, N+1)
        J (np.ndarray): Cost values at each time step (1, N)
        p_samples_all (np.ndarray): All parametric samples (Ns, n_params, Np, N)
        dJdu (np.ndarray): Cost gradients (nu*Np, N)
        xs_opt_all (np.ndarray): All optimal state trajectories (Ns, nx, Np+1, N)
        ys_opt_all (np.ndarray): All optimal output trajectories (Ns, ny, Np+1, N)
        output (list): Optimization solver outputs
        rewards (np.ndarray): Net rewards (economic - penalties) (1, N)
        penalties (np.ndarray): Constraint violation penalties (1, N)
        econ_rewards (np.ndarray): Economic rewards (1, N)
        solver_times (np.ndarray): Solver computation times (1, N)
        exit_message (np.ndarray): Solver exit status codes (1, N)
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


        return (xk_samples, terminal_xs, uk_samples, end_points)

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

    def solve_nsmpc(self, order: str="zero") -> None:
        """
        Execute the complete RL-SMPC closed-loop simulation.

        This method performs the complete closed-loop simulation of the RL-SMPC controller
        over the entire simulation period. At each time step, it generates scenarios,
        computes optimal control inputs, and updates the system state with parametric uncertainty.

        The simulation workflow includes:
        1. Generate parametric uncertainty samples for all scenarios
        2. Unroll RL policy to generate nominal trajectories
        3. Compute Taylor coefficients for value function approximation
        4. Solve the scenario-based optimization problem
        5. Apply optimal control and simulate system evolution
        6. Update performance metrics and results

        Args:
            order (str): The order of the RL approximation scheme ("zero" or "first")

        Notes:
            - Uses tqdm progress bar for simulation monitoring
            - Implements scenario-based uncertainty handling
            - Supports both zero-order and first-order approximation schemes
            - Tracks comprehensive performance metrics
            - Provides robust error handling for optimization failures
            - Uses warm-starting for optimization efficiency
        """
        self.mpc.eval_env.reset()
        theta_init = np.zeros((self.mpc.nu, self.mpc.Np))

        for ll in tqdm(range(self.mpc.N)):
            p_samples = self.generate_psamples()
            if ll == 0:
                self.mpc.eval_env.set_env_state(self.x[:, ll], self.x[:,ll], self.u[:,ll], ll)
            else:
                self.mpc.eval_env.set_env_state(self.x[:, ll], self.x[:,ll-1], self.u[:,ll], ll)
            (xk_samples, terminal_xs, uk_samples, end_points) =\
                self.generate_samples(p_samples)

            # Get Taylor coefficients for all samples
            taylor_coefficients = self.get_taylor_coefficients(end_points)

            ds = self.d[:, ll:ll+self.mpc.Np+1]
            timestep = [ll]

            # we have to transpose p_samples since MPC_func expects matrix of shape (n_params, Np)
            p_sample_list = [p_samples[i].T for i in range(self.mpc.Ns)]


            # Call MPC function with all inputs as CasADi DM
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

            # Since the first RL sample always depends on x0 all the samples input (u) at t=0 will the same;
            if self.mpc.rl_feedback:
                us_opt = uk_samples[0][:,0] + theta_opt[:, 0]
            else:
                us_opt = theta_opt[:, 0]

            self.u[:, ll+1] = us_opt

            # Shift decision variable theta to warm-start the next optimization
            theta_init = np.concatenate([theta_opt[:, 1:], ca.reshape(theta_opt[:, -1], (self.mpc.nu, 1))], axis=1)

            # Simulate the system with parametric uncertainty
            params = parametric_uncertainty(self.p, self.uncertainty_value, self.rng)
            self.x[:, ll+1] = self.mpc.F(self.x[:, ll], self.u[:, ll+1], self.d[:, ll], params).toarray().ravel()
            self.y[:, ll+1] = self.mpc.g(self.x[:, ll+1]).toarray().ravel()

            # Compute the economic reward and penalties
            delta_dw = self.x[0, ll+1] - self.x[0, ll]
            econ_rew = compute_economic_reward(delta_dw, get_parameters(), self.mpc.dt, self.u[:, ll+1])
            penalties = self.mpc.compute_penalties(self.y[:, ll+1])

            # Update the results
            self.update_results(
                us_opt.reshape(-1,1), 
                J_mpc_1, [],
                econ_rew,
                penalties, 
                ll, 
                solver_time, 
                exit_message=exit_message
            )

    def solve_smpc_OL_predictions(self, order: str = "zero") -> None:
        """
        Execute RL-SMPC simulation with comprehensive open-loop prediction storage.

        This method performs the complete RL-SMPC closed-loop simulation while storing
        all open-loop predictions for each scenario at each time step. Unlike the standard
        simulation, this method focuses on capturing the full optimization solution for
        detailed analysis of the scenario-based RL-SMPC approach.

        The method performs the following workflow for each time step:
        1. Generate parametric uncertainty samples for all scenarios
        2. Unroll RL policy to generate nominal trajectories
        3. Compute Taylor coefficients for value function approximation
        4. Solve the scenario-based optimization problem
        5. Store complete open-loop predictions for all scenarios
        6. Apply optimal control and simulate system evolution
        7. Update performance metrics and results

        The stored open-loop predictions include:
        - Optimal state trajectories for all scenarios (xs_opt_all)
        - Optimal output trajectories for all scenarios (ys_opt_all)
        - Parametric uncertainty samples (p_samples_all)
        - Nominal state trajectories from RL policy (xs_samples_all)

        Args:
            order (str, optional): The order of the RL approximation scheme. 
                Currently only supports "zero" order. Defaults to "zero".
        """
        self.mpc.eval_env.reset()
        theta_init = np.zeros((self.mpc.nu, self.mpc.Np))
        self.xs_samples_all = np.zeros((self.mpc.Ns, self.mpc.nx, self.mpc.Np+1, self.mpc.N))

        # Simulate for twenty days and save every open-loop solution
        for ll in tqdm(range(20*24*2)):
            p_samples = self.generate_psamples()
            if ll == 0:
                self.mpc.eval_env.set_env_state(self.x[:, ll], self.x[:,ll], self.u[:,ll], ll)
            else:
                self.mpc.eval_env.set_env_state(self.x[:, ll], self.x[:,ll-1], self.u[:,ll], ll)
            (xk_samples, terminal_xs, uk_samples, end_points) =\
                self.generate_samples(p_samples)

            # Get Taylor coefficients for all samples
            taylor_coefficients = self.get_taylor_coefficients(end_points)

            ds = self.d[:, ll:ll+self.mpc.Np+1]
            timestep = [ll]

            # we have to transpose p_samples since MPC_func expects matrix of shape (n_params, Np)
            p_sample_list = [p_samples[i].T for i in range(self.mpc.Ns)]

            # Call MPC function with all inputs as CasADi DM
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

            # Since the first RL sample always depends on x0 all the samples input (u) at t=0 will the same;
            us_opt = uk_samples[0][:,0] + theta_opt[:, 0]
            self.u[:, ll+1] = us_opt

            # use previous theta solution as warm start
            theta_init = np.concatenate([theta_opt[:, 1:], ca.reshape(theta_opt[:, -1], (self.mpc.nu, 1))], axis=1)

            # Evolve State
            params = parametric_uncertainty(self.p, self.uncertainty_value, self.rng)

            self.x[:, ll+1] = self.mpc.F(self.x[:, ll], self.u[:, ll+1], self.d[:, ll], params).toarray().ravel()
            self.y[:, ll+1] = self.mpc.g(self.x[:, ll+1]).toarray().ravel()

            delta_dw = self.x[0, ll+1] - self.x[0, ll]
            econ_rew = compute_economic_reward(delta_dw, get_parameters(), self.mpc.dt, self.u[:, ll+1])
            penalties = self.mpc.compute_penalties(self.y[:, ll+1])

            xs_opt = np.array(xs_opt).reshape(self.mpc.Ns, self.mpc.nx, self.mpc.Np+1)
            ys_opt = np.array(ys_opt).reshape(self.mpc.Ns, self.mpc.ny, self.mpc.Np)

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
            self.xs_samples_all[:, :, :, ll] = xk_samples

            self.update_results(
                uk_samples[0][:,:] + theta_opt[:, :], 
                J_mpc_1, [], 
                econ_rew, 
                penalties, 
                ll, 
                solver_time, 
                exit_message
            )

        # Save the open-loop predictions to file after all timesteps
        self.save_open_loop_predictions()

    def save_open_loop_predictions(self):
        """
        Save open-loop predictions to .npz file format.

        This method saves all open-loop prediction data to a compressed .npz file
        for efficient storage and later analysis. The file includes comprehensive
        trajectory data for all scenarios and time steps.

        The saved data includes:
        - xs_opt_all: Optimal state trajectories for all scenarios
        - ys_opt_all: Optimal output trajectories for all scenarios
        - us_opt: Optimal control sequences
        - p_samples_all: Parametric uncertainty samples
        - xs_samples_all: Nominal state trajectories from RL policy
        - d: Disturbance trajectories
        - x, y, u: Closed-loop trajectories
        - J, econ_rewards, penalties, rewards: Performance metrics
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
            xs_samples_all=self.xs_samples_all,
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


    def update_results(
        self, 
        us_opt, 
        Js_opt,
        sol,
        eco_rew,
        penalties,
        ll,
        solver_time,
        exit_message=None
    ):
        """
        Update experiment results with current optimization outcomes.

        This method stores the results from the current RL-SMPC optimization step,
        including optimal controls, cost values, economic rewards, penalties,
        solver performance metrics, and exit status.

        Args:
            us_opt (np.ndarray): Optimal control sequence from RL-SMPC
            Js_opt (float): Optimal cost value
            sol (list): Solver output information (currently unused)
            eco_rew (float): Economic reward for current step
            penalties (float): Constraint violation penalties
            ll (int): Current simulation time step
            solver_time (float): Time taken by the optimization solver
            exit_message (str, optional): Solver exit status message

        Notes:
            - Converts solver exit messages to binary success/failure codes
            - Stores all trajectories and metrics for post-processing
            - Calculates net rewards as economic rewards minus penalties
            - Tracks solver performance for optimization analysis
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
    terminal=True,
    rl_feedback=True,
    Ns=10,
):
    """
    Creates a instance of the RL-SMPC.

    Correctly sets the random number generator to sample parametric uncertainty.

    Args:
        h (int): Prediction horizon in hours
        env_params (dict): Environment parameters for the greenhouse simulation
        mpc_params (dict): Parameters for the MPC controller
        rl_env_params (dict): Parameters for the RL environment
        algoritm (str): RL algorithm to use
        env_path (str): Path to the normalized environment file
        rl_model_path (str): Path to the trained RL model
        vf_path (str): Path to the trained value function
        run (int): Simulation run number for seeding
        use_trained_vf (bool): Whether to use trained value function as terminal cost
        terminal (bool): Whether to terminal region constraint
        rl_feedback (bool): Whether to use RL policy as feedback law
        Ns (int): Number of scenarios to generate

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
        terminal=terminal,
        rl_feedback=rl_feedback,
    )

def load_experiment_parameters(
    project,
    env_id,
    algorithm,
    mode,
    model_name,
    uncertainty_value
):
    """
    Load and prepare all parameters needed for the RL-SMPC experiment.

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
    parser.add_argument("--terminal", action="store_true")
    parser.add_argument("--uncertainty_value", type=float, required=True)
    parser.add_argument("--mode", type=str, choices=['deterministic', 'stochastic'], required=True)
    parser.add_argument("--order", type=str, choices=["zero", "first"], required=True)
    parser.add_argument("--rl_feedback", action="store_true", help="Use RL feedback in MPC")
    args = parser.parse_args()

    save_path = f"data/{args.project}/stochastic/rlsmpc"
    os.makedirs(save_path, exist_ok=True)

    env_params, mpc_params, rl_env_params, env_path, rl_model_path, vf_path = \
        load_experiment_parameters(args.project, args.env_id, args.algorithm, args.mode, args.model_name, args.uncertainty_value)

    # run the experiment
    H = [1, 2, 3, 4, 5, 6]
    for h in H:
        save_name = f"{args.model_name}-{args.save_name}-{h}H-{args.uncertainty_value}.csv"
        p = get_parameters()
        exp_rng = np.random.default_rng(666) 
        print("vf", args.use_trained_vf)
        print("terminak", args.terminal)
        print("rl_feedback", args.rl_feedback)
        rl_mpc = create_rl_smpc(
            h=h, 
            env_params=env_params, 
            mpc_params=mpc_params,
            rl_env_params=rl_env_params, 
            algorithm=args.algorithm,
            env_path=env_path, 
            rl_model_path=rl_model_path, 
            vf_path=vf_path,
            run=0,
            use_trained_vf=args.use_trained_vf,
            terminal=args.terminal,
            rl_feedback=args.rl_feedback,
        )

        rl_mpc.define_zero_order_snlp(p)
        exp = Experiment(rl_mpc, save_name, args.project, args.weather_filename, args.uncertainty_value, p, exp_rng)
        exp.solve_nsmpc(args.order)
        exp.save_results(save_path)
