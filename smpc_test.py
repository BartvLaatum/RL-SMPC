import os
import argparse
from typing import Any, Dict, List
from time import time

from tqdm import tqdm
import numpy as np
import pandas as pd
import casadi as ca

from common.noise import parametric_uncertainty
from common.utils import (
    load_disturbances, 
    compute_economic_reward, 
    get_parameters,
    load_env_params,
    load_mpc_params,
    define_model,
    co2dens2ppm,
    vaporDens2rh
)

class SMPC:
    """
    A class to represent a Model Predictive Controller (SMPC) for nonlinear systems.

    The SMPC class utilizes CasADi's Opti stack to define and solve an optimization problem
    that minimizes a cost function while satisfying system dynamics and constraints over a 
    specified prediction horizon.

    Attributes:
        opti (casadi.Opti): The optimization problem instance.
        nu (int): Number of control inputs.
        nx (int): Number of state variables.
        ny (int): Number of output variables.
        Np (int): Prediction horizon length.
        x_initial (np.ndarray): Initial state of the system.
        u_initial (np.ndarray): Initial control input.
        lb_pen_w (np.ndarray): Lower bounds for penalty weights.
        ub_pen_w (np.ndarray): Upper bounds for penalty weights.
        du_max (float): Maximum allowable change in control input.
        nlp_opts (dict): Options for the IPOPT solver.

    Methods:
        define_nlp(p: Dict[str, Any]) -> None:
            Defines the nonlinear programming (NLP) problem, including decision variables,
            parameters, constraints, and the cost function.
            Defines a callable function `SMPC_func` that can be used to solve the optimization problem.

    """
    def __init__(
            self,
            nx: int,
            nu: int,
            ny: int,
            nd: int,
            dt: float,
            n_days: int,
            x0: List[float],
            u0: List[float],
            Np: int,
            Ns: int,
            start_day: int,
            uncertainty_value: float,
            lb_pen_w: List[float],
            ub_pen_w: List[float],
            constraints: Dict[str, Any],
            nlp_opts: Dict[str, Any],
            rng,
            ) -> None:
        self.x_initial = x0
        self.u_initial = u0
        self.nx = nx
        self.nu = nu
        self.ny = ny
        self.nd = nd
        self.dt = dt
        self.nDays = n_days
        self.Np = Np
        self.L = n_days * 86400
        self.start_day = start_day

        self.t = np.arange(0, self.L, self.dt)
        self.N = len(self.t)
        self.lb_pen_w = np.expand_dims(lb_pen_w, axis=0)
        self.ub_pen_w = np.expand_dims(ub_pen_w, axis=0)
        self.Ns = Ns
        self.uncertainty_value = uncertainty_value
        self.lbg = []
        self.ubg = []
        self.constraints = constraints
        self.nlp_opts = nlp_opts
        self.rng = rng

        # initialize the boundaries for the optimization problem
        self.init_nmpc()

        self.F, self.g = define_model(self.dt, self.x_min, self.x_max)

    def init_nmpc(self):
        """Initialize the nonlinear SMPC problem.
        Initialise the constraints, bounds, and the optimization problem.
        """
        self.x_min = np.array(
            [
                self.constraints["W_min"],
                self.constraints["state_co2_min"],
                self.constraints["state_temp_min"],
                self.constraints["state_vp_min"]
            ],
            dtype=np.float32
        )
        self.x_max = np.array(
            [
                self.constraints["W_max"],
                self.constraints["state_co2_max"],
                self.constraints["state_temp_max"],
                self.constraints["state_vp_max"]
            ],
        )

        self.y_min = np.array([self.constraints["W_min"]*1e3, self.constraints["co2_min"], self.constraints["temp_min"], self.constraints["rh_min"]])
        self.y_max = np.array([self.constraints["W_max"]*1e3, self.constraints["co2_max"], self.constraints["temp_max"], self.constraints["rh_max"]])

        # control input constraints vector
        self.u_min = np.array([self.constraints["co2_supply_min"], self.constraints["vent_min"], self.constraints["heat_min"]])
        self.u_max = np.array([self.constraints["co2_supply_max"], self.constraints["vent_max"], self.constraints["heat_max"]])
        # lower and upper bound change of u 
        self.du_max = np.divide(self.u_max, [10, 10, 10])

        # initial values of the decision variables (control signals) in the optimization
        self.u0 = np.zeros((self.nu, self.Np)) # this can be moved to the shooting function method.

    def solve_ocp(self, x0, u0, ds, p_samples, u_guess=None, x_guess=None):
        """
        Sets solver values for the optimization problem.
        This function initializes various parameters and values required for the scenario-based
        Model Predictive Control (MPC) solver, including initial states, control inputs, and weather trajectory.
        """
        self.opti.set_value(self.x0, x0)
        self.opti.set_value(self.init_u, u0)
        self.opti.set_value(self.ds, ds)

        for i, p_sample in enumerate(self.p_samples):
            self.opti.set_value(p_sample, p_samples[i].T)

        if u_guess is not None:
            self.opti.set_initial(self.us, u_guess)
        if x_guess is not None:
            for i, xs in enumerate(x_guess):
                self.opti.set_initial(self.xs_list[i], xs)

        start_time = time()
        try:
            solution = self.opti.solve()
        except RuntimeError as err:
            # Recover from the failed solve: you might use the last iterate available via opti.debug
            print("Solver failed with error:", err)
            solution = self.opti.debug  # Returns the current iterate even if not converged
        solver_time = time() - start_time

        exit_message = solution.stats()['return_status']
        xs_opt = [solution.value(self.xs_list[i]) for i in range(self.Ns)]
        ys_opt = [solution.value(self.ys_list[i]) for i in range(self.Ns)]
        us = solution.value(self.us)
        J = solution.value(self.opti.f)

        return xs_opt, ys_opt, us, J, solver_time, exit_message

    def define_nlp(self, p: Dict[str, Any]) -> None:
        """
        Define the optimization problem for the nonlinear SMPC using CasADi's Opti stack.
        Including the cost function, constraints, and bounds.
        """
        b = 3                   # branching factor (# samples per node)

        # total number of full “leaf” scenarios
        Ns = b**self.Np         # 3**2 = 9

        # Create an Opti instance
        self.opti = ca.Opti()

        # Number of penalties (CO2 lb, CO2 ub, Temp lb, Temp ub, humidity lb, humidity ub)
        num_penalties = 6

        # 1) for each stage k=0..Np-1 create one control‐variable per tree‐node
        #    stage 0: 3**0 = 1 node; stage 1: 3**1 = 3 nodes
        self.u_nodes = []
        for k in range(self.Np):
            n_nodes_k = b**k
            Uk = self.opti.variable(self.nu, n_nodes_k, name=f"u_node_{k}")
            self.u_nodes.append(Uk)

        # Decision Variables (Control inputs, slack variables, states, outputs)
        self.us = self.opti.variable(self.nu, self.Np)  # Control inputs (nu x Np)
        self.xs_list = [self.opti.variable(self.nx, self.Np+1) for _ in range(self.Ns)]
        self.ys_list = [self.opti.variable(self.ny, self.Np) for _ in range(self.Ns)]
        Ps     = [self.opti.variable(num_penalties, self.Np) for _ in range(self.Ns)]

        # Parameters
        self.p_samples = [self.opti.parameter(p.shape[0], self.Np) for _ in range(self.Ns)]
        self.x0 = self.opti.parameter(self.nx, 1)       # Initial state
        self.ds = self.opti.parameter(self.nd, self.Np) # Disturbances
        self.init_u = self.opti.parameter(self.nu, 1)   # Initial control input
        # self.noise_samples = [
        #     self.opti.parameter(self.ny, self.Np) for _ in range(self.Ns)
        # ]

        Js = 0
        for i in range(self.Ns):
            xs = self.xs_list[i]
            ys = self.ys_list[i]
            # ps = self.p_samples[i]
            P = Ps[i]

            # Initial Condition Constraint
            self.opti.subject_to(xs[:,0] == self.x0)

            for ll in range(self.Np):
                # pk = ps[:, ll]
                node_id = i // (b**(self.Np - k))
                # pick the shared control for that node
                u_k = self.u_nodes[k][:, node_id]

                self.opti.subject_to(xs[:, ll+1] == self.F(xs[:, ll], u_k, self.ds[:, ll], p))
                self.opti.subject_to(ys[:, ll] == self.g(xs[:, ll+1]))
                    
                # Input   Contraints
                self.opti.subject_to(self.u_min <= (self.us[:,ll] <= self.u_max))

                self.opti.subject_to(P[:, ll] >= 0)
                self.opti.subject_to(P[0, ll] >= self.lb_pen_w[0,0] * (self.y_min[1] - ys[1, ll]))
                self.opti.subject_to(P[1, ll] >= self.ub_pen_w[0,0] * (ys[1, ll] - self.y_max[1]))
                self.opti.subject_to(P[2, ll] >= self.lb_pen_w[0,1] * (self.y_min[2] - ys[2, ll]))
                self.opti.subject_to(P[3, ll] >= self.ub_pen_w[0,1] * (ys[2, ll] - self.y_max[2]))
                self.opti.subject_to(P[4, ll] >= self.lb_pen_w[0,2] * (self.y_min[3] - ys[3, ll]))
                self.opti.subject_to(P[5, ll] >= self.ub_pen_w[0,2] * (ys[3, ll] - (self.y_max[3]-2.0)))

                # COST FUNCTION WITH PENALTIES
                delta_dw = xs[0, ll+1] - xs[0, ll]
                Js -= compute_economic_reward(delta_dw, p, self.dt, self.us[:,ll])
                Js += (P[0, ll]+ P[1, ll]+P[2, ll]+P[3, ll]+P[4, ll]+P[5, ll])

                if ll < self.Np-1:
                    next_node_id = i // (b**(Np - (k+1)))
                    u_next = self.u_nodes[k+1][:, next_node_id]
                    self.opti.subject_to(-self.du_max <= (u_next - u_k))
                    self.opti.subject_to((u_next - u_k) <= self.du_max)
                    # self.opti.subject_to(-self.du_max<=(self.us[:,ll+1] - self.us[:,ll]<=self.du_max))     # Change in input Constraint

        Js = Js / self.Ns

        self.opti.subject_to(-self.du_max <= (self.us[:,0] - self.init_u <= self.du_max))  
        self.opti.minimize(Js)

        self.opti.solver('ipopt', self.nlp_opts)

    def constraint_violation(self, y: np.ndarray):
        """
        Function that computes the absolute penalties for violating system constraints.
        System constraints are currently non-dynamical, and based on observation bounds of gym environment.
        We do not look at dry mass bounds, since those are non-existent in real greenhouse.
        """
        lowerbound = self.y_min[1:] - y[1:]
        lowerbound[lowerbound < 0] = 0
        upperbound = y[1:] - self.y_max[1:]
        upperbound[upperbound < 0] = 0

        return lowerbound, upperbound

    def compute_penalties(self, y):
        lowerbound, upperbound = self.constraint_violation(y)
        penalties = np.dot(self.lb_pen_w, lowerbound) + np.dot(self.ub_pen_w, upperbound)
        return np.sum(penalties)


class Experiment:
    """Experiment manager to test the closed loop performance of SMPC.

    Attributes:
        project_name (str): The name of the project.
        save_name (str): The name under which results will be saved.
        mpc (SMPC): The SMPC controller instance.
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
            Solve the nonlinear SMPC problem.

        update_results(us_opt: np.ndarray, Js_opt: float, sol: Dict[str, Any], step: int) -> None:
            Update the results after solving the SMPC problem for a step.

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
            self.mpc.Np,
            self.mpc.nd
        )

        # Store open-loop predictions for all scenarios
        self.uopt = np.zeros((mpc.nu, mpc.Np, mpc.N+1))
        self.p_samples_all = np.zeros((mpc.Ns, 34, mpc.Np, mpc.N))
        self.J = np.zeros((1, mpc.N))
        self.xs_opt_all = np.zeros((mpc.Ns, mpc.nx, mpc.Np+1, mpc.N))
        self.ys_opt_all = np.zeros((mpc.Ns, mpc.ny, mpc.Np+1, mpc.N))
        self.dJdu = np.zeros((mpc.nu*mpc.Np, mpc.N))
        self.output = []
        self.penalties = np.zeros((1, mpc.N))
        self.econ_rewards = np.zeros((1, mpc.N))
        self.rewards = np.zeros((1, mpc.N))
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

    def initial_guess_xs(self, p_samples, x0, u_guess, ds) -> List[np.ndarray]:
        """
        Generate initial guesses for the states based on the provided parametric samples.

        Args:
            p_samples (List[np.ndarray]): A list of parametric samples for each scenario.
            x0 (np.ndarray): The initial state of the system.
            u_guess (np.ndarray): The initial guess for the control inputs.
            ds (np.ndarray): The disturbance trajectory.

        Returns:
            List: An List of arrays with initial guesses for the state trajectories for each realization of uncertainty.
        """
        x_initial_guess = []
        for i in range(self.mpc.Ns):
            xs = np.zeros((self.mpc.nx, self.mpc.Np + 1))
            xs[:, 0] = x0
            for ll in range(self.mpc.Np):
                xs[:, ll + 1] = self.mpc.F(
                    xs[:, ll],
                    u_guess[:, ll],
                    ds[:, ll],
                    p_samples[i][ll]
                ).toarray().ravel()
            x_initial_guess.append(xs)
        return x_initial_guess

    def solve_nmpc(self) -> None:
        """Solve the nonlinear SMPC problem.

        Args:
            p (Dict[str, Any]): the model parameters

        Returns:
            None
        """
        u_initial_guess = np.ones((self.mpc.nu, self.mpc.Np)) * np.array(self.mpc.u_initial).reshape(self.mpc.nu, 1)

        for ll in tqdm(range(self.mpc.N)):
            p_samples = self.generate_psamples()
            x_initial_guess = self.initial_guess_xs(p_samples, self.x[:, ll], u_initial_guess, self.d[:, ll:ll+self.mpc.Np])
            xs_opt, ys_opt, us_opt, J_opt, solver_time, exit_message = self.mpc.solve_ocp(
                self.x[:, ll],
                self.u[:, ll],
                self.d[:, ll:ll+self.mpc.Np],
                p_samples,
                u_guess=u_initial_guess,
                x_guess=x_initial_guess
            )

            self.u[:, ll+1] = us_opt[:, 0]

            params = parametric_uncertainty(self.p, self.uncertainty_value, self.rng)

            self.x[:, ll+1] = self.mpc.F(self.x[:, ll], self.u[:, ll+1], self.d[:, ll], params).toarray().ravel()
            self.y[:, ll+1] = self.mpc.g(self.x[:, ll+1]).toarray().ravel() + self.rng.random(self.mpc.ny) * 0.1

            delta_dw = self.x[0, ll+1] - self.x[0, ll]
            econ_rew = compute_economic_reward(delta_dw, get_parameters(), self.mpc.dt, self.u[:, ll+1])
            penalties = self.mpc.compute_penalties(self.y[:, ll+1])

            self.update_results(us_opt, J_opt, [], econ_rew, penalties, ll, solver_time, exit_message=exit_message)

            u_initial_guess = np.concatenate([us_opt[:, 1:], us_opt[:, -1][:, None]], axis=1)

    def solve_smpc_OL_predictions(self) -> None:
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
        for ll in tqdm(range(1)):
            print(f"solving timestep {ll}")
            p_samples = self.generate_psamples()

            # we have to transpose p_samples since MPC_func expects matrix of shape (n_params, Np)
            p_sample_list = [p_samples[i].T for i in range(self.mpc.Ns)]
            # breakpoint()
            us_opt, xs_opt, ys_opt, J_opt = self.mpc.SMPC_func(
                self.x[:, ll],
                self.d[:, ll:ll+self.mpc.Np], 
                self.u[:, ll], 
                *p_sample_list
            )

            self.u[:, ll+1] = us_opt[:, 0].toarray().ravel()
            params = parametric_uncertainty(self.p, self.mpc.uncertainty_value, self.rng)

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

            self.update_results(us_opt, J_opt, [], econ_rew, penalties, ll)

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

    def update_results(self, us_opt, Js_opt, sol, eco_rew, penalties, step, solver_time, exit_message=None):
        """
        Args:
            uopt (_type_): _description_
            J (_type_): _description_
            output (_type_): _description_
            step (_type_): _description_
        """
        if exit_message == "Solve_Succeeded" or exit_message == "Solved_To_Acceptable_Level":
            exit_message = 0
        else:
            exit_message = 1
        self.uopt[:,:, step] = us_opt
        self.J[:, step] = Js_opt
        self.output.append(sol)
        self.econ_rewards[:, step] = eco_rew
        self.penalties[:, step] = penalties
        self.rewards[:, step] = eco_rew - penalties
        self.solver_times[:, step] = solver_time
        self.exit_message[:, step] = exit_message

    def get_results(self, run):
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

    def retrieve_results(self, run=0):
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
        df.to_csv(f"{save_path}/{self.save_name}.csv", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str)
    parser.add_argument("--env_id", type=str, default="LettuceGreenhouse")
    parser.add_argument("--save_name", type=str)
    parser.add_argument("--uncertainty_value", type=float, required=True)
    parser.add_argument("--weather_filename", default="outdoorWeatherWurGlas2014.csv", type=str)
    args = parser.parse_args()

    save_path = f"data/{args.project}/stochastic/smpc"
    os.makedirs(save_path, exist_ok=True)

    env_params = load_env_params(args.env_id)
    mpc_params = load_mpc_params(args.env_id)
    mpc_params["uncertainty_value"] = args.uncertainty_value

    H = [1, 2, 3, 4, 5, 6]
    mpc_params["Ns"] = 5
    env_params["n_days"] = 20
    for h in H:
        mpc_rng = np.random.default_rng(42)
        exp_rng = np.random.default_rng(666)
        save_name = f"{args.save_name}-{h}H-{args.uncertainty_value}"
        mpc_params["rng"] = mpc_rng
        mpc_params["Np"] = int(h * 3600 / env_params["dt"])

        # p = DefineParameters()
        p = get_parameters()
        mpc = SMPC(**env_params, **mpc_params)
        mpc.define_nlp(p)

        exp = Experiment(mpc, save_name, args.project, args.weather_filename, args.uncertainty_value, p, exp_rng)
        exp.solve_nmpc()
        exp.save_results(save_path)
