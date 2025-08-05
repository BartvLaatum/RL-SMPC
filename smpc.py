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
    Stochastic Model Predictive Controller (SMPC) for nonlinear greenhouse systems.

    This class implements a scenario-based Stochastic Model Predictive Controller for
    greenhouse climate control, utilizing CasADi's Opti stack to formulate and solve
    nonlinear optimization problems with parametric uncertainty. The controller manages
    CO2 supply, ventilation, and heating to optimize crop growth while maintaining
    environmental constraints under uncertainty.

    The SMPC formulation includes:
    - Scenario-based uncertainty handling with multiple parameter realizations
    - Economic objective function with crop growth rewards and control costs
    - Soft constraints via slack variables on CO2 concentration, temperature, and humidity
    - Hard constraints on control inputs and their rates of change
    - Expected value optimization across multiple scenarios

    Attributes:
        nx (int): Number of state variables (4: dry mass, CO2, temperature, vapor pressure)
        nu (int): Number of control inputs (3: CO2 supply, ventilation, heating)
        ny (int): Number of output variables (4: dry mass, CO2, temperature, humidity)
        nd (int): Number of disturbance variables (weather conditions)
        dt (float): Time step for discretization (seconds)
        nDays (int): Simulation duration in days
        Np (int): Prediction horizon length
        Ns (int): Number of scenarios for uncertainty handling
        L (int): Total simulation time in seconds
        start_day (int): Starting day for weather data
        t (np.ndarray): Time vector for the simulation
        N (int): Total number of time steps
        x_initial (List[float]): Initial state values
        u_initial (List[float]): Initial control input values
        lb_pen_w (np.ndarray): Lower bounds for penalty weights
        ub_pen_w (np.ndarray): Upper bounds for penalty weights
        uncertainty_value (float): Level of parametric uncertainty
        constraints (Dict[str, Any]): System constraints dictionary
        nlp_opts (Dict[str, Any]): IPOPT solver options
        rng (np.random.Generator): Random number generator for uncertainty
        x_min, x_max (np.ndarray): State bounds
        y_min, y_max (np.ndarray): Output bounds
        u_min, u_max (np.ndarray): Control input bounds
        du_max (np.ndarray): Maximum control input rate of change
        F, g (callable): System dynamics and output functions
        opti (casadi.Opti): Optimization problem instance
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
        """
        Initialize the stochastic MPC problem by setting up constraints and bounds.

        This method sets up the system constraints and bounds for the stochastic MPC
        optimization problem, including state bounds, output bounds, control input bounds,
        and rate of change constraints.

        Notes:
            - Sets up state bounds for dry mass, CO2, temperature, and vapor pressure
            - Defines output bounds for economic variables and environmental conditions
            - Establishes control input bounds for CO2 supply, ventilation, and heating
            - Configures rate of change constraints for smooth control actions
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
        Solve the optimal control problem with scenario-based constraints.

        This method sets up and solves the scenario-based stochastic MPC optimization
        problem. It initializes all scenario parameters, sets initial guesses for
        optimization variables, and solves the resulting nonlinear programming problem.

        The method performs the following steps:
        1. Sets initial state and control input parameters
        2. Assigns parametric uncertainty samples for each scenario
        3. Sets initial guesses for optimization variables (if provided)
        4. Solves the nonlinear programming problem using IPOPT
        5. Extracts optimal trajectories and cost value
        6. Handles solver failures gracefully with debug information

        Args:
            x0 (np.ndarray): Initial state vector of shape (nx,)
            u0 (np.ndarray): Initial control input vector of shape (nu,)
            ds (np.ndarray): Disturbance trajectory of shape (nd, Np)
            p_samples (list): List of parametric uncertainty samples for each scenario.
                Each element should be of shape (n_params, Np)
            u_guess (np.ndarray, optional): Initial guess for control inputs of shape (nu, Np).
                If provided, can improve solver convergence.
            x_guess (list, optional): List of initial guesses for state trajectories.
                Each element should be of shape (nx, Np+1) for each scenario.

        Returns:
            tuple: Contains optimization results:
                - xs_opt (list): Optimal state trajectories for all scenarios.
                    List of Ns arrays, each of shape (nx, Np+1)
                - ys_opt (list): Optimal output trajectories for all scenarios.
                    List of Ns arrays, each of shape (ny, Np)
                - us (np.ndarray): Optimal control input trajectory of shape (nu, Np)
                - J (float): Optimal cost value (expected value across scenarios)
                - solver_time (float): Time taken by the optimization solver in seconds
                - exit_message (str): Solver exit status message
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
        Define the complete stochastic nonlinear programming problem.

        This method sets up the optimization problem for scenario-based stochastic MPC 
        using CasADi's Opti stack. It defines decision variables, constraints, and the 
        cost function for multiple scenarios with parametric uncertainty.

        The optimization problem includes:
        - Decision variables for control inputs, states, outputs, and slack variables
        - Input magnitude and rate constraints
        - Soft constraints with penalties for output variables
        - Economic objective function with expected value optimization

        Args:
            p (Dict[str, Any]): Dictionary containing model parameters used in system
                dynamics and cost function calculations
        """
        # Create an Opti instance
        self.opti = ca.Opti()

        # Number of penalties (CO2 lb, CO2 ub, Temp lb, Temp ub, humidity lb, humidity ub)
        num_penalties = 6

        # Decision Variables (Control inputs, slack variables, states, outputs)
        self.us = self.opti.variable(self.nu, self.Np)  # Control inputs (nu x Np)
        self.xs_list = [self.opti.variable(self.nx, self.Np+1) for _ in range(self.Ns)]
        self.ys_list = [self.opti.variable(self.ny, self.Np) for _ in range(self.Ns)]
        Ps     = [self.opti.variable(num_penalties, self.Np) for _ in range(self.Ns)]


        # Parameters
        self.p_samples = [self.opti.parameter(p.shape[0], self.Np) for _ in range(self.Ns)]
        self.x0 = self.opti.parameter(self.nx, 1)  # Initial state
        self.ds = self.opti.parameter(self.nd, self.Np)  # Disturbances
        self.init_u = self.opti.parameter(self.nu, 1)  # Initial control input

        # Define cost function
        Js = 0

        # Loop through all Ns scenarios 
        for i in range(self.Ns):
            # Select the current scenario trajectory            
            xs = self.xs_list[i]
            ys = self.ys_list[i]
            ps = self.p_samples[i]
            P = Ps[i]

            # Initial condition constraint
            self.opti.subject_to(xs[:,0] == self.x0)

            # The finite-horizon optimization problem for the current scenario
            for ll in range(self.Np):
                pk = ps[:, ll]

                # System dynamics and input constraints
                self.opti.subject_to(xs[:, ll+1] == self.F(xs[:, ll], self.us[:, ll], self.ds[:, ll], pk))
                self.opti.subject_to(ys[:, ll] == self.g(xs[:, ll+1]))
                self.opti.subject_to(self.u_min <= (self.us[:,ll] <= self.u_max))                   # Input   Contraints

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
                Js -= compute_economic_reward(delta_dw, p, self.dt, self.us[:,ll])
                Js += (P[0, ll]+ P[1, ll]+P[2, ll]+P[3, ll]+P[4, ll]+P[5, ll])

                # Input rate constraint
                if ll < self.Np-1:
                    self.opti.subject_to(-self.du_max<=(self.us[:,ll+1] - self.us[:,ll]<=self.du_max))     # Change in input Constraint

        self.opti.subject_to(-self.du_max <= (self.us[:,0] - self.init_u <= self.du_max))  

        # Expected value of the cost function over several samples
        Js = Js / self.Ns

        self.opti.minimize(Js)
        self.opti.solver('ipopt', self.nlp_opts)

    def constraint_violation(self, y: np.ndarray):
        """
        Compute constraint violations for output variables.

        This method calculates the absolute violations of system constraints for output
        variables. It focuses on environmental constraints (CO2, temperature, humidity)
        rather than dry mass bounds, as these are more relevant for real greenhouse operation.

        Args:
            y (np.ndarray): Output vector containing environmental variables

        Returns:
            tuple: Contains constraint violations:
                - lowerbound (np.ndarray): Lower bound violations
                - upperbound (np.ndarray): Upper bound violations
        """
        lowerbound = self.y_min[1:] - y[1:]
        lowerbound[lowerbound < 0] = 0
        upperbound = y[1:] - self.y_max[1:]
        upperbound[upperbound < 0] = 0

        return lowerbound, upperbound

    def compute_penalties(self, y):
        """
        Calculate penalty costs for constraint violations.

        This method computes the total penalty cost for violating system constraints
        by combining lower and upper bound violations with their respective penalty weights.

        Args:
            y (np.ndarray): Output vector containing environmental variables

        Returns:
            float: Total penalty cost for constraint violations
        """
        lowerbound, upperbound = self.constraint_violation(y)
        penalties = np.dot(self.lb_pen_w, lowerbound) + np.dot(self.ub_pen_w, upperbound)
        return np.sum(penalties)


class Experiment:
    """Experiment manager for closed-loop stochastic MPC performance evaluation.

    This class provides a comprehensive framework for conducting closed-loop simulations
    of stochastic Model Predictive Control (SMPC) systems. It manages the complete
    experimental workflow including parameter sampling, optimization, system simulation,
    and result storage.

    The experiment workflow includes:
    - Initialization of SMPC controller and simulation parameters
    - Scenario-based parametric uncertainty generation
    - RL policy unrolling for nominal trajectory generation
    - Taylor coefficient computation for value function approximation
    - Closed-loop simulation with receding horizon optimization
    - Comprehensive result collection and analysis
    - Data export in multiple formats (numpy arrays, pandas DataFrames, .npz files)


    Attributes:
        project_name (str): Project identifier for organizing results
        save_name (str): Base filename for saving experiment results
        mpc (SMPC): Stochastic MPC controller instance
        uncertainty_value (float): Uncertainty level for parametric sampling
        p: Model parameters dictionary
        rng: Random number generator for reproducible experiments
        x (np.ndarray): State trajectory of shape (nx, N+1)
        y (np.ndarray): Output trajectory of shape (ny, N+1)
        u (np.ndarray): Control input trajectory of shape (nu, N+1)
        d (np.ndarray): Disturbance trajectory of shape (nd, N+1)
        uopt (np.ndarray): Optimal control inputs from optimization
        J (np.ndarray): Cost values at each time step
        p_samples_all (np.ndarray): Parametric samples used in optimization
        xs_opt_all (np.ndarray): Open-loop state predictions for all scenarios
        ys_opt_all (np.ndarray): Open-loop output predictions for all scenarios
        dJdu (np.ndarray): Cost gradients (nu*Np, N)
        output (list): Optimization solver output information
        rewards (np.ndarray): Net rewards (economic - penalties) at each time step
        econ_rewards (np.ndarray): Economic rewards at each time step
        penalties (np.ndarray): Constraint violation penalties at each time step
        solver_times (np.ndarray): Optimization solver computation times
        exit_message (np.ndarray): Solver exit status codes
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
        self.J = np.zeros((1, mpc.N))
        self.p_samples_all = np.zeros((mpc.Ns, 34, mpc.Np, mpc.N))
        self.xs_opt_all = np.zeros((mpc.Ns, mpc.nx, mpc.Np+1, mpc.N))
        self.ys_opt_all = np.zeros((mpc.Ns, mpc.ny, mpc.Np+1, mpc.N))
        self.dJdu = np.zeros((mpc.nu*mpc.Np, mpc.N))
        self.output = []
        self.rewards = np.zeros((1, mpc.N))
        self.econ_rewards = np.zeros((1, mpc.N))
        self.penalties = np.zeros((1, mpc.N))
        self.solver_times = np.zeros((1, mpc.N))
        self.exit_message = np.zeros((1, mpc.N))

    def generate_psamples(self) -> List[np.ndarray]:
        """
        Generate parametric uncertainty samples for scenario-based SMPC.

        The method generates Ns scenarios, each containing Np parametric samples
        corresponding to the prediction horizon. Each sample is generated using the
        parametric_uncertainty function with the specified uncertainty level.

        Args:
            None

        Returns:
            List[np.ndarray]: List of parametric samples for each scenario.
                Each element has shape (n_params, Np) where n_params is the number
                of uncertain parameters in the model.
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
        Generate initial state trajectory guesses for scenario-based optimization.

        The method simulates the system dynamics for each scenario using the provided
        control input sequence and the corresponding parametric uncertainty samples.
        This provides a reasonable starting point for the optimization solver.

        Args:
            p_samples (List[np.ndarray]): List of parametric uncertainty samples for
                each scenario. Each element has shape (n_params, Np)
            x0 (np.ndarray): Initial state vector of shape (nx,)
            u_guess (np.ndarray): Initial guess for control input trajectory of shape (nu, Np)
            ds (np.ndarray): Disturbance trajectory of shape (nd, Np)

        Returns:
            List[np.ndarray]: List of initial state trajectory guesses for each scenario.
                Each element has shape (nx, Np+1) representing the state trajectory
                over the prediction horizon plus the initial state.
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
        """
        Execute closed-loop stochastic MPC simulation.

        This method performs a complete closed-loop simulation of the stochastic MPC
        system over the entire simulation horizon. At each time step, it generates
        parametric uncertainty samples, solves the SMPC optimization problem, applies
        the first control action, and simulates the system evolution.

        The method implements the receding horizon control strategy:
        1. Generate parametric uncertainty samples for all scenarios
        2. Create initial guesses for optimization variables
        3. Solve the SMPC optimization problem
        4. Apply the first control action to the system
        5. Simulate system evolution with uncertainty
        6. Compute performance metrics (economic rewards, penalties)
        7. Update results and prepare for next time step

        Args:
            None

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
            self.y[:, ll+1] = self.mpc.g(self.x[:, ll+1]).toarray().ravel()

            delta_dw = self.x[0, ll+1] - self.x[0, ll]
            econ_rew = compute_economic_reward(delta_dw, get_parameters(), self.mpc.dt, self.u[:, ll+1])
            penalties = self.mpc.compute_penalties(self.y[:, ll+1])

            self.update_results(us_opt, J_opt, [], econ_rew, penalties, ll, solver_time, exit_message=exit_message)

            # Shift predicted control trajectory 
            u_initial_guess = np.concatenate([us_opt[:, 1:], us_opt[:, -1][:, None]], axis=1)

    def solve_smpc_OL_predictions(self) -> None:
        """
        Execute closed-loop SMPC simulation with comprehensive open-loop prediction storage.

        This method performs a complete closed-loop simulation of the stochastic MPC
        system while storing detailed open-loop predictions for all scenarios at each
        time step. This enables comprehensive analysis of the controller's prediction
        behavior and scenario-based decision making.

        The method extends the standard SMPC simulation by:
        1. Storing complete open-loop state and output predictions for all scenarios
        2. Recording parametric uncertainty samples used in optimization
        3. Maintaining prediction history for the entire simulation horizon
        4. Enabling detailed analysis of scenario-based control decisions

        Args:
            None

        Returns:
            None
        """
        u_initial_guess = np.ones((self.mpc.nu, self.mpc.Np)) * np.array(self.mpc.u_initial).reshape(self.mpc.nu, 1)
        for ll in tqdm(range(20*24*2)):
            p_samples = self.generate_psamples()

            x_initial_guess = self.initial_guess_xs(p_samples, self.x[:, ll], u_initial_guess, self.d[:, ll:ll+self.mpc.Np])

            # we have to transpose p_samples since MPC_func expects matrix of shape (n_params, Np)
            p_sample_list = [p_samples[i].T for i in range(self.mpc.Ns)]

            xs_opt, ys_opt, us_opt, J_opt, solver_time, exit_message = self.mpc.solve_ocp(
                self.x[:, ll],
                self.u[:, ll],
                self.d[:, ll:ll+self.mpc.Np],
                p_samples,
                u_guess=u_initial_guess,
                x_guess=x_initial_guess
            )

            self.u[:, ll+1] = us_opt[:, 0]
            params = parametric_uncertainty(self.p, self.mpc.uncertainty_value, self.rng)

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

            self.update_results(us_opt, J_opt, [], econ_rew, penalties, ll, solver_time, exit_message)
            u_initial_guess = np.concatenate([us_opt[:, 1:], us_opt[:, -1][:, None]], axis=1)

        # Save the open-loop predictions to file after all timesteps
        self.save_open_loop_predictions()

    def save_open_loop_predictions(self):
        """
        Save comprehensive open-loop predictions to file for post-processing analysis.

        This method saves all open-loop predictions, trajectories, and performance
        metrics to a compressed NumPy file (.npz format) for efficient storage and
        later analysis. The saved data includes state predictions, output predictions,
        parametric samples, and performance metrics for all scenarios and time steps.

        The saved data enables comprehensive analysis of:
        - Scenario-based prediction behavior
        - Controller performance across different uncertainty realizations
        - Optimization solution quality and convergence
        - Economic performance and constraint satisfaction

        Args:
            None

        Returns:
            None
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
        Update experiment results with current optimization step data.

        This method stores the results from a single optimization step including
        optimal control inputs, cost values, performance metrics, and solver
        information. It processes solver exit messages and computes net rewards
        from economic rewards and penalties.

        Args:
            us_opt (np.ndarray): Optimal control input trajectory from optimization
            Js_opt (float): Optimal cost value from optimization
            sol (list): Optimization solver output information
            eco_rew (float): Economic reward for current time step
            penalties (float): Constraint violation penalties for current time step
            step (int): Current simulation time step
            solver_time (float): Time taken by optimization solver in seconds
            exit_message (str, optional): Solver exit status message

        Returns:
            None
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
        Generate comprehensive results array for simulation analysis.

        This method creates a 2D array containing all closed-loop trajectories,
        performance metrics, and simulation data for a complete experiment run.
        It processes weather data to appropriate units and organizes all data
        into a structured format suitable for analysis and visualization.

        The method transforms and organizes:
        - Time series data (converted to days)
        - State trajectories for all system states
        - Output trajectories for all system outputs
        - Control input trajectories
        - Disturbance trajectories (weather data)
        - Performance metrics (cost, rewards, penalties)
        - Solver performance data (times, success rates)

        Args:
            run (int): Run identifier for the simulation experiment

        Returns:
            np.ndarray: 2D array with shape (N, n_columns) containing all
                simulation data with run ID in the last column. Each row represents
                a time step with all corresponding trajectory and performance data.
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
        Create pandas DataFrame with comprehensive simulation results.

        This method creates a structured pandas DataFrame containing all simulation
        data including trajectories, performance metrics, and solver information.
        It processes weather data to appropriate units and provides a tabular
        format suitable for statistical analysis and visualization.

        The DataFrame includes:
        - Time series data (converted to days)
        - State trajectories with labeled columns (x_0, x_1, etc.)
        - Output trajectories with labeled columns (y_0, y_1, etc.)
        - Control input trajectories with labeled columns (u_0, u_1, etc.)
        - Disturbance trajectories with labeled columns (d_0, d_1, etc.)
        - Performance metrics (cost, economic rewards, penalties, net rewards)
        - Solver performance data (computation times, success rates)
        - Run identifier for multi-experiment analysis

        Args:
            run (int, optional): Run identifier for the simulation experiment. Defaults to 0.

        Returns:
            pd.DataFrame: Structured DataFrame with all simulation data organized
                in columns with descriptive names. Each row represents a time step
                with corresponding trajectory and performance data.
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
        Save experiment results to CSV file for analysis and archiving.

        This method exports the complete experiment results to a CSV file containing
        all simulation data including trajectories, performance metrics, and solver
        information. The CSV format enables easy data sharing, analysis, and
        visualization using standard tools.

        The saved CSV file includes:
        - Complete time series data with appropriate units
        - All state, output, control, and disturbance trajectories
        - Performance metrics (cost, rewards, penalties)
        - Solver performance data (computation times, success rates)
        - Run identifier for experiment tracking

        Args:
            save_path (str): Directory path where the CSV file will be saved.
                The filename is constructed using the experiment's save_name.

        Returns:
            None
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
    mpc_params["Ns"] = 10
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
