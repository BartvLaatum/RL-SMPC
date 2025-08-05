import os
import argparse
from time import time
from typing import Any, Dict, List, Tuple

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


class MPC:
    """
    A Model Predictive Controller (MPC) for nonlinear greenhouse systems using CasADi.

    This class implements a Model Predictive Controller for greenhouse
    climate control, using CasADi's Opti stack to formulate and solve nonlinear
    linear programs (NLP). The controller manages CO2 supply, ventilation, and heating
    to optimize crop growth while maintaining environmental constraints.

    The MPC formulation includes:
    - Economic objective function with crop growth rewards and control costs
    - Soft constraints via slack variables on CO2 concentration, temperature, and humidity
    - Hard constraints on control inputs and their rates of change
    - System dynamics constraints based on greenhouse physics

    Attributes:
        nx (int): Number of state variables (4: dry mass, CO2, temperature, vapor pressure)
        nu (int): Number of control inputs (3: CO2 supply, ventilation, heating)
        ny (int): Number of output variables (4: dry mass, CO2, temperature, humidity)
        nd (int): Number of disturbance variables (weather conditions)
        dt (float): Time step for discretization (seconds)
        nDays (int): Simulation duration in days
        Np (int): Prediction horizon length
        L (int): Total simulation time in seconds
        start_day (int): Starting day for weather data
        t (np.ndarray): Time vector for the simulation
        N (int): Total number of time steps
        x_initial (List[float]): Initial state values
        u_initial (List[float]): Initial control input values
        lb_pen_w (np.ndarray): Lower bounds for penalty weights
        ub_pen_w (np.ndarray): Upper bounds for penalty weights
        Ns (int): Number of scenarios for robust MPC
        uncertainty_value (float): Parameter uncertainty level
        constraints (Dict[str, Any]): System constraints dictionary
        nlp_opts (Dict[str, Any]): IPOPT solver options
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

        # initialize the boundaries for the optimization problem
        self.init_nmpc()

        self.F, self.g = define_model(self.dt, self.x_min, self.x_max)

    def init_nmpc(self):
        """
        Set the boundaries for the optimization problem.
        """
        # box constraints for the state variables
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

        # soft constraints for the output variables
        self.y_min = np.array(
            [
                self.constraints["W_min"]*1e3,
                self.constraints["co2_min"],
                self.constraints["temp_min"],
                self.constraints["rh_min"]
            ]
        )
        self.y_max = np.array(
            [
                self.constraints["W_max"]*1e3,
                self.constraints["co2_max"],
                self.constraints["temp_max"],
                self.constraints["rh_max"]
            ]
        )

        # control input constraints vector
        self.u_min = np.array([self.constraints["co2_supply_min"], self.constraints["vent_min"], self.constraints["heat_min"]])
        self.u_max = np.array([self.constraints["co2_supply_max"], self.constraints["vent_max"], self.constraints["heat_max"]])

        # lower and upper bound change of u 
        self.du_max = np.divide(self.u_max, [10, 10, 10])

    def solve_ocp(self, x0, u0, ds, u_guess=None, x_guess=None):
        """
        Sets solver values for the optimization problem.
        This function initializes various parameters and values required for the scenario-based
        Model Predictive Control (MPC) solver, including initial states, control inputs, and weather trajectory.

        Args:
            x0 (np.ndarray): Initial state
            u0 (np.ndarray): Initial control input
            ds (np.ndarray): Disturbances
            u_guess (np.ndarray): Initial guess for control inputs
            x_guess (np.ndarray): Initial guess for states
        
        Returns:
            xs_opt (np.ndarray): Optimal state trajectory
            ys_opt (np.ndarray): Optimal output trajectory
            us (np.ndarray): Optimal control input trajectory
            J (float): Cost value
            solver_time (float): Time taken to solve the problem
            exit_message (str): Exit message from the solver
        """

        self.opti.set_value(self.x0, x0)
        self.opti.set_value(self.init_u, u0)
        self.opti.set_value(self.ds, ds)

        if u_guess is not None:
            self.opti.set_initial(self.us, u_guess)
        if x_guess is not None:
            self.opti.set_initial(self.xs, x_guess)

        start_time = time()
        try:
            solution = self.opti.solve()
        except RuntimeError as err:
            # Recover from the failed solve: you might use the last iterate available via opti.debug
            print("Solver failed with error:", err)
            solution = self.opti.debug  # Returns the current iterate even if not converged
        solver_time = time() - start_time

        exit_message = solution.stats()['return_status']
        xs_opt = solution.value(self.xs)
        ys_opt = solution.value(self.ys)
        us = solution.value(self.us)
        J = solution.value(self.opti.f)

        return xs_opt, ys_opt, us, J, solver_time, exit_message


    def define_nlp(self, p: np.ndarray) -> None:
        """
        This method sets up the complete nonlinear Model Predictive Control (MPC) problem including
        the cost function, constraints, and bounds. It creates decision variables, defines system
        dynamics, input constraints, and soft constraints with penalties for the controlled
        variables.

        Args:
            p (np.ndarray): Array containing model parameters and settings

        The method sets up:
        - Decision variables (states, inputs, outputs, slack variables)
        - System dynamics
        - Input magnitude and rate constraints  
        - Soft constraints via slack variables for:
            - CO2 concentration (lower/upper bounds)
            - Temperature (lower/upper bounds)
            - Humidity (lower/upper bounds)
        - Economic objective function with penalties for violating the soft constraints 

        The optimization problem is configured to use IPOPT solver with specified options.
        """
        # Create an Opti instance
        self.opti = ca.Opti()

        # Number of penalties (CO2 lb, CO2 ub, Temp lb, Temp ub, humidity lb, humidity ub)
        num_penalties = 6

        # Decision Variables (Control inputs, slack variables, states, outputs)
        self.us = self.opti.variable(self.nu, self.Np)  # Control inputs (nu x Np)
        P = self.opti.variable(num_penalties, self.Np)
        self.xs = self.opti.variable(self.nx, self.Np+1)
        self.ys = self.opti.variable(self.ny, self.Np)

        # Parameters
        self.x0 = self.opti.parameter(self.nx, 1)  # Initial state
        self.ds = self.opti.parameter(self.nd, self.Np)  # Disturbances
        self.init_u = self.opti.parameter(self.nu, 1)  # Initial control input

        J = 0

        for ll in range(self.Np):
            self.opti.subject_to(self.xs[:, ll+1] == self.F(self.xs[:, ll], self.us[:, ll], self.ds[:, ll], p))
            self.opti.subject_to(self.ys[:, ll] == self.g(self.xs[:, ll+1]))
            # Input constraints
            self.opti.subject_to(self.u_min <= (self.us[:,ll] <= self.u_max))

            # Soft constraints via slack variables
            self.opti.subject_to(P[:, ll] >= 0)
            self.opti.subject_to(P[0, ll] >= self.lb_pen_w[0,0] * (self.y_min[1] - self.ys[1, ll]))
            self.opti.subject_to(P[1, ll] >= self.ub_pen_w[0,0] * (self.ys[1, ll] - self.y_max[1]))
            self.opti.subject_to(P[2, ll] >= self.lb_pen_w[0,1] * (self.y_min[2] - self.ys[2, ll]))
            self.opti.subject_to(P[3, ll] >= self.ub_pen_w[0,1] * (self.ys[2, ll] - self.y_max[2]))
            self.opti.subject_to(P[4, ll] >= self.lb_pen_w[0,2] * (self.y_min[3] - self.ys[3, ll]))
            self.opti.subject_to(P[5, ll] >= self.ub_pen_w[0,2] * (self.ys[3, ll] - (self.y_max[3] - 2.0)))

            # Cost function with penalties
            delta_dw = self.xs[0, ll+1] - self.xs[0, ll]
            J -= compute_economic_reward(delta_dw, p, self.dt, self.us[:,ll])
            J += (P[0, ll]+ P[1, ll]+P[2, ll]+P[3, ll]+P[4, ll]+P[5, ll])

            if ll < self.Np-1:
                # Change in input constraints
                self.opti.subject_to(-self.du_max<=(self.us[:,ll+1] - self.us[:,ll]<=self.du_max))

        # Constraints on initial state and input 
        self.opti.subject_to(-self.du_max <= (self.us[:,0] - self.init_u <= self.du_max))
        self.opti.subject_to(self.xs[:,0] == self.x0)
        self.opti.minimize(J)

        self.opti.solver('ipopt', self.nlp_opts)

    def first_initial_guess(self, d, p):
        """
        Compute the first initial guess for the optimization problem.
        This function sets the initial guess for the control inputs and states in the optimization problem.
        It is used to provide a starting point for the optimization solver.
        
        Args:
            d (np.ndarray): Disturbances
            p (np.ndarray): Model parameters

        Returns:
            u_initial_guess (np.ndarray): Initial guess for control inputs
            x_initial_guess (np.ndarray): Initial guess for states
        """
        u_initial_guess = np.ones((self.nu, self.Np)) * np.array(self.u_initial).reshape(self.nu, 1)
        x_initial_guess = np.zeros((self.nx, self.Np+1))
        x_initial_guess[:, 0] = self.x_initial

        for i in range(self.Np):
            x_initial_guess[:, i+1] = self.F(x_initial_guess[:, i], u_initial_guess[:, i], d[:, i], p).toarray().ravel()

        return u_initial_guess, x_initial_guess


    def constraint_violation(self, y: np.ndarray):
        """
        Function that computes the absolute penalties for violating system constraints.
        System constraints are currently non-dynamical, and based on observation bounds of gym environment.
        We do not look at dry mass bounds, since those are non-existent in real greenhouse.

        Args:
            y (np.ndarray): Output

        Returns:
            lowerbound (np.ndarray): Lower bound violation
            upperbound (np.ndarray): Upper bound violation
        """
        lowerbound = self.y_min[1:] - y[1:]
        lowerbound[lowerbound < 0] = 0
        upperbound = y[1:] - self.y_max[1:]
        upperbound[upperbound < 0] = 0

        return lowerbound, upperbound

    def compute_penalties(self, y):
        """
        Compute the penalties for violating the system constraints.

        Args:
            y (np.ndarray): Output

        Returns:
            penalties (float): Penalties for violating the system constraints
        """
        lowerbound, upperbound = self.constraint_violation(y)
        penalties = np.dot(self.lb_pen_w, lowerbound) + np.dot(self.ub_pen_w, upperbound)
        return np.sum(penalties)


class Experiment:
    """
    Experiment manager for closed-loop MPC simulation and performance evaluation.

    This class manages the complete simulation of a Model Predictive Controller
    in closed-loop operation, including disturbance handling, parametric uncertainty,
    and comprehensive result tracking. It performs the receding horizon optimization
    over the entire simulation period while collecting performance metrics.

    The experiment workflow includes:
    - Initialization with MPC controller and simulation parameters
    - Closed-loop simulation with receding horizon optimization
    - Parametric uncertainty injection for robustness testing
    - Comprehensive result collection and analysis
    - Data export in multiple formats (numpy arrays and pandas DataFrames)

    Attributes:
        project_name (str): Name of the project for result organization
        save_name (str): Filename for saving experiment results
        mpc (MPC): The MPC controller instance
        uncertainty_value (float): Level of parametric uncertainty for robustness testing
        p (np.ndarray): Model parameters
        rng (np.random.Generator): Random number generator for uncertainty
        x (np.ndarray): State trajectory over simulation period (nx, N+1)
        y (np.ndarray): Output trajectory over simulation period (ny, N+1)
        u (np.ndarray): Control input trajectory (nu, N+1)
        d (np.ndarray): Disturbance trajectory (weather data)
        uopt (np.ndarray): Optimal control sequences from MPC (nu, Np, N+1)
        J (np.ndarray): Cost values at each time step (1, N)
        dJdu (np.ndarray): Cost gradients (nu, Np, N)
        output (list): Optimization solver outputs
        penalties (np.ndarray): Constraint violation penalties (1, N)
        econ_rewards (np.ndarray): Economic rewards (1, N)
        rewards (np.ndarray): Net rewards (economic - penalties) (1, N)
        solver_times (np.ndarray): Solver computation times (1, N)
        exit_message (np.ndarray): Solver exit status codes (1, N)
    """
    def __init__(
        self,
        mpc: MPC,
        save_name: str,
        project_name: str,
        weather_filename: str,
        uncertainty_value: float,
        p: np.ndarray,
        rng: np.random.Generator,
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

        # Initialize the results arrays
        self.uopt = np.zeros((mpc.nu, mpc.Np, mpc.N+1))
        self.J = np.zeros((1, mpc.N))
        self.dJdu = np.zeros((mpc.nu*mpc.Np, mpc.N))
        self.output = []
        self.penalties = np.zeros((1, mpc.N))
        self.econ_rewards = np.zeros((1, mpc.N))
        self.rewards = np.zeros((1, mpc.N))
        self.solver_times = np.zeros((1, mpc.N))
        self.exit_message = np.zeros((1, mpc.N))

    def initial_guess_xs(self, x0, u_guess, ds) -> np.ndarray:
        """
        Generate initial guesses for the states based on the provided parametric samples.

        Args:
            x0 (np.ndarray): The initial state of the system.
            u_guess (np.ndarray): The initial guess for the control inputs.
            ds (np.ndarray): The disturbance trajectory.

        Returns:
            Array: An array with initial guess for the state trajectory given input trajectory u.
        """
        xs = np.zeros((self.mpc.nx, self.mpc.Np + 1))
        xs[:, 0] = x0
        for ll in range(self.mpc.Np):
            xs[:, ll + 1] = self.mpc.F(
                xs[:, ll],
                u_guess[:, ll],
                ds[:, ll],
                self.p
            ).toarray().ravel()
        return xs


    def solve_nmpc(self) -> None:
        """
        Execute the complete closed-loop MPC simulation.

        This method performs the receding horizon optimization over the entire
        simulation period. At each time step, it:
        1. Generates initial guesses for states and controls
        2. Solves the MPC optimization problem
        3. Applies the first control input to the system
        4. Simulates the system with parametric uncertainty
        5. Updates all performance metrics and results

        The simulation implements a standard receding horizon approach where
        the optimization window moves forward one step at a time, and the
        optimal control sequence is updated based on the current state.
        """
        # Set the initial guess for input and state
        u_initial_guess = np.ones((self.mpc.nu, self.mpc.Np)) * np.array(self.mpc.u_initial).reshape(self.mpc.nu, 1)

        # Solve the MPC optimization problem for each time step
        for ll in tqdm(range(self.mpc.N)):
            # Generate the initial guess for the states
            x_initial_guess = self.initial_guess_xs(
                self.x[:, ll],
                u_initial_guess,
                self.d[:, ll:ll+self.mpc.Np]
            )

            # Solve the MPC optimization problem
            xs_opt, ys_opt, us_opt, J_opt, solver_time, exit_message = self.mpc.solve_ocp(
                self.x[:, ll],
                self.u[:, ll],
                self.d[:, ll:ll+self.mpc.Np],
                u_guess=u_initial_guess,
                x_guess=x_initial_guess
            )

            # Update the control input
            self.u[:, ll+1] = us_opt[:, 0]

            # Inject parametric uncertainty
            params = parametric_uncertainty(self.p, self.uncertainty_value, self.rng)

            # Simulate the system with parametric uncertainty
            self.x[:, ll+1] = self.mpc.F(self.x[:, ll], self.u[:, ll+1], self.d[:, ll], params).toarray().ravel()
            self.y[:, ll+1] = self.mpc.g(self.x[:, ll+1]).toarray().ravel()

            # Compute the economic reward and penalties
            delta_dw = self.x[0, ll+1] - self.x[0, ll]
            econ_rew = compute_economic_reward(delta_dw, get_parameters(), self.mpc.dt, self.u[:, ll+1])
            penalties = self.mpc.compute_penalties(self.y[:, ll+1])

            # Update the results
            self.update_results(us_opt, J_opt, [], econ_rew, penalties, ll, solver_time, exit_message=exit_message)

            # Update the initial guess for the next time step
            u_initial_guess = np.concatenate([us_opt[:, 1:], us_opt[:, -1][:, None]], axis=1)

    def update_results(self, us_opt, Js_opt, sol, eco_rew, penalties, step, solver_time, exit_message=None):
        """
        Update experiment results with current optimization outcomes.

        This method stores the results from the current MPC optimization step,
        including optimal controls, cost values, economic rewards, penalties,
        solver performance metrics, and exit status.

        Args:
            us_opt (np.ndarray): Optimal control sequence from MPC
            Js_opt (float): Optimal cost value
            sol (list): Solver output information (currently unused)
            eco_rew (float): Economic reward for current step
            penalties (float): Constraint violation penalties
            step (int): Current simulation time step
            solver_time (float): Time taken by the optimization solver
            exit_message (str, optional): Solver exit status message

        Notes:
            - Converts solver exit messages to binary success/failure codes
            - Stores all trajectories and metrics for post-processing
            - Calculates net rewards as economic rewards minus penalties
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
        Export results as numpy array with all trajectories and performance metrics.

        This method organizes all simulation data into a structured numpy array
        with time series data for states, outputs, inputs, disturbances, and
        performance metrics. The data is formatted for easy analysis and plotting.

        Args:
            run (int): Run identifier for the simulation experiment

        Returns:
            np.ndarray: 2D array with shape (N, n_columns) containing:
                - Time vector (days)
                - State trajectories (nx columns)
                - Output trajectories (ny columns) 
                - Input trajectories (nu columns)
                - Disturbance trajectories (nd columns)
                - Performance metrics (cost, rewards, penalties, solver times)
                - Run identifier
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
        Export results as pandas DataFrame with labeled columns.

        This method creates a structured pandas DataFrame with all simulation
        data organized in columns with descriptive names. This format is ideal
        for data analysis, plotting, and statistical evaluation.

        Args:
            run (int, optional): Run identifier for the simulation experiment. Defaults to 0.

        Returns:
            pd.DataFrame: DataFrame with columns for:
                - time: Simulation time in days
                - x_0, x_1, ...: State variables
                - y_0, y_1, ...: Output variables  
                - u_0, u_1, ...: Control inputs
                - d_0, d_1, ...: Disturbance variables
                - J: Cost values
                - econ_rewards: Economic rewards
                - penalties: Constraint violation penalties
                - rewards: Net rewards (economic - penalties)
                - solver_times: Optimization solver computation times
                - solver_success: Binary solver success indicators
                - run: Run identifier
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
        Save experiment results to CSV file.

        This method exports all simulation results to a CSV file for
        permanent storage and sharing. The CSV format allows easy
        import into other analysis tools and programming languages.

        Args:
            save_path (str): Directory path where the CSV file will be saved.
                           The filename is determined by self.save_name.
        """
        df = self.retrieve_results()
        df.to_csv(f"{save_path}/{self.save_name}", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str)
    parser.add_argument("--env_id", type=str, default="LettuceGreenhouse")
    parser.add_argument("--save_name", type=str)
    parser.add_argument("--weather_filename", default="outdoorWeatherWurGlas2014.csv", type=str)
    args = parser.parse_args()

    save_path = f"data/{args.project}/mpc"
    os.makedirs(save_path, exist_ok=True)

    env_params = load_env_params(args.env_id)
    mpc_params = load_mpc_params(args.env_id)
    env_params["n_days"] = 10
    mpc_params["Np"] = 12

    # p = DefineParameters()
    p = get_parameters()
    mpc = MPC(**env_params, **mpc_params)
    mpc.define_nlp(p)
    uncertainty_value = 0.1
    rng = np.random.default_rng(666)
    exp = Experiment(mpc, args.save_name, args.project, args.weather_filename, uncertainty_value, p, rng)
    exp.solve_nmpc()
