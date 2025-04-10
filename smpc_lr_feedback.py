import os
import argparse
from typing import Any, Dict, List
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
    co2ppm2dens,
    rh2vaporDens,
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

        y_target_day = np.array([0.0035, 700, 15, 70])
        y_target_night = np.array([0.0035, 500, 10, 78])

        self.x_target_day = y_target_day.copy()
        self.x_target_night = y_target_night.copy()

        self.x_target_day[1] = co2ppm2dens(self.x_target_day[2], self.x_target_day[1])
        self.x_target_day[3] = rh2vaporDens(self.x_target_day[2], self.x_target_day[3])

        self.x_target_night[1] = co2ppm2dens(self.x_target_night[2], self.x_target_night[1])
        self.x_target_night[3] = rh2vaporDens(self.x_target_night[2], self.x_target_night[3])


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

    def compute_LQR_gain(self, A, B, Q, R):
        """
        Compute the LQR gain for a discrete-time system:
            x[k+1] = A x[k] + B u[k]
        using an iterative solution to the Riccati equation.
        """
        P = Q.copy()
        max_iter = 1000
        eps = 1e-6
        for _ in range(max_iter):
            P_next = A.T @ P @ A - A.T @ P @ B @ np.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A + Q
            if np.linalg.norm(P_next - P) < eps:
                break
            P = P_next
        K = -np.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A
        return K

    def compute_climate_gain(self, x_ss, u_ss, d_nom, p_nom, Q_climate, R_climate):
        """
        Compute a full gain matrix that only uses the climate states.
        
        Parameters:
        F         : system dynamics function F(x,u,w,p)
        x_ss      : steady-state operating point (4x1 vector)
        u_ss      : steady-state control input (3x1 vector)
        d_nom     : nominal weather parameters (column vector)
        p_nom     : nominal uncertainty parameters (column vector)
        Q_climate : weighting matrix for climate states (3x3)
        R_climate : weighting matrix for control (3x3)
        
        Returns:
        K_full    : Full gain matrix of size 3x4, with first column zero.
        """
        nx = x_ss.shape[0]
        nu = u_ss.shape[0]

        # Define symbolic variables for linearization
        x_sym = ca.MX.sym('x', nx)
        u_sym = ca.MX.sym('u', nu)
        d_sym = ca.MX.sym('d', d_nom.shape[0])
        p_sym = ca.MX.sym('p', p_nom.shape[0])
        
        f_sym = self.F(x_sym, u_sym, d_sym, p_sym)
        
        # Compute Jacobians
        A_sym = ca.jacobian(f_sym, x_sym)
        B_sym = ca.jacobian(f_sym, u_sym)
        A_fun = ca.Function('A_fun', [x_sym, u_sym, d_sym, p_sym], [A_sym])
        B_fun = ca.Function('B_fun', [x_sym, u_sym, d_sym, p_sym], [B_sym])
        
        # Evaluate the Jacobians at the steady state and nominal conditions
        A_full = np.array(A_fun(x_ss, u_ss, d_nom, p_nom))
        B_full = np.array(B_fun(x_ss, u_ss, d_nom, p_nom))
        
        # Extract the climate subsystem dynamics.
        # Assuming state ordering: [crop biomass; climate states]
        A_climate = A_full[1:, 1:]  # Last 3 rows and columns (3x3)
        B_climate = B_full[1:, :]   # Last 3 rows (3x3 if B_full is 4x3)
        
        # Compute the LQR gain for the climate subsystem
        K_climate = self.compute_LQR_gain(A_climate, B_climate, Q_climate, R_climate)
        
        # Embed the climate gain into a full gain matrix:
        # The full gain is 3x4. The first column (for crop biomass) is zeros.
        K_full = np.hstack([np.zeros((nu, 1)), K_climate])
        return K_full

    def time_varying_K(self, D, p):
        K_list = []
        for i in range(self.Np):
            Q_climate, R_climate = np.eye(3), np.eye(3)

            d = D[:, i]
            if d[0] < 10:
                xss = self.x_target_night
            else:
                xss = self.x_target_day

            uss = self.compute_steady_input(xss, d, p)
            K_full = self.compute_climate_gain(xss, uss, d, p, Q_climate, R_climate)
            K_list.append(K_full)

        return K_list


    def compute_steady_input(self, x_target, d, p_nom):
        """
        Compute the steady control input u_ss such that x_ss = x_target for nominal weather and uncertainty.
        Here we solve:

            0 = g(f(x_target, u_ss, d, p_nom)) - y_target
        
        for u_ss.

        Parameters:
        F       : the system dynamics function F(x, u, d, p)
        y_target: desired steady-state (target) state (nx x 1)
        d   : weather disturbance (nd x 1)
        p_nom   : nominal uncertainty parameters (np x 1)

        Returns:
        u_ss    : computed steady-state control input (nu x 1)
        """
        # Create an Opti instance for steady-state computation
        opti_ss = ca.Opti()
        u_ss = opti_ss.variable(self.nu, 1)
        opti_ss.subject_to(self.u_min <= (u_ss <= self.u_max))  # Input constraints

        # Steady state constraint: f(x_target, u_ss, d, p_nom) = x_target
        x_next = self.F(x_target, u_ss, d, p_nom)
        # Objective can be to minimize the norm of the difference (or just set it to zero)
        opti_ss.minimize(ca.sumsqr(x_next[1:] - x_target[1:]))

        # Set up and solve the steady-state problem
        opts = {"print_time": False, "ipopt.print_level": 0}
        opti_ss.solver("ipopt", opts)
        sol = opti_ss.solve()
        return sol.value(u_ss)

    def define_nlp(self, p: Dict[str, Any]) -> None:
        """
        Define the optimization problem for the nonlinear SMPC using CasADi's Opti stack.
        Including the cost function, constraints, and bounds.
        """
        # Create an Opti instance
        self.opti = ca.Opti()

        # Number of penalties (CO2 lb, CO2 ub, Temp lb, Temp ub, humidity lb, humidity ub)
        num_penalties = 6

        # Decision Variables (Theta, slack variables, states, outputs)
        theta = self.opti.variable(self.nu, self.Np)  # Theta (nu x Np)
        xs_list = [self.opti.variable(self.nx, self.Np+1) for _ in range(self.Ns)]
        ys_list = [self.opti.variable(self.ny, self.Np) for _ in range(self.Ns)]
        Ps = [self.opti.variable(num_penalties, self.Np) for _ in range(self.Ns)]

        # Parameters
        p_samples = [self.opti.parameter(p.shape[0], self.Np) for _ in range(self.Ns)]
        K_list = [self.opti.parameter(self.nu, self.nx) for _ in range(self.Np)]
        x0 = self.opti.parameter(self.nx, 1)  # Initial state
        ds = self.opti.parameter(self.nd, self.Np)  # Disturbances
        init_u = self.opti.parameter(self.nu, 1)  # Initial control input

        # self.opti.set_value(x0, self.x_initial)
        # self.opti.set_value(init_u, self.u_initial)
        # self.opti.set_value(ds, ca.DM.zeros(ds.shape))

        Js = 0
        for i in range(self.Ns):
            xs = xs_list[i]
            ys = ys_list[i]
            ps = p_samples[i]
            P = Ps[i]

            self.opti.subject_to(xs[:,0] == x0)     # Initial Condition Constraint

            for ll in range(self.Np):
                pk = ps[:, ll]

                uk = K_list[ll] @ xs[:, ll] + theta[:, ll]

                self.opti.subject_to(xs[:, ll+1] == self.F(xs[:, ll], uk, ds[:, ll], pk))
                self.opti.subject_to(ys[:, ll] == self.g(xs[:, ll+1]))
                self.opti.subject_to(self.u_min <= (uk <= self.u_max))

                self.opti.subject_to(P[:, ll] >= 0)
                self.opti.subject_to(P[0, ll] >= self.lb_pen_w[0,0] * (self.y_min[1] - ys[1, ll]))
                self.opti.subject_to(P[1, ll] >= self.ub_pen_w[0,0] * (ys[1, ll] - self.y_max[1]))
                self.opti.subject_to(P[2, ll] >= self.lb_pen_w[0,1] * (self.y_min[2] - ys[2, ll]))
                self.opti.subject_to(P[3, ll] >= self.ub_pen_w[0,1] * (ys[2, ll] - self.y_max[2]))
                self.opti.subject_to(P[4, ll] >= self.lb_pen_w[0,2] * (self.y_min[3] - ys[3, ll]))
                self.opti.subject_to(P[5, ll] >= self.ub_pen_w[0,2] * (ys[3, ll] - (self.y_max[3]-2.0)))

                # COST FUNCTION WITH PENALTIES
                delta_dw = xs[0, ll+1] - xs[0, ll]
                Js -= compute_economic_reward(delta_dw, p, self.dt, uk)
                Js += (P[0, ll]+ P[1, ll]+P[2, ll]+P[3, ll]+P[4, ll]+P[5, ll])

                if ll < self.Np-1:
                    uk_1 = K_list[ll+1] @ xs[:, ll+1] + theta[:, ll+1]
                    self.opti.subject_to(-self.du_max<=(uk_1 - uk<=self.du_max))     # Change in input Constraint

        Js = Js / self.Ns

        u0 = K_list[0] @ x0 + theta[:, 0] 
        self.opti.subject_to(-self.du_max <= (u0 - init_u <= self.du_max))  
        self.opti.minimize(Js)

        self.opti.solver('ipopt', self.nlp_opts)
        self.SMPC_func = self.opti.to_function(
            'SMPC_func',
            [x0, ds, init_u, *K_list, *p_samples],
            [theta, ca.vertcat(*xs_list), ca.vertcat(*ys_list), Js],
            ['x0','ds','u0'] + \
            [f"K_list_{i}" for i in range(self.Np)] + \
            [f"p_sample_{i}" for i in range(self.Ns)], 
            ['theta_opt', 'x_opt_all', 'y_opt_all', 'J_opt'],
        )

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

    def solve_nmpc(self) -> None:
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
        for ll in tqdm(range(self.mpc.N)):
            p_samples = self.generate_psamples()

            K_list = self.mpc.time_varying_K(self.d[:, ll:ll+self.mpc.Np], self.p)

            # we have to transpose p_samples since MPC_func expects matrix of shape (n_params, Np)
            p_sample_list = [p_samples[i].T for i in range(self.mpc.Ns)]
            theta_opt, xs_opt, ys_opt, J_opt = self.mpc.SMPC_func(
                self.x[:, ll],
                self.d[:, ll:ll+self.mpc.Np], 
                self.u[:, ll],
                *K_list,
                *p_sample_list
            )

            # compute the optimal control input, predicted by SMPC
            self.u[:, ll+1] = K_list[0] @ self.x[:, ll] + theta_opt[:, 0].toarray().ravel()

            params = parametric_uncertainty(self.p, self.uncertainty_value, self.rng)

            self.x[:, ll+1] = self.mpc.F(self.x[:, ll], self.u[:, ll+1], self.d[:, ll], params).toarray().ravel()
            self.y[:, ll+1] = self.mpc.g(self.x[:, ll+1]).toarray().ravel()

            delta_dw = self.x[0, ll+1] - self.x[0, ll]
            econ_rew = compute_economic_reward(delta_dw, get_parameters(), self.mpc.dt, self.u[:, ll+1])
            penalties = self.mpc.compute_penalties(self.y[:, ll+1])

            self.update_results(theta_opt, J_opt, [], econ_rew, penalties, ll)

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


    def update_results(self, us_opt, Js_opt, sol, eco_rew, penalties, step):
        """
        Args:
            uopt (_type_): _description_
            J (_type_): _description_
            output (_type_): _description_
            step (_type_): _description_
        """
        self.uopt[:,:, step] = us_opt
        self.J[:, step] = Js_opt
        self.output.append(sol)
        self.econ_rewards[:, step] = eco_rew
        self.penalties[:, step] = penalties
        self.rewards[:, step] = eco_rew - penalties

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

    def retrieve_results(self, run=0):
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

        df = pd.DataFrame(data, columns=data.keys())
        df['run'] = run
        return df

    def save_results(self, df, save_path, run=0):
        """
        """
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
    # env_params["n_days"] = 1
    for h in H:
        mpc_rng = np.random.default_rng(42)
        exp_rng = np.random.default_rng(666)
        save_name = f"{args.save_name}-{h}H-{mpc_params['Ns']}Ns-{args.uncertainty_value}"
        mpc_params["rng"] = mpc_rng
        mpc_params["Np"] = int(h * 3600 / env_params["dt"])

        # p = DefineParameters()
        p = get_parameters()
        mpc = SMPC(**env_params, **mpc_params)
        mpc.define_nlp(p)

        exp = Experiment(mpc, save_name, args.project, args.weather_filename, args.uncertainty_value, p, exp_rng)
        exp.solve_nmpc()
        df = exp.retrieve_results()
        exp.save_results(df, save_path)
