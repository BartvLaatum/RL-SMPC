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

        self.F, self.g = define_model(self.dt)

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

        self.y_min = np.array([self.constraints["W_min"], self.constraints["co2_min"], self.constraints["temp_min"], self.constraints["rh_min"]])
        self.y_max = np.array([self.constraints["W_max"], self.constraints["co2_max"], self.constraints["temp_max"], self.constraints["rh_max"]])

        # control input constraints vector
        self.u_min = np.array([self.constraints["co2_supply_min"], self.constraints["vent_min"], self.constraints["heat_min"]])
        self.u_max = np.array([self.constraints["co2_supply_max"], self.constraints["vent_max"], self.constraints["heat_max"]])
        # lower and upper bound change of u 
        self.du_max = np.divide(self.u_max, [10, 10, 10])

        # initial values of the decision variables (control signals) in the optimization
        self.u0 = np.zeros((self.nu, self.Np)) # this can be moved to the shooting function method.

    def define_nlp(self, p: Dict[str, Any]) -> None:
        """
        Define the optimization problem for the nonlinear SMPC using CasADi's Opti stack.
        Including the cost function, constraints, and bounds.
        """
        # Create an Opti instance
        self.opti = ca.Opti()

        # Number of penalties (CO2 lb, CO2 ub, Temp lb, Temp ub, humidity lb, humidity ub)
        num_penalties = 6

        # Decision Variables (Control inputs, slack variables, states, outputs)
        self.us = self.opti.variable(self.nu, self.Np)  # Control inputs (nu x Np)
        # self.P = self.opti.variable(num_penalties, self.Np)
        # self.xs = self.opti.variable(self.nx, self.Np+1)
        # self.ys = self.opti.variable(self.ny, self.Np)
        self.xs_list = [self.opti.variable(self.nx, self.Np+1) for _ in range(self.Ns)]
        self.ys_list = [self.opti.variable(self.ny, self.Np) for _ in range(self.Ns)]
        self.Ps     = [self.opti.variable(num_penalties, self.Np) for _ in range(self.Ns)]


        # Parameters
        self.x0 = self.opti.parameter(self.nx, 1)  # Initial state
        self.ds = self.opti.parameter(self.nd, self.Np)  # Disturbances
        self.init_u = self.opti.parameter(self.nu, 1)  # Initial control input

        self.opti.set_value(self.x0, self.x_initial)
        self.opti.set_value(self.init_u, self.u_initial)

        self.opti.set_value(self.ds, ca.DM.zeros(self.ds.shape))

        self.Js = 0
        for i in range(self.Ns):
            xs = self.xs_list[i]
            ys = self.ys_list[i]
            P = self.Ps[i]
            
            self.opti.subject_to(xs[:,0] == self.x0)     # Initial Condition Constraint

            for ll in range(self.Np):
                params = parametric_uncertainty(p, self.uncertainty_value, self.rng)
                self.opti.subject_to(xs[:, ll+1] == self.F(xs[:, ll], self.us[:, ll], self.ds[:, ll], params))
                self.opti.subject_to(ys[:, ll] == self.g(xs[:, ll+1]))
                self.opti.subject_to(self.u_min <= (self.us[:,ll] <= self.u_max))                     # Input   Contraints

                self.opti.subject_to(P[:, ll] >= 0)
                self.opti.subject_to(P[0, ll] >= self.lb_pen_w[0,0] * (self.y_min[1] - ys[1, ll]))
                self.opti.subject_to(P[1, ll] >= self.ub_pen_w[0,0] * (ys[1, ll] - self.y_max[1]))
                self.opti.subject_to(P[2, ll] >= self.lb_pen_w[0,1] * (self.y_min[2] - ys[2, ll]))
                self.opti.subject_to(P[3, ll] >= self.ub_pen_w[0,1] * (ys[2, ll] - self.y_max[2]))
                self.opti.subject_to(P[4, ll] >= self.lb_pen_w[0,2] * (self.y_min[3] - ys[3, ll]))
                self.opti.subject_to(P[5, ll] >= self.ub_pen_w[0,2] * (ys[3, ll] - self.y_max[3]))

                # COST FUNCTION WITH PENALTIES
                delta_dw = xs[0, ll+1] - xs[0, ll]
                self.Js -= compute_economic_reward(delta_dw, p, self.dt, self.us[:,ll])
                self.Js += (P[0, ll]+ P[1, ll]+P[2, ll]+P[3, ll]+P[4, ll]+P[5, ll])

                if ll < self.Np-1:
                    self.opti.subject_to(-self.du_max<=(self.us[:,ll+1] - self.us[:,ll]<=self.du_max))     # Change in input Constraint

        self.Js = self.Js / self.Ns

        # Intial constraint on input
        self.opti.subject_to(-self.du_max <= (self.us[:,0] - self.init_u <= self.du_max))   # Initial change in input Constraint
        self.opti.minimize(self.Js)

        self.opti.solver('ipopt', self.nlp_opts)
        self.SMPC_func = self.opti.to_function(
            'SMPC_func',
            [self.x0, self.ds, self.init_u],
            [self.us, ca.vertcat(*self.xs_list), ca.vertcat(*self.ys_list), self.Js],
            ['x0','ds','u0'],
            ['u_opt', 'x_opt_all', 'y_opt_all', 'J_opt']
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
        uncertainty_scale: float,
        rng,
    ) -> None:

        self.project_name = project_name
        self.save_name = save_name
        self.mpc = mpc
        self.uncertainty_scale = uncertainty_scale
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

        self.uopt = np.zeros((mpc.nu, mpc.Np, mpc.N+1))
        self.J = np.zeros((1, mpc.N))
        self.dJdu = np.zeros((mpc.nu*mpc.Np, mpc.N))
        self.output = []
        self.penalties = np.zeros((1, mpc.N))
        self.econ_rewards = np.zeros((1, mpc.N))
        self.rewards = np.zeros((1, mpc.N))

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
        for ll in tqdm(range(self.mpc.N)):

            us_opt, xs_opt, ys_opt, J_opt = self.mpc.SMPC_func(self.x[:, ll], self.d[:, ll:ll+self.mpc.Np], self.u[:, ll])
            self.u[:, ll+1] = us_opt[:, 0].toarray().ravel()

            params = parametric_uncertainty(p, self.uncertainty_scale, self.rng)

            self.x[:, ll+1] = self.mpc.F(self.x[:, ll], self.u[:, ll+1], self.d[:, ll], params).toarray().ravel()
            self.y[:, ll+1] = self.mpc.g(self.x[:, ll+1]).toarray().ravel()

            delta_dw = self.x[0, ll+1] - self.x[0, ll]
            econ_rew = compute_economic_reward(delta_dw, get_parameters(), self.mpc.dt, self.u[:, ll+1])
            penalties = self.mpc.compute_penalties(self.y[:, ll+1])

            self.update_results(us_opt, J_opt, [], econ_rew, penalties, ll)

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
        # for i in range(self.uopt.shape[0]):
        #     for j in range(self.uopt.shape[1]):
        #         data[f"uopt_{i}_{j}"] = self.uopt[i, j, :-1]
        data["J"] = self.J.flatten()
        data["econ_rewards"] = self.rewards.flatten()
        data["penalties"] = self.penalties.flatten()
        data["rewards"] = self.rewards.flatten()

        df = pd.DataFrame(data, columns=data.keys())
        df.to_csv(f"{save_path}/{self.save_name}.csv", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str)
    parser.add_argument("--env_id", type=str, default="LettuceGreenhouse")
    parser.add_argument("--save_name", type=str)
    parser.add_argument("--weather_filename", default="outdoorWeatherWurGlas2014.csv", type=str)
    args = parser.parse_args()

    save_path = f"data/{args.project}/stochastic/smpc"
    os.makedirs(save_path, exist_ok=True)
    uncertainty_scale = 0.05

    env_params = load_env_params(args.env_id)
    mpc_params = load_mpc_params(args.env_id)
    mpc_params["uncertainty_value"] = uncertainty_scale

    H = [1, 2, 3, 4, 5, 6]
    mpc_params["Ns"] = 5

    for h in H:
        save_name = f"{args.save_name}-{h}H-{mpc_params['Ns']}Ns-{uncertainty_scale}"
        mpc_params["rng"] = np.random.default_rng(42)
        mpc_params["Np"] = int(h * 3600 / env_params["dt"])

        # p = DefineParameters()
        p = get_parameters()
        mpc = SMPC(**env_params, **mpc_params)
        mpc.define_nlp(p)

        rng = np.random.default_rng(666)
        exp = Experiment(mpc, save_name, args.project, args.weather_filename, uncertainty_scale, rng)
        exp.solve_nmpc(p)
        exp.save_results(save_path)
