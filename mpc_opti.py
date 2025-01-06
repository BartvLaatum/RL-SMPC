import argparse
from typing import Any, Dict, List

import yaml
import numpy as np
import pandas as pd
import casadi as ca

from common.utils import *

class MPC:
    def __init__(
            self,
            nx: int,
            nu: int,
            ny: int,
            nd: int,
            h: float,
            n_days: int,
            x0: List[float],
            Np: int,
            Ns: int,
            start_day: int,
            sigmad: float,
            lb_pen_w: List[float],
            ub_pen_w: List[float],
            constraints: Dict[str, Any],
            nlp_opts: Dict[str, Any],
            ) -> None:
        self.x_initial = x0
        self.nx = nx
        self.nu = nu
        self.ny = ny
        self.nd = nd
        self.h = h
        self.nDays = n_days
        self.Np = Np
        self.L = n_days * 86400
        self.start_day = start_day

        self.t = np.arange(0, self.L + self.h, self.h)
        self.N = len(self.t)
        self.lb_pen_w = np.expand_dims(lb_pen_w, axis=0)
        self.ub_pen_w = np.expand_dims(ub_pen_w, axis=0)
        self.Ns = Ns
        self.sigmad = sigmad
        self.lbg = []
        self.ubg = []
        self.constraints = constraints
        self.nlp_opts = nlp_opts

        # initialize the boundaries for the optimization problem
        self.init_nmpc()

        self.F, self.g = define_model(self.h)

    def init_nmpc(self):
        """Initialize the nonlinear MPC problem.
        Initialise the constraints, bounds, and the optimization problem.
        """
        # ah_max = rh2vaporDens(self.constraints["temp_max"], self.constraints["rh_max"])    # upper bound on vapor density                  [kg/m^{-3}]      
        ah_min = rh2vaporDens(self.constraints["temp_min"], self.constraints["rh_min"])    # lower bound on vapor density                  [kg/m^{-3}]
        co2_dens_min = co2ppm2dens(self.constraints["co2_min"], self.constraints["temp_min"]) # lower bound on co2 density                   [kg/m^{-3}]           
        co2_dens_max = co2ppm2dens(self.constraints["co2_max"], self.constraints["temp_max"]) # upper bound on co2 density                   [kg/m^{-3}]
        self.x_min = np.array([self.constraints["W_min"], 0, 5., ah_min])
        self.x_max = np.array([self.constraints["W_max"], 0.004, 40., 0.051])

        self.y_min = np.array([self.constraints["W_min"], self.constraints["co2_min"], self.constraints["temp_min"], self.constraints["rh_min"]])
        self.y_max = np.array([self.constraints["W_max"], self.constraints["co2_max"], self.constraints["temp_max"], self.constraints["rh_max"]])

        # control input constraints vector
        self.u_min = np.array([self.constraints["co2_supply_min"], self.constraints["vent_min"], self.constraints["heat_min"]])
        self.u_max = np.array([self.constraints["co2_supply_max"], self.constraints["vent_max"], self.constraints["heat_max"]])
        # lower and upper bound change of u 
        self.du_max = np.divide(self.u_max, [10, 10, 10])

        # initial values of the decision variables (control signals) in the optimization
        self.u0 = np.zeros((self.nu, self.Np)) # this can be moved to the shooting function method.

    def set_slack_variables(self, ll: int, p: Dict[str, Any]) -> None:
        # Penalty Constraints
        self.opti.subject_to(self.P[:, ll] >= 0)
        # CO2 Lower Bound Penalty
        self.opti.subject_to(self.P[0, ll] >= self.lb_pen_w[0,0] * (self.y_min[1] - self.ys[1, ll]))
        # CO2 Upper Bound Penalty
        self.opti.subject_to(self.P[1, ll] >= self.ub_pen_w[0,0] * (self.ys[1, ll] - self.y_max[1]))
        # Temperature Lower Bound Penalty
        self.opti.subject_to(self.P[2, ll] >= self.lb_pen_w[0,1] * (self.y_min[2] - self.ys[2, ll]))
        # Temperature Upper Bound Penalty
        self.opti.subject_to(self.P[3, ll] >= self.ub_pen_w[0,1] * (self.ys[2, ll] - self.y_max[2]))

        # Humidity Lower Bound Penalty
        self.opti.subject_to(self.P[4, ll] >= self.lb_pen_w[0,2] * (self.y_min[3] - self.ys[3, ll]))
        # Humidity Upper Bound Penalty
        self.opti.subject_to(self.P[5, ll] >= self.ub_pen_w[0,2] * (self.ys[3, ll] - self.y_max[3]))


    def define_nlp(self, p: Dict[str, Any]) -> None:
        """
        Define the optimization problem for the nonlinear MPC using CasADi's Opti stack.
        Including the cost function, constraints, and bounds.
        """
        # Create an Opti instance
        self.opti = ca.Opti()

        # Control Variables
        self.us = self.opti.variable(self.nu, self.Np)  # Control inputs (nu x Np)

        # Number of penalties (CO2 lb, CO2 ub, Temp lb, Temp ub, humidity lb, humidity ub)
        num_penalties = 6

        # Penalty Variables
        self.P = self.opti.variable(num_penalties, self.Np)

        # State Variables
        self.xs = self.opti.variable(self.nx, self.Np+1)

        # Output Variables
        self.ys = self.opti.variable(self.ny, self.Np)

        # Parameters
        self.x0 = self.opti.parameter(self.nx, 1)  # Initial state
        self.ds = self.opti.parameter(self.nd, self.Np)  # Disturbances
        self.init_u = self.opti.parameter(self.nu, 1)  # Initial control input

        self.Js = 0

        for ll in range(self.Np):
            self.opti.subject_to(self.xs[:, ll+1] == self.F(self.xs[:, ll], self.us[:, ll], self.ds[:, ll], p))
            self.opti.subject_to(self.ys[:, ll] == self.g(self.xs[:, ll+1]))
            
            if ll < self.Np-1:
                self.opti.subject_to(-self.du_max<=(self.us[:,ll+1] - self.us[:,ll]<=self.du_max))     # Change in input Constraint
            self.opti.subject_to(self.u_min <= (self.us[:,ll] <= self.u_max))                     # Input   Contraints

            self.set_slack_variables(ll, p)

            # COST FUNCTION WITH PENALTIES
            delta_dw = self.xs[0, ll+1] - self.xs[0, ll]
            self.Js -= compute_economic_reward(delta_dw, p, self.h, self.us[:,ll])
            self.Js += self.slack_penalty(self.P[:, ll])
            # self.cost_function_with_P(p)

        # Constraints on intial state and input
        self.opti.subject_to(-self.du_max <= (self.us[:,0] - self.init_u <= self.du_max))   # Initial change in input Constraint
        self.opti.subject_to(self.xs[:,0] == self.x0)     # Initial Condition Constraint
        self.opti.minimize(self.Js)

        # Solver options
        self.opti.solver('ipopt', self.nlp_opts)

    def slack_penalty(self, Pk) -> ca.SX:
        """
        Slack variable penalty function.
        """
        return ca.sum1(Pk)

    def cost_function_with_P(self, params) -> ca.SX:
        """
        Cost function including penalty variables `P` for constraint violations.
        """
        Js = self.economic_cost_function(params)\
            + ca.sum1(ca.sum2(self.P))  # Sum of penalty variables
        return Js


    def solve_nlp_problem(self, x0_value, u0_value, d_value):
        """Given initial values for state and disturbances, solve the optimization problem.

        Args:
            x0_value (np.ndarray): Initial state value.
            d_value (np.ndarray): Disturbance values for the prediction horizon.

        Returns:
            np.ndarray: Optimal control inputs.
            float: Optimal cost value.
            dict: Solver output.
        """

        # Set parameter values
        self.opti.set_value(self.x0, x0_value)
        self.opti.set_value(self.ds, d_value)
        self.opti.set_value(self.init_u, u0_value)

        # Solve the problem
        sol = self.opti.solve()
        us_opt = sol.value(self.us)
        Js_opt = sol.value(self.Js)
        # Retrieve P if needed
        P_opt = sol.value(self.P)
        ys_opt = sol.value(self.ys)
        xs_opt = sol.value(self.xs)
        # Update initial guess for next iteration
        self.u0 = us_opt
        delta_dw = xs_opt[0, 1] - xs_opt[0, 0]
        rew = compute_economic_reward(delta_dw, get_parameters(), self.h, us_opt[:, 0])

        return us_opt, Js_opt, sol, rew

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
    ) -> None:

        self.project_name = project_name
        self.save_name = save_name
        self.mpc = mpc
        self.x = np.zeros((self.mpc.nx, self.mpc.N+1))
        self.y = np.zeros((self.mpc.nx, self.mpc.N+1))
        self.x[:, 0] = np.array(mpc.x_initial)
        self.y[:, 0] = mpc.g(self.x[:, 0]).toarray().ravel()
        self.u = np.zeros((self.mpc.nu, self.mpc.N+1))
        self.d = load_disturbances(
            weather_filename,
            self.mpc.L,
            self.mpc.start_day,
            self.mpc.h,
            self.mpc.Np,
            self.mpc.nd
        )

        self.uopt = np.zeros((mpc.nu, mpc.Np, mpc.N+1))
        self.J = np.zeros((1, mpc.N))
        self.dJdu = np.zeros((mpc.nu*mpc.Np, mpc.N))
        self.output = []
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

        for kk in range(self.mpc.N):

            print(f"Step: {kk}")
            us_opt, Js_opt, sol, eco_rew = mpc.solve_nlp_problem(self.x[:, kk], self.u[:, kk], self.d[:, kk:kk+self.mpc.Np])
            # if the solver fails, use the previous control input

            self.u[:, kk+1] = us_opt[:, 0]
            self.update_results(us_opt, Js_opt, sol, eco_rew, kk)
            self.x[:, kk+1] = mpc.F(self.x[:, kk], self.u[:, kk+1], self.d[:, kk], p).toarray().ravel()
            self.y[:, kk+1] = mpc.g(self.x[:, kk+1]).toarray().ravel()

    def update_results(self, us_opt, Js_opt, sol, eco_rew, step):
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
        self.rewards[:, step] = eco_rew
        

    def save_results(self):
        """
        """
        data = {}
        # transform the weather variables to the right units
        self.d[1, :] = co2dens2ppm(self.d[2, :], self.d[1, :])
        self.d[3, :] = vaporDens2rh(self.d[2, :], self.d[3, :])
        data["time"] = self.mpc.t / 86400
        for i in range(self.x.shape[0]):
            data[f"x_{i}"] = self.x[i, :mpc.N]
        for i in range(self.y.shape[0]):
            data[f"y_{i}"] = self.y[i, :mpc.N]
        for i in range(self.u.shape[0]):
            data[f"u_{i}"] = self.u[i, 1:]
        for i in range(self.d.shape[0]):
            data[f"d_{i}"] = self.d[i, :mpc.N]
        # for i in range(self.uopt.shape[0]):
        #     for j in range(self.uopt.shape[1]):
        #         data[f"uopt_{i}_{j}"] = self.uopt[i, j, :-1]
        data["J"] = self.J.flatten()
        data["econ_rewards"] = self.rewards.flatten()

        df = pd.DataFrame(data, columns=data.keys())
        df.to_csv(f"data/{self.project_name}/{self.save_name}.csv", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_name", type=str)
    parser.add_argument("--env_id", type=str, default="LettuceGreenhouse")
    parser.add_argument("--save_name", type=str)
    parser.add_argument("--weather_filename", default="outdoorWeatherWurGlas2014.csv", type=str)
    args = parser.parse_args()
    # load the config file
    with open(f"configs/envs/{args.env_id}.yml", "r") as file:
        env_params = yaml.safe_load(file)

    with open("configs/models/mpc.yml", "r") as file:
        mpc_params = yaml.safe_load(file)

    # p = DefineParameters()
    p = get_parameters()
    mpc = MPC(**env_params, **mpc_params[args.env_id])
    mpc.define_nlp(p)
    exp = Experiment(mpc, args.save_name, args.project_name, args.weather_filename)
    exp.solve_nmpc(p)
    exp.save_results()
