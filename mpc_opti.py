import argparse
from typing import Any, Dict, List

import yaml
import numpy as np
import pandas as pd
import casadi as ca

from utils import *

class MPC:
    def __init__(
            self,
            nx: int,
            nu: int,
            ny: int,
            nd: int,
            h: float,
            n_days: int,
            Np: int,
            Ns: int,
            start_day: int,
            sigmad: float,
            lb_pen_w: List[float],
            ub_pen_w: List[float],
            constraints: Dict[str, Any],
            nlp_opts: Dict[str, Any]
            ) -> None:
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

        self.F, self.g = define_model(self.h, self.x_min, self.x_max)

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

    def define_nlp(self, p: Dict[str, Any], constraints) -> None:
        """
        Define the optimization problem for the nonlinear MPC using CasADi's Opti stack.
        Including the cost function, constraints, and bounds.
        """
        # Create an Opti instance
        opti = ca.Opti()
        self.opti = opti  # Save the Opti instance for use in other methods

        # Control Variables
        self.us = opti.variable(self.nu, self.Np)  # Control inputs (nu x Np)

        # Penalty Variables
        num_penalties = 4  # Number of penalties (CO₂ lb, CO₂ ub, Temp lb, Temp ub)
        self.P = opti.variable(num_penalties, self.Np)  # Penalty variables

        # State Variables
        self.xs = opti.variable(self.nx, self.Np+1)  # States (nx x Np+1)

        # Output Variables
        self.ys = opti.variable(self.ny, self.Np)  # Outputs (ny x Np)

        # Parameters
        self.x0 = opti.parameter(self.nx, 1)  # Initial state
        self.ds = opti.parameter(self.nd, self.Np)  # Disturbances
        self.init_u = opti.parameter(self.nu, 1)  # Initial control input

        # ys = []
        for ll in range(self.Np):
            opti.subject_to(self.xs[:, ll+1] == self.F(self.xs[:, ll], self.us[:, ll], self.ds[:, ll], p))
            # self.ys[:, ll] = self.g(self.xs[:, ll+1])
            opti.subject_to(self.ys[:, ll] == self.g(self.xs[:, ll+1]))
            
            if ll < self.Np-1:                                         
                opti.subject_to(-self.du_max<=(self.us[:,ll+1] - self.us[:,ll]<=self.du_max))     # Change in input Constraint
            opti.subject_to(self.u_min <= (self.us[:,ll] <= self.u_max))                     # Input   Contraints


        # Convert xs and ys to CasADi matrices
        # xs = ca.horzcat(*xs)
        # ys = ca.horzcat(*ys)

        # Objective Function
        if constraints == "pen-dec-var":
            # COST FUNCTION WITH PENALTIES
            Js = self.cost_function_with_P(self.ys, self.us, p, self.P)
            # opti.minimize(Js)

            # Constraints
            for ll in range(self.Np):
                # Penalty Constraints
                opti.subject_to(self.P[:, ll] >= 0)
                # CO2 Lower Bound Penalty
                opti.subject_to(self.P[0, ll] >= self.lb_pen_w[0,0] * (self.y_min[1] - self.ys[1, ll]))
                # CO2 Upper Bound Penalty
                opti.subject_to(self.P[1, ll] >= self.ub_pen_w[0,0] * (self.ys[1, ll] - self.y_max[1]))
                # Temperature Lower Bound Penalty
                opti.subject_to(self.P[2, ll] >= self.lb_pen_w[0,1] * (self.y_min[2] - self.ys[2, ll]))
                # Temperature Upper Bound Penalty
                opti.subject_to(self.P[3, ll] >= self.ub_pen_w[0,1] * (self.ys[2, ll] - self.y_max[2]))

        elif constraints == "hard":
            # COST FUNCTION WITHOUT PENALTIES
            Js = -1e3 * p[26] * (self.ys[0, -1] - self.ys[0, 0]) + \
                 1e-6 * self.h * p[24] * ca.sum2(self.us[0, :]) + \
                 self.h / 3600 * p[23] * ca.sum2(self.us[2, :])
            # opti.minimize(Js)

            # Hard constraints on outputs
            opti.subject_to(self.ys[1:, :] >= self.y_min[1:, None])
            opti.subject_to(self.ys[1:, :] <= self.y_max[1:, None])

        opti.subject_to(-self.du_max <= (self.us[:,0] - self.init_u <= self.du_max))   # Initial change in input Constraint
        opti.subject_to(self.xs[:,0] == self.x0)     # Initial Condition Constraint
        opti.minimize(Js)

        # Save variables
        self.Js = Js

        # Solver options
        opti.solver('ipopt', self.nlp_opts)

    def cost_function_with_P(self, ys, us, params, P) -> ca.SX:
        """
        Cost function including penalty variables `P` for constraint violations.
        """
        # Objective Terms
        print("lettuce price", params[26])
        print("Co2 price", params[24] * 1e-6 * self.h)
        print("heating price", self.h / 3600 * params[23] * 1e-3)

        Js = -(1e-3 * params[26] * (ys[0, -1] - ys[0, 0]) - \
             1e-6 * self.h * params[24] * ca.sum2(us[0, :]) - \
             self.h / 3600 * params[23] * 1e-3 * ca.sum2(us[2, :]) - \
             ca.sum1(ca.sum2(P)))  # Sum of penalty variables

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
        opti = self.opti

        # Set parameter values
        opti.set_value(self.x0, x0_value)
        opti.set_value(self.ds, d_value)
        opti.set_value(self.init_u, u0_value)

        # Optionally set initial guesses
        opti.set_initial(self.us, self.u0)

        # Solve the problem
        try:
            sol = opti.solve()
            us_opt = sol.value(self.us)
            Js_opt = sol.value(self.Js)
            # Retrieve P if needed
            P_opt = sol.value(self.P)
            # Update initial guess for next iteration
            self.u0 = us_opt

        except RuntimeError as e:
            print("Solver failed to converge")
            # Handle the failure
            sol = None
            us_opt = None
            Js_opt = None
            P_opt = None
            self.u0 = np.zeros((self.nu, self.Np))

        return us_opt, Js_opt, sol

class Experiment:
    def __init__(self, mpc: MPC, save_name: str) -> None:
        self.save_name = save_name
        self.mpc = mpc
        self.x = np.zeros((self.mpc.nx, self.mpc.N+1))
        self.y = np.zeros((self.mpc.nx, self.mpc.N))
        self.x[:, 0] = np.array([0.0035, 1e-03, 15, 0.008])
        self.u = np.zeros((self.mpc.nu, self.mpc.N+1))
        self.d = LoadDisturbancesMpccsv(self.mpc)

        self.uopt = np.zeros((mpc.nu, mpc.Np, mpc.N+1))
        self.J = np.zeros((1, mpc.N))
        self.dJdu = np.zeros((mpc.nu*mpc.Np, mpc.N))
        self.output = []
        self.gradJ = np.zeros((mpc.nu*mpc.Np, mpc.N, 1))
        self.H = np.zeros((mpc.nu*mpc.Np, mpc.nu*mpc.Np, mpc.N))

        self.step = 0

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
            print(f"Step {self.step}")
            self.x[:, kk+1] = mpc.F(self.x[:, kk], self.u[:, kk], self.d[:, kk], p).toarray().ravel()
            self.y[:, kk] = mpc.g(self.x[:, kk]).toarray().ravel()

            # mpc.update_nlp(self.x[:, kk], self.u[:, kk], self.d[:, kk:kk+self.mpc.Np])

            us_opt, Js_opt, sol = mpc.solve_nlp_problem(self.x[:, kk], self.u[:, kk], self.d[:, kk:kk+self.mpc.Np])

            self.update_results(us_opt, Js_opt, sol, kk)

            # if the solver fails, use the previous control input
            if not sol:
                self.u[:, kk+1] = self.uopt[:, 0, kk-1]
            else:
                self.u[:, kk+1] = us_opt[:, 0]
            # mpc.u0 = us_opt
            self.step += 1

        return us_opt, Js_opt, sol

    def update_results(self, us_opt, Js_opt, sol, step):
        """

        Args:
            uopt (_type_): _description_
            J (_type_): _description_
            output (_type_): _description_
            gradJ (_type_): _description_
            H (_type_): _description_
            step (_type_): _description_
        """
        self.uopt[:,:, step] = us_opt
        self.J[:, step] = Js_opt
        self.output.append(sol)
        # self.gradJ[:, step] = gradJ
        # self.H[:,:, step] = H

    def save_results(self):
        """
        """
        data = {}
        # transform the weather variables to the right units
        self.d[1, :] = co2dens2ppm(self.d[2, :], self.d[1, :])
        self.d[3, :] = vaporDens2rh(self.d[2, :], self.d[3, :])
        data["time"] = self.mpc.t/86400
        for i in range(self.x.shape[0]):
            data[f"x_{i}"] = self.x[i, :mpc.N]
        for i in range(self.y.shape[0]):
            data[f"y_{i}"] = self.y[i, :mpc.N]
        for i in range(self.u.shape[0]):
            data[f"u_{i}"] = self.u[i, :mpc.N]
        for i in range(self.d.shape[0]):
            data[f"d_{i}"] = self.d[i, :mpc.N]
        # for i in range(self.uopt.shape[0]):
        #     for j in range(self.uopt.shape[1]):
        #         data[f"uopt_{i}_{j}"] = self.uopt[i, j, :-1]
        data["J"] = self.J.flatten()

        df = pd.DataFrame(data, columns=data.keys())
        df.to_csv(f"data/mpc/{self.save_name}.csv", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_name", type=str)
    parser.add_argument("--constraints", type=str, default="pen-dec-var")
    args = parser.parse_args()
    # load the config file
    with open("configs/mpc.yml", "r") as file:
        mpc_params = yaml.safe_load(file)

    # p = DefineParameters()
    p = get_parameters()
    mpc = MPC(**mpc_params["lettuce"])
    mpc.define_nlp(p, constraints=args.constraints)
    exp = Experiment(mpc, args.save_name)
    exp.solve_nmpc(p)
    exp.save_results()
