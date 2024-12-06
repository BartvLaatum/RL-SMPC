import argparse
from typing import Any, Dict, List, Tuple

import yaml
import numpy as np
import casadi as ca
import pandas as pd

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

    def economic_cost_function(self, params) -> ca.SX:
        """
        Economic cost function.
        """
        Js = -22.285124999999997 *  (self.xs[0, -1] - self.xs[0, 0])\
            + 0.00017154 * ca.sum2(self.us[0, :])\
            + 3.2025e-5 * ca.sum2(self.us[2, :])
        # Js = -params[26] *  (self.xs[0, -1] - self.xs[0, 0])\
        #     + 1e-6 * self.h * params[24] * ca.sum2(self.us[0, :])\
        #     + self.h / 3600 * params[23] * 1e-3 * ca.sum2(self.us[2, :])
        return Js


    def cost_function_with_P(self, params) -> ca.SX:
        """
        Cost function including penalty variables `P` for constraint violations.
        """
        Js = self.economic_cost_function(params)\
            + ca.sum1(ca.sum2(self.P))  # Sum of penalty variables
        return Js

    def set_slack_variables(self, ll: int, p: Dict[str, Any]) -> Tuple[List[ca.SX], List[float], List[float]]:
        """
        Define slack variable constraints without using Opti stack.

        Returns:
            constraints (List[ca.SX]): List of constraint expressions.
            lbg (List[float]): Lower bounds for the constraints.
            ubg (List[float]): Upper bounds for the constraints.
        """
        constraints = []
        lbg = []
        ubg = []

        num_penalties = self.P.shape[0]  # Number of penalty variables
        for i in range(num_penalties):
            constraints.append(self.P[i, ll])
            lbg.append(0)
            ubg.append(ca.inf)

        # CO2 Lower Bound Penalty
        expr = self.P[0, ll] - self.lb_pen_w[0, 0] * (self.y_min[1] - self.ys[1, ll])
        constraints.append(expr)
        lbg.append(0)
        ubg.append(ca.inf)

        # CO2 Upper Bound Penalty
        expr = self.P[1, ll] - self.ub_pen_w[0, 0] * (self.ys[1, ll] - self.y_max[1])
        constraints.append(expr)
        lbg.append(0)
        ubg.append(ca.inf)

        # Temperature Lower Bound Penalty
        expr = self.P[2, ll] - self.lb_pen_w[0, 1] * (self.y_min[2] - self.ys[2, ll])
        constraints.append(expr)
        lbg.append(0)
        ubg.append(ca.inf)

        # Temperature Upper Bound Penalty
        expr = self.P[3, ll] - self.ub_pen_w[0, 1] * (self.ys[2, ll] - self.y_max[2])
        constraints.append(expr)
        lbg.append(0)
        ubg.append(ca.inf)

        # Humidity Lower Bound Penalty
        expr = self.P[4, ll] - self.lb_pen_w[0, 2] * (self.y_min[3] - self.ys[3, ll])
        constraints.append(expr)
        lbg.append(0)
        ubg.append(ca.inf)

        # Humidity Upper Bound Penalty
        expr = self.P[5, ll] - self.ub_pen_w[0, 2] * (self.ys[3, ll] - self.y_max[3])
        constraints.append(expr)
        lbg.append(0)
        ubg.append(ca.inf)

        return constraints, lbg, ubg


    def define_nlp(self, p) -> None:
        """
        Define the optimization problem for the nonlinear MPC without using CasADi's Opti stack.
        """
        # Decision Variables
        num_penalties = 6
        self.us = ca.MX.sym('us', self.nu, self.Np)
        self.P = ca.MX.sym('P', num_penalties, self.Np)
        self.xs = ca.MX.sym('xs', self.nx, self.Np + 1)
        self.ys = ca.MX.sym('ys', self.ny, self.Np)

        # Parameters
        self.x0 = ca.MX.sym('x0', self.nx, 1)
        self.ds = ca.MX.sym('ds', self.nd, self.Np)
        self.init_u = ca.MX.sym('init_u', self.nu, 1)

        # Initialize constraints and bounds
        g = []
        lbg = []
        ubg = []

        # Initial State Constraint
        g.append(self.xs[:, 0] - self.x0)
        lbg.extend([0] * self.nx)
        ubg.extend([0] * self.nx)


        for ll in range(self.Np):
            # State Transition Constraint
            x_next = self.F(self.xs[:, ll], self.us[:, ll], self.ds[:, ll], p)
            g.append(self.xs[:, ll + 1] - x_next)
            lbg.extend([0] * self.nx)
            ubg.extend([0] * self.nx)

            # Output Constraint
            y_current = self.g(self.xs[:, ll + 1])
            g.append(self.ys[:, ll] - y_current)
            lbg.extend([0] * self.ny)
            ubg.extend([0] * self.ny)

            # Corrected Input Constraints
            g.append(self.us[:, ll])
            lbg.extend(self.u_min.tolist())
            ubg.extend(self.u_max.tolist())

            # Slack Variable Constraints
            P_constraints, P_lbg, P_ubg  = self.set_slack_variables(ll, p)
            g.extend(P_constraints)
            lbg.extend(P_lbg)
            ubg.extend(P_ubg)

            # Change in Input Constraints
            if ll < self.Np - 1:
                du = self.us[:, ll + 1] - self.us[:, ll]
                g.append(du)
                lbg.extend(-self.du_max)
                ubg.extend(self.du_max)

        # Corrected Initial Change in Input Constraint
        du0 = self.us[:, 0] - self.init_u
        g.append(du0)
        lbg.extend(-self.du_max)
        ubg.extend(self.du_max)

        # Cost Function
        Js = self.cost_function_with_P(p)

        # Decision Variable Vector
        w = ca.vertcat(
            ca.reshape(self.us, -1, 1),
            ca.reshape(self.P, -1, 1),
            ca.reshape(self.xs, -1, 1),
            ca.reshape(self.ys, -1, 1)
        )

        # Parameter Vector
        p_vector = ca.vertcat(
            ca.reshape(self.x0, -1, 1),
            ca.reshape(self.ds, -1, 1),
            ca.reshape(self.init_u, -1, 1)
        )

        # Create NLP Problem
        nlp = {'x': w, 'f': Js, 'g': ca.vertcat(*g), 'p': p_vector}

        # Solver Options
        # opts = {'ipopt': {'print_level': 0, 'tol': 1e-6}}
        self.solver = ca.nlpsol('solver', 'ipopt', nlp, self.nlp_opts)

        # Store Bounds
        self.lbg = lbg
        self.ubg = ubg

    def solve_nlp_problem(self, x0_value, u0_value, d_value, step):
        """
        Solve the optimization problem using the nlpsol interface.
        """
        # Prepare Parameters
        p_value = ca.vertcat(
            ca.reshape(x0_value, -1, 1),
            ca.reshape(d_value, -1, 1),
            ca.reshape(u0_value, -1, 1)
        )

        u0 = np.zeros(self.us.shape)
        P0 = np.zeros(self.P.shape)
        x0 = np.zeros(self.xs.shape)
        y0 = np.zeros(self.ys.shape)
        x0[:, 0] = x0_value
        y0[:, 0] = self.g(x0_value).toarray().ravel()

        w0 = np.concatenate([u0.flatten(), P0.flatten(), x0.flatten(), y0.flatten()])

        # Initial Guess for Decision Variables
        # w0 = np.zeros(self.u0)

        # Solve the NLP Problem
        try:
            sol = self.solver(x0=w0, p=p_value, lbg=self.lbg, ubg=self.ubg)

            w_opt = sol['x'].full().flatten()

            num_u = self.nu * self.Np
            num_xs = self.nx * (self.Np + 1)
            num_ys = self.ny * self.Np
            num_P = 6 * self.Np

            xs_opt = w_opt[num_u:num_u + num_xs].reshape(self.nx, self.Np + 1)
            ys_opt = w_opt[num_u + num_xs:num_u + num_xs + num_ys].reshape(self.ny, self.Np)
            P_opt = w_opt[num_u + num_xs + num_ys:num_u + num_xs + num_ys + num_P].reshape(6, self.Np)

            us_opt = w_opt[:num_u].reshape(self.nu, self.Np)


            Js_opt = sol['f'].full()[0][0]
            # Update Initial Control Input
            self.w0 = w_opt

            # fig, ax = plt.subplots(4, 4)

            # t = np.arange(0, 4)

            # # ax[0,0].step(t, ys_opt[2, :])
            # for i in range(self.ny):
            #     ax[0, i].step(t, xs_opt[i, :t.size])

            # for i in range(self.ny):
            #     ax[1, i].step(t, ys_opt[i, :t.size])
            # for i in range(3):
            #     ax[2, i].step(t, us_opt[i, :t.size])
            # for i in range(3):
            #     ax[3, i+1].step(t, P_opt[i*2, :t.size])
            #     ax[3, i+1].step(t, P_opt[i*2+1, :t.size])
            # fig.savefig(f"resultst={step}.png")
            # plt.show()


        except RuntimeError as e:
            print("Solver failed to converge")
            sol = None
            us_opt = None
            Js_opt = None
            self.u0 = np.zeros((self.nu, self.Np))

        return us_opt, Js_opt, sol

class Experiment:
    def __init__(
        self,
        mpc: MPC,
        save_name: str,
        project_name: str,
        weather_filename: str,
    ) -> None:

        self.save_name = save_name
        self.project_name = project_name
        self.mpc = mpc
        self.x = np.zeros((self.mpc.nx, self.mpc.N+1))
        self.y = np.zeros((self.mpc.nx, self.mpc.N))
        
        self.x[:, 0] = np.array(mpc.x_initial)
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
        self.gradJ = np.zeros((mpc.nu*mpc.Np, mpc.N, 1))
        self.H = np.zeros((mpc.nu*mpc.Np, mpc.nu*mpc.Np, mpc.N))


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
        # for kk in range(5):
            print(f"Step: {kk}")
            us_opt, Js_opt, sol = mpc.solve_nlp_problem(self.x[:, kk], self.u[:, kk], self.d[:, kk:kk+self.mpc.Np], step=kk)
            # if the solver fails, use the previous control input

            self.u[:, kk+1] = us_opt[:, 0]
            self.update_results(us_opt, Js_opt, sol, kk)

            self.x[:, kk+1] = mpc.F(self.x[:, kk], self.u[:, kk+1], self.d[:, kk], p).toarray().ravel()
            self.y[:, kk] = mpc.g(self.x[:, kk+1]).toarray().ravel()


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
