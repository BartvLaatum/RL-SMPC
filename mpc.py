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
        ah_max = rh2vaporDens(self.constraints["temp_max"], self.constraints["rh_max"])    # upper bound on vapor density                  [kg/m^{-3}]      
        ah_min = rh2vaporDens(self.constraints["temp_min"], self.constraints["rh_min"])    # lower bound on vapor density                  [kg/m^{-3}]
        co2_dens_min = co2ppm2dens(self.constraints["co2_min"], self.constraints["temp_min"]) # lower bound on co2 density                   [kg/m^{-3}]           
        co2_dens_max = co2ppm2dens(self.constraints["co2_max"], self.constraints["temp_max"]) # upper bound on co2 density                   [kg/m^{-3}]
        self.x_min = np.array([self.constraints["W_min"], 0., 5., ah_min])
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

    def state_constraint_function(self, ys: ca.SX) -> ca.DM:
        """Putting state constraints in the form of inequality constraints.

        Args:
            ys (ca.SX): the state measurements

        Returns:
            ca.DM: the inequality constraints
        """
        cs = ca.horzcat(
            ys[1:, 1:]-self.y_max[1:], 
            -ys[1:, 1:]+self.y_min[1:]
        )       # Ensure ys >= y_min
        cs = ca.reshape(cs, 1, -1)

        return cs

    def set_control_bounds(self) -> None:
        """Set the control bounds for the optimization problem.
        """
        self.lbu = []
        self.ubu = []

        for ll in range(self.nu):
            self.lbu = np.append(self.lbu, self.u_min[ll]*np.ones((self.Np, 1)))
            self.ubu = np.append(self.ubu, self.u_max[ll]*np.ones((self.Np, 1)))

    def set_state_bounds(self):
        """Set the bounds for the inequality constraints. 
        """
        num_constraints = self.cs.shape[1]  # Total number of constraints in `self.cs`

        # Set lbg to -inf for all constraints since we only have upper bounds (<=0)
        num_constraints = self.cs.shape[1]  # Total number of constraints in `cs`

        # Set lbg to -inf for all constraints since we only have upper bounds (<=0)
        self.lbg = -np.inf * np.ones((num_constraints,))

        # Set ubg to 0 for all constraints to enforce the inequalities in `cs`
        self.ubg = np.zeros((num_constraints,))


    def set_control_change_bounds(self):
        """Set the control bounds for the optimization problem.
        """
    # set constraints for control input changes
        for ll in range(self.nu):
            self.lbg   = np.append(self.lbg, -self.du_max[ll]*np.ones((self.Np, 1)))
            self.ubg   = np.append(self.ubg,  self.du_max[ll]*np.ones((self.Np, 1)))


    def cost_pen_function(self, ys, us, p) -> ca.SX:
        """Set the cost function penalties for the constraints in the optimization problem.

        Args:
            ys (ca.SX): the state measurements
            us (ca.SX): the control measurements

        Returns:
            ca.SX: the cost function with penalties
        """
        l = ca.horzcat(self.y_min[1:] - ys[1:, :])
        l_pen = ca.mtimes(self.lb_pen_w, ca.fmax(0, l))
        u = ca.horzcat(ys[1:, :] - self.y_max[1:])
        u_pen = ca.mtimes(self.ub_pen_w, ca.fmax(0, u))

        Js = -1e3 * p[26] * (ys[0,-1]-ys[0,0]) +\
            1e-6 * self.h*p[24] * ca.sum2(us[0,:])+\
            self.h/3600 * p[23] * 1e-3 * ca.sum2(us[2,:]) +\
            ca.sum2(l_pen) + ca.sum2(u_pen)
        return Js

    def cost_function_with_P(self, ys, us, params, P) -> ca.SX:
        """
        Cost function including penalty variables `P` for constraint violations.
        """
        # Objective Terms (e.g., maximizing dry mass growth, minimizing control effort)
        Js = -1e3 * params[26] * (ys[0, -1] - ys[0, 0]) + \
            1e-6 * self.h * params[24] * ca.sum2(us[0, :]) + \
            self.h / 3600. * params[23] * 1e-3 * ca.sum2(us[2, :]) + \
            ca.sum1(ca.sum2(P))  # Sum of penalty variables

        return Js

    def define_nlp(self, p: Dict[str, Any], constraints) -> Dict[str, Any]:
        """
        Define the optimization problem for the nonlinear MPC.
        Including the cost function, constraints, and bounds.
        """
        self.x0  = ca.SX.sym('x0', self.nx, 1)          # initial state# initial state
        self.us  = ca.SX.sym('us', self.nu, self.Np)    # control inputs
        self.dus = ca.diff(self.us, 1, 1)               # control inputs rate of change
        self.Us  = ca.transpose(self.us)[:]             # vectorized control inputs Us 
        self.ds  = ca.SX.sym('ds', self.nd, self.Np)    # disturbance variables
        xs  = ca.SX.sym('xs', self.nx, self.Np+1)       # state variables
        ys  = ca.SX.sym('ys', self.ny, self.Np)         # output variables
        self.P = ca.SX.sym('P', 4, self.Np)  # Penalty variables
        penalty_constraints = []

        xs[:, 0] = self.x0
        for ll in range(self.Np):
            xs[:,ll+1] = self.F(xs[:,ll], self.us[:,ll], self.ds[:,ll], p)
            ys[:,ll]   = self.g(xs[:,ll])

            penalty_constraints += [self.P[:, ll] >= 0]
            penalty_constraints.append(
                self.P[0, ll] >= self.lb_pen_w[0,0] * (self.y_min[1] - ys[1, ll])
            )
            # CO₂ Upper Bound Penalty
            penalty_constraints.append(
                self.P[1, ll] >= self.ub_pen_w[0,0] * (ys[1, ll] - self.y_max[1])
            )
            # Temperature Lower Bound Penalty
            penalty_constraints.append(
                self.P[2, ll] >= self.lb_pen_w[0,1] * (self.y_min[2] - ys[2, ll])
            )
            # Temperature Upper Bound Penalty
            penalty_constraints.append(
                self.P[3, ll] >= self.ub_pen_w[0,1] * (ys[2, ll] - self.y_max[2])
            )

        if constraints == "penalty":
        # Penalty Constraints for CO₂ and Temperature
            # COST FUNCTION FOR LETTUCE PRODUCTION WITH PENALTY ON CONSTRAINTS
            self.Js = self.cost_pen_function(ys, self.us, p)
            self.cs = ca.horzcat([],[])

        elif constraints == "pen-dec-var":
            # COST FUNCTION FOR LETTUCE PRODUCTION NO CONSTRAINTS in the cost function
            self.Js = self.cost_function_with_P(ys, self.us, p, self.P)
            self.cs = ca.horzcat([],[])
            
        elif constraints == "hard":
            # COST FUNCTION FOR LETTUCE PRODUCTION NO CONSTRAINTS in the cost function
            self.Js = -1e3*p[26]*(ys[0,-1]-ys[0,0]) + \
                1e-6*self.h*p[24]*casadi.sum2(self.us[0,:])+ \
                self.h*p[23]*casadi.sum2(self.us[2,:])


            self.cs = self.state_constraint_function(ys)
            self.set_state_bounds()

        # set the control bounds
        self.set_control_bounds()
        self.set_control_change_bounds()

        # Combine self.Us and self.P in the decision vector
        self.w = ca.vertcat(ca.reshape(self.Us, -1, 1), ca.reshape(self.P, -1, 1))


        # equality constraints
        self.ceqs = ca.horzcat([],[]) #ca.horzcat(ys[0,-1]-2, ys[1,-1]-1)                    
        self.xs = ca.Function('xs', [self.x0, self.us, self.ds], [xs])
        self.ys = ca.Function('ys', [self.x0, self.us, self.ds], [ys])
        self.Fs = ca.Function(
            'Fs',
            [self.x0, self.w, self.ds],
            [self.Js, self.cs, self.ceqs],
            ['x0', 'w', 'ds'],
            ['J', 'c', 'ceq']
        )

    def update_nlp(self, x0, u0, d):
        """Update the optimization problem with new values for the state, control, and disturbance variables.

        Args:
            x0 (np.ndarray): the initial state
            u0 (np.ndarray): value of the previous executed control signal.
            d (np.ndarray): the disturbances
        """
        # compute the change of the previous control input.
        self.dUs = ca.transpose(ca.horzcat(self.us[:,0] - u0, self.dus))
        self.dUs = self.dUs[:]

    def solve_nlp_problem(self, x0, d):
        """Given intial values for state, control, and disturbances, solve the optimization problem.

        Args:
            mpc (MPC): class that holds the model predictive control problem.
            x0 (np.array-like): _description_
            d (_type_): _description_
            p (_type_): _description_
            ops (_type_): _description_

        Returns:
            _type_: _description_
        """
        # u0      = self.u0                 # initial guess for the control signals
        # u0      = np.matrix.flatten(u0)
        g       = []
        lbg     = []
        ubg     = []
        cost    = 0.0

        # for loop for if we aim to use randomized MPC on the weather variables
        for ll in range(self.Ns):

            dmax                      = (1+self.sigmad)*d
            dmin                      = (1-self.sigmad)*d

            di                        = np.random.uniform(low=dmin, high=dmax, size=(self.nd, self.Np))
            
            costi, gi, lbu, ubu, lbgi, ubgi = self.costfunction_nonlinearconstraints(x0, di)

            lbg  = ca.vertcat(lbg, lbgi)
            ubg  = ca.vertcat(ubg, ubgi)

            g = ca.vertcat(g, gi)
            cost = cost + costi

        cost = cost/self.Ns

        nlp    = {'f': cost, 'x': self.w, 'g': g}
        solver  = ca.nlpsol('solver', 'ipopt', nlp, self.nlp_opts)

        # Solve the NLP
        # if 'lam_x0' in ops: # if you want to set the lagrange multipliers
        #     output  = solver(x0=u0, lbx=lbu, ubx=ubu, \
        #         lbg=lbg, ubg=ubg, lam_x0=ops['lam_x0'], lam_g0=ops['lam_g0'])
        # else:
        output  = solver(x0=self.u0.flatten(), lbx=lbu, ubx=ubu,lbg=lbg, ubg=ubg)

        Uopt    = output['x'].full().flatten()
        uopt    = Uopt.reshape(self.nu, self.Np)
        V       = output['f'].full().flatten()

        dUs     = self.dUs
        Fs      = self.Fs
        grad    = np.zeros((self.nu*self.Np, 1))
        hessian = np.zeros((self.nu*self.Np, self.nu*self.Np))

        if solver.stats()['success'] == False: # check if solution converged
            print("Solver failed to converge")
            grad[:, 0] = 1
        
        return uopt, V[0], Fs, output, dUs, grad, hessian



    def costfunction_nonlinearconstraints(self, x0, d):
        """Compute the cost function and nonlinear constraints for a given optimization problem.

        Args:
            x0 (array-like): Initial state vector.
            d (array-like): Disturbance vector.
        Returns:
            cost (float) : The computed cost value.
            g (array-like) : Vector of inequality and equality constraints.
            lbu (array-like): Lower bounds on the control inputs.
            ubu (array-like): Upper bounds on the control inputs.
            lbg (array-like): Lower bounds on the constraints.
            ubg (array-like): Upper bounds on the constraints.
        """
        # print("Computing cost function and constraints")
        # print(f"x0: {x0}")
        # print("d: ", d[:,0])
        # print(d.shape)
        temp  = self.Fs.call({'x0': x0, 'Us': self.Us, 'ds': d})

        cost  = temp['J']
        c     = temp['c']
        ceq   = temp['ceq']
                    
        lbu   = self.lbu
        ubu   = self.ubu

        g     = ca.horzcat(c, ceq, ca.transpose(self.dUs))
        lbg   = self.lbg
        ubg   = self.ubg
        return cost, g, lbu, ubu, lbg, ubg 

class Experiment:
    def __init__(self, mpc: MPC, save_name: str) -> None:
        self.save_name = save_name
        self.mpc = mpc
        self.x = np.zeros((self.mpc.nx, self.mpc.N+1))
        self.y = np.zeros((self.mpc.nx, self.mpc.N))
        self.x[:, 0] = np.array([0.0035, 1e-03, 15, 0.008])
        self.u = np.zeros((self.mpc.nu, self.mpc.N+1))
        self.d = LoadDisturbancesMpc(self.mpc)

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

            mpc.update_nlp(self.x[:, kk], self.u[:, kk], self.d[:, kk:kk+self.mpc.Np])

            uopt, J, _, output, _, gradJ, H = mpc.solve_nlp_problem(self.x[:, kk], self.d[:, kk:kk+self.mpc.Np])
            self.update_results(uopt, J, output, gradJ, H, kk)

            # propagate the state.
            self.u[:, kk+1] = uopt[:, 0]
            mpc.u0 = uopt
            self.step += 1

        return uopt, J, output, gradJ, H

    def update_results(self, uopt, J, output, gradJ, H, step):
        """

        Args:
            uopt (_type_): _description_
            J (_type_): _description_
            output (_type_): _description_
            gradJ (_type_): _description_
            H (_type_): _description_
            step (_type_): _description_
        """
        self.uopt[:,:, step] = uopt
        self.J[:, step] = J
        self.output.append(output)
        self.gradJ[:, step] = gradJ
        self.H[:,:, step] = H

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
    mpc.init_nmpc()
    mpc.define_nlp(p, constraints=args.constraints)
    exp = Experiment(mpc, args.save_name)
    exp.solve_nmpc(p)
    exp.save_results()
