from pprint import pprint
from typing import Any, Dict, List
import matplotlib.pyplot as plt
import pandas as pd

import yaml
import numpy as np
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
            sigmad: float,
            weights: List[float],
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
        self.t = np.arange(0, self.L + self.h, self.h)
        self.N = len(self.t)
        self.weights = weights
        self.Ns = Ns
        self.sigmad = sigmad
        self.lbg = []
        self.ubg = []
        self.constraints = constraints
        self.nlp_opts = nlp_opts

        # initialize the boundaries for the optimization problem
        self.init_nmpc()


    def init_nmpc(self):
        """Initialize the nonlinear MPC problem.
        Especially, initialise the constraints, bounds, and the optimization problem.
        """
        ahMax = rh2vaporDens(self.constraints["temp_max"], self.constraints["rh_max"])    # upper bound on vapor density                  [kg/m^{-3}]                 
        self.x_min = np.array([self.constraints["W_min"], self.constraints["co2_min"], self.constraints["temp_min"], self.constraints["rh_min"]])
        self.x_max = np.array([self.constraints["W_max"], self.constraints["co2_max"], self.constraints["temp_max"], ahMax])
        self.y_min = np.array([self.constraints["W_min"], self.constraints["co2_min"], self.constraints["temp_min"], self.constraints["rh_min"]])
        self.y_max = np.array([self.constraints["W_max"], co2dens2ppm(self.constraints["temp_max"], self.constraints["co2_max"]) / 1e3, self.constraints["temp_max"], vaporDens2rh(self.constraints["temp_max"], ahMax)])
        # # output constraints vector
        # self.y_min_k'] = []
        # ops['y_max_k'] = []

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
        l = ca.horzcat(self.y_min[1:] - ys[1:, 1:])
        l_pen = ca.mtimes(self.weights, ca.fmax(0, l))

        u = ca.horzcat(ys[1:, 1:] - self.y_max[1:])
        u_pen = ca.mtimes(self.weights, ca.fmax(0, u))

        Js = -1e3 * p['productPrice2'] * (ys[0,-1]-ys[0,0]) +\
            1e-6 * self.h*p['co2Cost'] * ca.sum2(us[0,:])+\
            self.h * p['energyCost'] * ca.sum2(us[2,:]) +\
            ca.sum2(l_pen) + ca.sum2(u_pen)
        return Js

    def define_nlp(self, p: Dict[str, Any], penalty) -> Dict[str, Any]:
        """Define the optimization problem for the nonlinear MPC.
        Including the cost function, constraints, and bounds.
        
        """
        self.x0  = ca.SX.sym('x0', self.nx, 1)       # initial state# initial state
        self.us  = ca.SX.sym('us', self.nu, self.Np) # control inputs
        self.dus = ca.diff(self.us, 1, 1)                 # control inputs rate of change
        self.Us  = ca.transpose(self.us)[:]               # vectorized control inputs Us 
        self.ds  = ca.SX.sym('ds', self.nd, self.Np) # disturbance variables
        xs  = ca.SX.sym('xs', self.nx, self.Np+1)  # state variables
        ys  = ca.SX.sym('ys', self.ny, self.Np)    # output variables

        xs[:, 0] = self.x0
        for ll in range(self.Np):
            xs[:,ll+1] = f(xs[:,ll], self.us[:,ll], self.ds[:,ll], p, self.h)
            ys[:,ll]   = g(xs[:,ll])

        if penalty:
            # COST FUNCTION FOR LETTUCE PRODUCTION WITH PENALTY ON CONSTRAINTS
            self.Js = self.cost_pen_function(ys, self.us, p)
            self.cs = ca.horzcat([],[]) 
        else:
            # COST FUNCTION FOR LETTUCE PRODUCTION NO CONSTRAINTS in the cost function
            self.Js = -1e3*p['productPrice2']*(ys[0,-1]-ys[0,0]) + \
                1e-6*self.h*p['co2Cost']*casadi.sum2(self.us[0,:])+ \
                self.h*p['energyCost']*casadi.sum2(self.us[2,:])

            # self.Js = -1e3*p['productPrice2'] * (ys[0,-1]-ys[0,0]) + \
            #     1e-6*self.h*p['co2Cost'] * ca.sum2(self.us[0,:]) + \
            #             self.h*p['energyCost'] * ca.sum2(self.us[2,:])

            self.cs = self.state_constraint_function(ys)
            self.set_state_bounds()
        # we always set the control bounds
        self.set_control_bounds()
        self.set_control_change_bounds()
        # THIS IS THE COST FUNCTION USED BY MATLAB...
        # Js = -1e3*ys[0,-1] + 10.*ca.sum2(us[0,:])+ \
        #             ca.sum2(us[1,:]) + ca.sum2(us[2,:]) + ca.sum2(l_pen) + ca.sum2(u_pen)
        # equality constraints
        self.ceqs = ca.horzcat([],[]) #ca.horzcat(ys[0,-1]-2, ys[1,-1]-1)                    
        self.xs = ca.Function('xs', [self.x0, self.us, self.ds], [xs])
        self.ys = ca.Function('ys', [self.x0, self.us, self.ds], [ys])
        self.Fs = ca.Function(
            'Fs',
            [self.x0, self.Us, self.ds],
            [self.Js, self.cs, self.ceqs],
            ['x0', 'Us', 'ds'],
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
        self.dUs = casadi.transpose(casadi.horzcat(self.us[:,0] - u0, self.dus))
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

            lbg  = casadi.vertcat(lbg, lbgi)
            ubg  = casadi.vertcat(ubg, ubgi)

            g = casadi.vertcat(g, gi)
            cost = cost + costi

        cost = cost/self.Ns

        nlp    = {'f': cost, 'x': self.Us, 'g': g}
        solver  = casadi.nlpsol('solver', 'ipopt', nlp, self.nlp_opts)

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

        g     = casadi.horzcat(c, ceq, casadi.transpose(self.dUs))
        lbg   = self.lbg
        ubg   = self.ubg
        return cost, g, lbu, ubu, lbg, ubg 

class Experiment:
    def __init__(self, mpc: MPC) -> None:
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
        # mpc.update_nlp(self.x[:, self.step], self.u[:, self.step], self.d[:, self.step:self.step+self.mpc.Np])

        # uopt, J, _, output, _, gradJ, H = mpc.solve_nlp_problem(self.x[:, self.step], self.d[:, self.step:self.step+self.mpc.Np])
        # self.update_results(uopt, J, output, gradJ, H, self.step)
        for kk in range(self.mpc.N):
            print(f"Step {self.step}")
            self.x[:, kk+1] = f(self.x[:, kk], self.u[:, kk], self.d[:, kk], p, mpc.h)
            self.y[:, kk] = g(self.x[:, kk])

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
        df.to_csv("data/results.csv", index=False)

if __name__ == "__main__":
    # load the config file
    with open("configs/mpc.yml", "r") as file:
        mpc_params = yaml.safe_load(file)
    p = DefineParameters()
    mpc = MPC(**mpc_params["lettuce"])
    mpc.init_nmpc()
    mpc.define_nlp(p, penalty=False)
    exp = Experiment(mpc)
    exp.solve_nmpc(p)
    exp.save_results()
