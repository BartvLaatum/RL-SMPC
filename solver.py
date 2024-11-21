import numpy as np

from utils import *
from time import time
from plots import *

if __name__ == "__main__":
    # Define the parameters
    ops = {"opts": {}}
    ops["opts"]["ipopt"] = dict(
        linear_solver="ma57", 
        warm_start_init_point="yes", 
        nlp_scaling_method = 'gradient-based',
        hessian_approximation="limited-memory",
        print_level=1,
        max_iter=1e3,
        # max_iter=1000,
    )
    print(ops["opts"])
    p = DefineParameters()
    ops["nx"] = 4
    ops["nu"] = 3
    ops["nd"] = 4
    ops["ny"] = 4

    nDays = 1
    c = 86400
    ops["h"] = 15*60                                    # control interval (s)
    ops["L"]  = nDays*c                                 # final time simulation
    ops["t"]  = np.arange(0,ops["L"]+ops["h"],ops["h"]) # initial time vector
    ops["N"]  = len(ops["t"])                           # number of samples in initial time vector

    # this initialises the nonlinear MPC
    ops        = InitNonlinearMpc(ops)

    # cost and nonlinear constraints
    ops        = CostConstraintFunctions(p,ops)

    # signals
    d          = LoadDisturbances(ops)
    x          = np.zeros((ops['nx'], ops['N'] + 1))
    y          = np.zeros((ops['nx'], ops['N']))
    x[:,0]     = np.array([0.0035, 1e-03, 15, 0.008])
    u          = np.zeros((ops['nu'], ops['N'] + 1))
    u[:,0]     = np.array([0, 0, 0])

    res           = {'J': np.zeros((1, ops['N']))}
    res['dJdu']   = np.zeros((ops['nu']*ops['Np'], ops['N']))
    res['Fs']     = []
    res['output'] = []
    res['dUs']    = []
    res['gradJ']  = np.zeros((ops['nu']*ops['Np'], ops['N'], 1))
    res['H']      = np.zeros((ops['nu']*ops['Np'], ops['nu']*ops['Np'], ops['N']))
    uopt          = np.zeros((ops['nu'],ops['Np'], ops['N']+1))

    start_time = time()

    ops        = UpdateCostConstraintFunctions(x[:,0], u[:,0], d[:,0:0+ops['Np']], p, ops)

    uopt[:,:,0], res['J'][0,0], __, res['output'], __,\
        res['gradJ'][:,0], res['H'][:,:,0] \
        = NmpcCasadiIpOpt_SingleShooting(x[:,0], d[:, 0:0+ops['Np']], p, ops)

    # here the optimization problem is propagated in time using the optimal control solution
    for kk in range(ops['N']):

        # propagate model one time step ahead
        x[:,kk+1]  = f(x[:, kk], u[:, kk], d[:, kk], p, ops["h"])

        # measure output from model
        y[:,kk] = g(x[:, kk])
        u[:,kk+1] = uopt[:, kk, 0]
    
    
    np.save('data/opt-control-1day.npy', u)
    np.save('data/opt-weather-1day.npy', d)
    # np.save('Opt-controls-penalties.npy', u)
    plot_measurements(ops, y, c, title='Measurements-noupdate', save=True)
    plot_controls(ops, u[:,:-1], c, title='controls-noupdate', save=True)

    print(' ')
    print(f"CPU time for the optimization:{time()- start_time:.2f}")
    print(' ')
