from typing import Dict, Any
import numpy as np
import casadi 
import math
import scipy.io as sio
# from scipy.interpolate import PchipInterpolator, interp1d, CubicSpline
from scipy.signal import resample
import statistics  




def CostConstraintFunctions(
        p: Dict[str, Any],
        ops: Dict[str, Any]
    ) -> Dict[str, Any]:
    """
    Computes the cost and constraint functions for a given set of parameters and options.

    Args:
        p (dict): Dictionary containing the parameters for the cost function.
        ops (dict): Dictionary containing the options and settings for the cost function. 
    Returns:
        dict: Updated ops dictionary with the following keys added:
                - 'xs': CasADi function for state variables.
                - 'ys': CasADi function for output variables.
                - 'Fs': CasADi function for cost, inequality constraints, and equality constraints.
                - 'x0': Symbolic variable for initial state.
                - 'ds': Symbolic variable for disturbances.
                - 'us': Symbolic variable for control inputs.
                - 'Us': Vectorized control signal.
                - 'dus': Symbolic variable for control input changes.
                - 'Js': Cost function.
                - 'cs': Inequality constraints.
                - 'ceqs': Equality constraints.
                - 'lbu': Lower bounds for control inputs.
                - 'ubu': Upper bounds for control inputs.
                - 'lbg': Lower bounds for constraints.
                - 'ubg': Upper bounds for constraints.
    """

    x0  = casadi.SX.sym('x0',ops['nx'], 1)
    us  = casadi.SX.sym('us',ops['nu'], ops['Np'])
    dus = casadi.diff(us, 1, 1)
    Us  = casadi.transpose(us)[:] # vectorized control signal Us 
    ds  = casadi.SX.sym('ds',ops['nd'], ops['Np'])
    xs  = casadi.SX.sym('xs',ops['nx'], ops['Np']+1)
    ys  = casadi.SX.sym('ys',ops['nx'], ops['Np'])

    xs[:, 0] = x0
    for ll in range(ops['Np']):
        xs[:,ll+1] = f(xs[:,ll], us[:,ll], ds[:,ll], p, ops['h'])
        ys[:,ll]   = g(xs[:,ll])

    # weights = casadi.horzcat(1e2, 1e2, 5e2)
    # l = casadi.horzcat(-ys[1:, 1:]+ops['y_min'][1:])
    # l_pen = casadi.mtimes(weights, casadi.fmax(0, l))

    # u = casadi.horzcat(ys[1:, 1:] - ops['y_max'][1:])
    # u_pen = casadi.mtimes(weights, casadi.fmax(0, u))

    # COST FUNCTION FOR LETTUCE PRODUCTION WITH CONSTRAINTS
    # Js = -1e3*p['productPrice2']*(ys[0,-1]-ys[0,0]) + 1e-6*ops['h']*p['co2Cost']*casadi.sum2(us[0,:])+ \
    #             ops['h']*p['energyCost']*casadi.sum2(us[2,:]) + casadi.sum2(l_pen) + casadi.sum2(u_pen)

    # # COST FUNCTION FOR LETTUCE PRODUCTION NO CONSTRAINTS in the cost function
    Js = -1e3*p['productPrice2']*(ys[0,-1]-ys[0,0]) + 1e-6*ops['h']*p['co2Cost']*casadi.sum2(us[0,:])+ \
                ops['h']*p['energyCost']*casadi.sum2(us[2,:])

    # THIS IS THE COST FUNCTION USED BY MATLAB...
    # Js = -1e3*ys[0,-1] + 10.*casadi.sum2(us[0,:])+ \
    #             casadi.sum2(us[1,:]) + casadi.sum2(us[2,:]) + casadi.sum2(l_pen) + casadi.sum2(u_pen)
    cs = casadi.horzcat(
        ys[1:, 1:]-ops['y_max'][1:], 
        -ys[1:, 1:]+ops['y_min'][1:]
    )       # Ensure ys >= y_min

    cs = casadi.reshape(cs, 1, -1)

    ceqs            = casadi.horzcat([],[]) #casadi.horzcat(ys[0,-1]-2, ys[1,-1]-1)                    
                  
    ops['xs']       = casadi.Function('xs',[x0,us,ds],[xs])
    ops['ys']       = casadi.Function('ys',[x0,us,ds],[ys])
    ops['Fs']       = casadi.Function('Fs',[x0,Us,ds],[Js,cs,ceqs],['x0','Us','ds'],['J','c','ceq'])

    ops['x0']       = x0
    ops['ys']       = ys
    ops['ds']       = ds
    ops['us']       = us
    ops['Us']       = Us
    ops['dus']      = dus
    ops['Js']       = Js
    ops['cs']       = cs
    ops['ceqs']     = ceqs

    # ops['lbg'] = []
    # ops['ubg'] = []

    # bounds on the control signals
    ops['lbu'] = []
    ops['ubu'] = []
    for ll in range(ops['nu']):
        ops['lbu'] = np.append(ops['lbu'], ops['u_min'][ll]*np.ones((ops['Np'], 1)))
        ops['ubu'] = np.append(ops['ubu'], ops['u_max'][ll]*np.ones((ops['Np'], 1)))

    num_constraints = cs.shape[1]  # Total number of constraints in `cs`

    # Set lbg to -inf for all constraints since we only have upper bounds (<=0)

    num_constraints = cs.shape[1]  # Total number of constraints in `cs`

    # Set lbg to -inf for all constraints since we only have upper bounds (<=0)
    ops['lbg'] = -np.inf * np.ones((num_constraints,))

    # Set ubg to 0 for all constraints to enforce the inequalities in `cs`
    ops['ubg'] = np.zeros((num_constraints, ))

    # set constraints for control input changes
    for ll in range(ops['nu']):
        ops['lbg']   = np.append(ops['lbg'], -ops['du_max'][ll]*np.ones((ops['Np'], 1)))
        ops['ubg']   = np.append(ops['ubg'],  ops['du_max'][ll]*np.ones((ops['Np'], 1)))

    return ops

def UpdateCostConstraintFunctions(x, u0, d, p, ops):
    d = d[:, 0]

    ds = ops['ds']
    Us = ops['Us']
    x0 = ops['x0']
    ys = ops['ys']
    Js = ops['Js']

    # # Update temperature constraints based on daytime/nighttime conditions
    cs = casadi.horzcat(
        ys[1:, 1:]-ops['y_max'][1:], 
        -ys[1:, 1:]+ops['y_min'][1:]
    )       # Ensure ys >= y_min
    cs = casadi.reshape(cs, 1, -1)
    # cs = casadi.horzcat([],[]) #    
    ceqs = casadi.horzcat([], [])  # Empty array for equality constraints

    # Update the CasADi function for constraints with the modified `cs` and `ceqs`
    ops['Fs'] = casadi.Function('Fs', [x0, Us, ds], [Js, cs, ceqs], ['x0', 'Us', 'ds'], ['J', 'c', 'ceq'])

    # Update the linear inequality constraint `dUs` to enforce control signal rate limits
    # print(u0)
    ops['dUs'] = casadi.transpose(casadi.horzcat(ops['us'][:,0] - u0, ops['dus']))
    print(ops['dUs'].shape)
    ops['dUs'] = ops['dUs'][:]
    print(ops['dUs'].shape)
    return ops

def NmpcCasadiIpOpt_SingleShooting(x0, d, p, ops):
      
    u0      = ops['u0']                 # initial guess for the control signals
    u0      = np.matrix.flatten(u0)
    g       = []
    lbg     = []
    ubg     = []
    cost    = 0.0
    sigmad  = .0
    Ns      = 1

    for ll in range(Ns):

        dmax                      = (1+sigmad)*d
        dmin                      = (1-sigmad)*d 
        di                        = np.random.uniform(low=dmin, high=dmax, size=(ops['nd'], ops['Np']))
        [costi, gi, lbu, ubu, lbgi, ubgi] = costfunction_nonlinearconstraints(x0, di, p, ops)

        lbg  = casadi.vertcat(lbg, lbgi)
        ubg  = casadi.vertcat(ubg, ubgi)

        g = casadi.vertcat(g, gi)
        cost = cost + costi

    cost = cost/Ns

    prob    = {'f': cost,'x': ops['Us'],'g': g} 
    solver  = casadi.nlpsol('solver','ipopt', prob, ops['opts'])
    

    # Solve the NLP
    if 'lam_x0' in ops:
        output  = solver(x0=u0, lbx=lbu, ubx=ubu, \
            lbg=lbg, ubg=ubg, lam_x0=ops['lam_x0'], lam_g0=ops['lam_g0'])
    else:
        output  = solver(x0=u0, lbx=lbu, ubx=ubu,lbg=lbg, ubg=ubg)

    Uopt    = output['x'].full().flatten()
    uopt    = np.reshape(Uopt, (ops['nu'], ops['Np']))
    V       = output['f'].full().flatten()

    dUs     = ops['dUs']
    Fs      = ops['Fs']
    grad    = np.zeros((ops['nu']*ops['Np'], 1))
    hessian = np.zeros((ops['nu']*ops['Np'], ops['nu']*ops['Np']))

    if solver.stats()['success']==False: # check if solution converged
        print("Solver failed to converge")
        grad[:,0] = 1
       
    return uopt, V[0], Fs, output, dUs, grad, hessian

def costfunction_nonlinearconstraints(x0, d, p, ops):

    temp  = ops['Fs'].call({'x0': x0, 'Us': ops['Us'], 'ds': d})

    cost  = temp['J']
    c     = temp['c']
    ceq   = temp['ceq']
                
    lbu   = ops['lbu']
    ubu   = ops['ubu']

    g     = casadi.horzcat(c, ceq, casadi.transpose(ops['dUs']))
    lbg   = ops['lbg']
    ubg   = ops['ubg']

    return cost, g, lbu, ubu, lbg, ubg 

def InitNonlinearMpc(ops):

    # NMPC prediction horizon (#time steps)
    ops['Np'] = ops['N']
    ops['Np_hours'] = ops['Np']*ops['h']/60/60

    # Controller bounds: u = [co2, vent, heat]      
    # parameter                     description                                  [unit]                 nominal value
    co2SupplyMin = 0             # lower bound on CO2 supply rate                [kg{co2} m^{-2} s^{-1}]    0
    co2SupplyMax = 1.2           # upper bound on CO2 supply rate                [mg{co2} m^{-2} s^{-1}]    1.2
    ventMin = 0                  # lower bound on ventilation rate               [m s^{-1}]                 0
    ventMax = 7.5                   # upper bound on ventilation rate            [mm s^{-1}]                7.5e
    heatMin = 0                  # lower bound on energy input through heating   [W m^{-2}]                 0
    heatMax = 150                # upper bound on energy input through heating   [W m^{-2}]                 150

    # state constraints: x = [dw, co2In, tempIn, vapIn]
    # parameter                             description 									[unit] 					nominal value
    WMin = 0                               # lower bound on dry weight                     [kg{crop} m^{-2}]        
    WMax = np.inf                    # upper bound on dry weight                     [kg{crop} m^{-2}]        
    co2Min = 0                             # lower bound on CO2 concentration              [kg{co2} m^{-3}]        0
    co2Max = 2.75e-3                       # upper bound on CO2 concentration              [kg{co2} m^{-3}]        2.75e-3
    tempMin = 10                           # lower bound on temperature                    [°C]                    6.5
    tempMax = 20                           # upper bound on temperature                    [°C]                    40
    rhMin = 0                               # lower bound on relative humidity              [#]                     0
    rhMax = 70                             # upper bound on relative humidity              [#]                     90
    ahMax = rh2vaporDens(tempMax,rhMax)    # upper bound on vapor density                  [kg/m^{-3}]                     

    # state constraints vector
    ops['x_min']   = np.array([WMin, co2Min, tempMin, rhMin])
    ops['x_max']   = np.array([WMax, co2Max, tempMax, ahMax])

    # output constraints vector
    ops['y_min']   = np.array([WMin, co2Min, tempMin, rhMin])
    ops['y_max']   = np.array([WMax, co2dens2ppm(tempMax,co2Max), tempMax, vaporDens2rh(tempMax,ahMax)])
    ops['y_min_k'] = []
    ops['y_max_k'] = []

    # control input constraints vector
    ops['u_min']   = np.array([co2SupplyMin, ventMin, heatMin])
    ops['u_max']   = np.array([co2SupplyMax, ventMax, heatMax])

    # lower and upper bound change of u 
    ops['du_max']  = np.divide(ops['u_max'],[10,10,10])   

    # initial values of the decision variables (control signals) in the optimization
    ops['u0']      = np.zeros((ops['nu'],ops['Np']))

    return ops

def LoadDisturbances(ops):
    c = 86400
    nDays = ops['L']/c      # number of days in the simulation
    D = sio.loadmat('weather/outdoorWeatherWurGlas2014.mat')
    D = D['d']
    t = D[:,0]                          # Time [days]
    dt = np.mean(np.diff(t-t[0]))*c     # sample period of data [s]
    Ns = math.ceil(nDays*c/dt)          # Number of samples we need
    N0 = 0                              # Start sample [TODO: set given starting date]
    # new sample period p times the original sample rate
    p = math.floor(1/(dt/ops['h']))

    if N0 + Ns + p*ops['Np'] > len(t):
       print(' ') 
       print('Not enough samples in the data.')
       print(' ')
    if ops['h'] < dt:
       print(' ') 
       print('Increase ops.h, sample period too small.')
       print(' ')



    # extract only data for current simulation
    t       = D[N0:N0+Ns-1 + p*ops['Np'], 0]*c       # Time [s]
    rad = D[N0:N0+Ns-1 + p*ops['Np'], 1]          # Outdoor Global radiation [W m^{-2}]
    tempOut = D[N0:N0+Ns-1 + p*ops['Np'], 2]          # Outdoor temperature [°C]
    co2Out = D[N0:N0+Ns-1 + p*ops['Np'], 5]          # Outdoor CO2 concentration [ppm]
    co2Out = co2ppm2dens(tempOut, co2Out)        # Outdoor CO2 concentration [kg/m3]
    vapOut = D[N0:N0+Ns-1 + p*ops['Np'], 3]          # Outdoor relative humidity [#]
    vapOut = rh2vaporDens(tempOut, vapOut)       # Outdoor humidity [kg/m3]

    # model: d(0) = rad, d(1) = co2Out, d(2) = tempOut, d(3) = vapOut
    # d0              = np.array([rad[0], co2Out[0], tempOut[0], vapOut[0]])

    ns              = math.ceil(len(rad)/p)
    original_d = np.array([rad, co2Out, tempOut, vapOut])

    ns              = math.ceil(len(rad)/p)
    d               = np.zeros((ops['nd'], ns))

    # t       = D[N0:N0+Ns-1, 0]*c                 # Time [s]
    time_res = np.linspace(t[0], t[-1], ns)

    for i in range(ops['nd']):
         d[i, :]  = np.interp(time_res, t, original_d[i, :])
        # interpolator = CubicSpline(t, original_d[i, :])
        # d[i, :] = interpolator(time_res)
    #    = interpolator(time_res)

    d[0, d[0, :] < 1e-6] = 0

    return d


import pandas as pd
def LoadDisturbancesMpccsv(mpc):
    c = 86400
    nDays = mpc.L/c      # number of days in the simulation
    D = pd.read_csv('weather/outdoorWeatherWurGlas2014.csv', header=None, delimiter=',', names=['time', 'Io', 'To','RH','Vo','C02ppm']).values
    # D = D['d']
    t = D[:,0]                              # Time [days]
    dt = np.mean(np.diff(t-t[0]))           # sample period of data [s]
    Ns = math.ceil(nDays*c/dt)              # Number of samples we need
    N0 = int(np.ceil(mpc.start_day*c/dt))   # Start index                             # Start sample [TODO: set given starting date]



    # new sample period p times the original sample rate
    p = math.floor(1/(dt/mpc.h))

    if N0 + Ns + p*mpc.Np > len(t):
       print(' ') 
       print('Not enough samples in the data.')
       print(' ')
    if mpc.h < dt:
       print(' ') 
       print('Increase ops.h, sample period too small.')
       print(' ')



    # extract only data for current simulation
    t       = D[N0:N0+Ns-1 + p*mpc.Np, 0]      # Time [s]
    rad = D[N0:N0+Ns-1 + p*mpc.Np, 1]          # Outdoor Global radiation [W m^{-2}]
    tempOut = D[N0:N0+Ns-1 + p*mpc.Np, 2]          # Outdoor temperature [°C]
    co2Out = D[N0:N0+Ns-1 + p*mpc.Np, 5]          # Outdoor CO2 concentration [ppm]
    co2Out = co2ppm2dens(tempOut, co2Out)        # Outdoor CO2 concentration [kg/m3]
    vapOut = D[N0:N0+Ns-1 + p*mpc.Np, 3]          # Outdoor relative humidity [#]
    vapOut = rh2vaporDens(tempOut, vapOut)       # Outdoor humidity [kg/m3]

    # model: d(0) = rad, d(1) = co2Out, d(2) = tempOut, d(3) = vapOut
    # d0              = np.array([rad[0], co2Out[0], tempOut[0], vapOut[0]])

    ns              = math.ceil(len(rad)/p)
    original_d = np.array([rad, co2Out, tempOut, vapOut])

    ns              = math.ceil(len(rad)/p)
    d               = np.zeros((mpc.nd, ns))
    

    for i in range(mpc.nd):
        d[i, :] = original_d[i, ::p]
    
    d[0, d[0, :] < 1e-6] = 0

    return d


def LoadDisturbancesMpc(mpc):
    c = 86400
    nDays = mpc.L/c      # number of days in the simulation
    D = sio.loadmat('weather/outdoorWeatherWurGlas2014.mat')
    D = D['d']
    t = D[:,0]                          # Time [days]
    dt = np.mean(np.diff(t-t[0]))*c     # sample period of data [s]
    Ns = math.ceil(nDays*c/dt)          # Number of samples we need
    N0 = int(np.ceil(mpc.start_day*c/dt))   # Start index                             # Start sample [TODO: set given starting date]

    # new sample period p times the original sample rate
    p = math.floor(1/(dt/mpc.h))

    if N0 + Ns + p*mpc.Np > len(t):
       print(' ') 
       print('Not enough samples in the data.')
       print(' ')
    if mpc.h < dt:
       print(' ') 
       print('Increase ops.h, sample period too small.')
       print(' ')



    # extract only data for current simulation
    t       = D[N0:N0+Ns-1 + p*mpc.Np, 0]*c       # Time [s]
    rad = D[N0:N0+Ns-1 + p*mpc.Np, 1]          # Outdoor Global radiation [W m^{-2}]
    tempOut = D[N0:N0+Ns-1 + p*mpc.Np, 2]          # Outdoor temperature [°C]
    co2Out = D[N0:N0+Ns-1 + p*mpc.Np, 5]          # Outdoor CO2 concentration [ppm]
    co2Out = co2ppm2dens(tempOut, co2Out)        # Outdoor CO2 concentration [kg/m3]
    vapOut = D[N0:N0+Ns-1 + p*mpc.Np, 3]          # Outdoor relative humidity [#]
    vapOut = rh2vaporDens(tempOut, vapOut)       # Outdoor humidity [kg/m3]

    # model: d(0) = rad, d(1) = co2Out, d(2) = tempOut, d(3) = vapOut
    # d0              = np.array([rad[0], co2Out[0], tempOut[0], vapOut[0]])

    ns              = math.ceil(len(rad)/p)
    original_d = np.array([rad, co2Out, tempOut, vapOut])

    ns              = math.ceil(len(rad)/p)
    d               = np.zeros((mpc.nd, ns))
    
    time_res = t[::p]  # Take every third time point

    # time_res = np.linspace(t[0], t[-1], ns)
    for i in range(mpc.nd):
        d[i, :] = original_d[i, ::p]
    # for i in range(mpc.nd):
        # d[i, :] = resample(original_d[i, :], ns)
    
    # for i in range(mpc.nd):
    #      d[i, :]  = np.interp(time_res, t, original_d[i, :])

    d[0, d[0, :] < 1e-6] = 0

    return d


def rh2vaporDens(temp,rh):
    
    # constants
    R = 8.3144598 # molar gas constant [J mol^{-1} K^{-1}]
    C2K = 273.15 # conversion from Celsius to Kelvin [K]
    Mw = 18.01528e-3 # molar mass of water [kg mol^-{1}]

    # parameters used in the conversion
    c = np.array([610.78, 238.3, 17.2694, -6140.4, 273.15, 28.916])

    satP = c[0]*np.exp(c[2]*np.divide(temp,(temp+c[1]))) 
    # Saturation vapor pressure of air in given temperature [Pa]
    
    pascals=(rh/100)*satP # Partial pressure of vapor in air [Pa]
    
    # convert to density using the ideal gas law pV=nRT => n=pV/RT 
    # so n=p/RT is the number of moles in a m^3, and Mw*n=Mw*p/(R*T) is the 
    # number of kg in a m^3, where Mw is the molar mass of water.
    
    vaporDens = np.divide(pascals*Mw,(R*(temp+c[4])))
        
    return vaporDens
   
def vaporDens2rh(temp,vaporDens):

    # constants
    R = 8.3144598 # molar gas constant [J mol^{-1} K^{-1}]
    C2K = 273.15 # conversion from Celsius to Kelvin [K]
    Mw = 18.01528e-3 # molar mass of water [kg mol^-{1}]

    # parameters used in the conversion
    c = np.array([610.78, 238.3, 17.2694, -6140.4, 273, 28.916])
    
    satP = c[0]*np.exp(c[2]*np.divide(temp,(temp+c[1]))) 
    # Saturation vapor pressure of air in given temperature [Pa]

    # convert to relative humidity using the ideal gas law pV=nRT => n=pV/RT 
    # so n=p/RT is the number of moles in a m^3, and Mw*n=Mw*p/(R*T) is the 
    # number of kg in a m^3, where Mw is the molar mass of water.    
    if isinstance(temp, float) or isinstance(temp, int) : 
        rh = min(100.0,np.divide(100*R*(temp+C2K),(Mw*satP))*vaporDens)
    else:
        rh = np.divide(100*R*(temp+C2K),(Mw*satP))*vaporDens

    return rh

def co2dens2ppm(temp, dens):
    R = 8.3144598 # molar gas constant [J mol^{-1} K^{-1}]
    C2K = 273.15 # conversion from Celsius to Kelvin [K]
    M_CO2 = 44.01e-3 # molar mass of CO2 [kg mol^-{1}]
    P = 101325 # pressure (assumed to be 1 atm) [Pa]

    ppm = 1e6*R*(temp+C2K)*dens/(P*M_CO2)
    
    return ppm

def co2ppm2dens(temp, ppm):        
    R = 8.3144598 # molar gas constant [J mol^{-1} K^{-1}]
    C2K = 273.15 # conversion from Celsius to Kelvin [K]
    M_CO2 = 44.01e-3 # molar mass of CO2 [kg mol^-{1}]
    P = 101325 # pressure (assumed to be 1 atm) [Pa]

    # number of moles n=m/M_CO2 where m is the mass [kg] and M_CO2 is the
    # molar mass [kg mol^{-1}]. So m=p*V*M_CO2*P/RT where V is 10^-6*ppm    
    co2Dens = np.divide(P*1e-6*ppm*M_CO2,(R*(temp+C2K)))

    return co2Dens


def get_parameters():
    return np.array(list(DefineParameters().values()))

def DefineParameters():
# Model parameters
    # parameter 					description 									[unit] 					nominal value
    p = {}
    p['satH2O1'] = 9348 	# 0		# saturation water vapour parameter 			[J m^{-3}] 				9348            0
    p['satH2O2'] = 17.4 	# 1      # saturation water vapour parameter 			[-] 					17.4            1
    p['satH2O3'] = 239      # 2     # saturation water vapour parameter 			[°C] 					239                 2
    p['satH2O4'] = 10998    # 3 	# saturation water vapour parameter 			[J m^{-3}] 				10998           3
    p['R'] = 8314           # 4     # ideal gas constant                            [J K^{-1} kmol^{-1}] 	8314            4
    p['T'] = 273.15 		# 5		# conversion from C to K 						[K] 					273.15        5

    p['leak'] = 7.5e-6 	# 6    # 0.75e-4		# ventilation leakage through the cover 		[m s^{-1}] 				0.75e-4     6
    p['CO2cap'] = 4.1 		# 7		# CO2 capacity of the greenhouse 				[m^3{air} m^{-2}{gh}]   4.1         7
    p['H2Ocap'] = 4.1 		# 8		# Vapor capacity of the greenhouse 				[m^3{air} m^{-2}{gh}]   4.1         8
    p['aCap'] = 3e4 		# 9		# effective heat capacity of the greenhouse air [J m^{-2}{gh} °C^{-1}]  3e4         9
    p['ventCap'] = 1290 	# 10		# heat capacity per volume of greenhouse air 	[J m^{-3}{gh} °C^{-1}]  1290        10
    p['trans_g_o'] = 6.1 	# 11		# overall heat transfer through the cover 		[W m^{-2}{gh} °C^{-1}]  6.1         11
    p['rad_o_g'] = 0.2 		# 12	    # heat load coefficient due to solar radiation 	[-] 					0.2             12

    p['alfaBeta'] = 0.544 	# 13		    # yield factor 									[-] 					0.544   13
    p['Wc_a'] = 2.65e-7     # 14            # respiration rate 								[s^{-1}] 				2.65e-7 14
    p['CO2c_a'] = 4.87e-7 	# 15		    # respiration coefficient 						[s^{-1}]  				4.87e-7 15
    p['laiW'] = 53 			# 16	        # effective canopy surface 						[m^2{leaf} kg^{-1}{dw}] 53      16
    p['photI0'] = 3.55e-9 	# 17		    # light use efficiency 							[kg{CO2} J^{-1}]  		3.55e-9 17
    p['photCO2_1'] = 5.11e-6  	# 18	    # temperature influence on photosynthesis 		[m s^{-1} °C^{-2}] 		5.11e-6 18
    p['photCO2_2'] = 2.3e-4		# 19	    # temperature influence on photosynthesis 		[m s^{-1} °C^{-1}] 		2.3e-4  19
    p['photCO2_3'] = 6.29e-4 	# 20		# temperature influence on photosynthesis 		[m s^{-1}] 				6.29e-4 20
    p['photGamma'] = 5.2e-5 	# 21		# carbon dioxide compensation point 			[kg{CO2} m^{-3}{air}] 	5.2e-5  21
    p['evap_c_a'] = 3.6e-3 		# 22	    # coefficient of leaf-air vapor flow 			[m s^{-1}] 				3.6e-3  22
    p['energyCost'] = 0.1281    # 23     # price of energy                               [€ J^{-1}]              6.35e-9 [Dfl J^{-1}] (division by 2.20371 represents currency conversion)   23
    p['co2Cost'] = 0.1906       # 24        # price of CO2                                  [€ kg^{-1}{CO2}]        42e-2 [Dfl kg^{-1}{CO2}] (division by 2.20371 represents currency conversion)   24
    p['productPrice1'] = 1.8/2.20371 # 25    # parameter for price of product                [€ m^{-2}{gh}]          1.8 [Dfl kg^{-1}{gh}] (division by 2.20371 represents currency conversion)  25
    p['productPrice2'] = 22.285125   # 26   # parameter for price of product                [€ kg^{-1}{gh} m^{-2}{gh}] 16 (division by 2.20371 represents currency conversion)  26

    return p

def ode(x, u, d, p):
    """
    Function that defines the ordinary differential equations of the model.

    Args:
        x (array-like): State variables of the system.
        u (array-like): Control inputs to the system.
        d (array-like): Disturbances affecting the system.
        p (array-like): Parameters of the model.

    Returns:
        casadi: The derivatives of the state variables.
    """
    ode = casadi.vertcat(
        p[13] * (
            (1 - np.exp(-p[16] * x[0])) * p[17] * d[0] *
            (-p[18] * x[2]**2 + p[19] * x[2] - p[20]) * (x[1] - p[21]) 
            / (p[17] * d[0] + (-p[18] * x[2]**2 + p[19] * x[2] - p[20]) * (x[1] - p[21]))
        )
        - p[14]*x[0] * 2**(0.1*x[2] - 2.5),

        1 / p[7] * (
            -((1 - np.exp(-p[16] * x[0])) * p[17] * d[0] *
            (-p[18] * x[2]**2 + p[19] * x[2] - p[20]) * (x[1] - p[21])
            / (p[17] * d[0] + (-p[18] * x[2]**2 + p[19] * x[2] - p[20]) * (x[1] - p[21])))
            + p[15] * x[0] * 2 ** (0.1 * x[2] - 2.5) + u[0]/1e6 - (u[1] / 1e3 + p[6]) * (x[1] - d[1])
        ),

        1 / p[9] * (
            u[2] - (p[10] * u[1] * 1e-3 + p[11]) * (x[2] - d[2]) + p[12] * d[0]
        ),
        1 / p[8] * (
            (1 - np.exp(-p[16] * x[0])) * p[22] * (p[0] / (p[4] * (x[2] + p[5])) *
            np.exp((p[1] * x[2]) / (x[2] + p[2])) - x[3]) - 
            (u[1]*1e-3 + p[6]) * (x[3] - d[3])
        )
    )

    return ode

def g_measure(x):
    y = casadi.vertcat(
        1e3*x[0],
        co2dens2ppm(x[2], x[1]),
        x[2],
        vaporDens2rh(x[2], x[3])
    )   
    return y

def define_model(h, xmin, xmax):    
    # Convert xmin and xmax to CasADi constants (DM) with appropriate shape
    xmin_cas = casadi.DM(xmin).reshape((-1, 1))  # Ensure it's a column vector
    xmax_cas = casadi.DM(xmax).reshape((-1, 1))  # Ensure it's a column vector

    # Define the model
    x = casadi.SX.sym('x', 4, 1) # state vector: [dw, co2In, tempIn, vapIn]
    u = casadi.SX.sym('u', 3) # control vector: [co2, vent, heat]
    d = casadi.SX.sym('d', 4) # disturbance vector: [rad, co2Out, tempOut, vapOut]
    params = casadi.SX.sym('params', 27)

    # define the ode, and measurement function
    dxdt = ode(x, u, d, params)
    y = g_measure(x)

    # generate a casadi function
    f = casadi.Function('f', [x, u, d, params], [dxdt], ['x', 'u', 'd', 'params'], ['dxdt'])
    g = casadi.Function('g', [x], [y], ['x'], ['y'])
    # expand casadi function for computational efficiency
    f.expand()
    g.expand()

    opts = {"simplify": True, "number_of_finite_elements": 4}
    input_args = casadi.vertcat(d, params)

    integrator_func = casadi.integrator('integrator', 'rk', {'x': x, 'u':u, 'p': input_args, 'ode': dxdt}, 0., h, opts)
    res = integrator_func(x0=x, u=u, p=input_args)
    # Limit the state variables between xmin and xmax
    # xnext_unbounded = res["xf"]
    # xnext_limited = casadi.fmin(casadi.fmax(xnext_unbounded, xmin_cas), xmax_cas)
    F = casadi.Function('F', [x, u, d, params], [res["xf"]], ['x', 'u', 'd', 'p'], ['xnext'])    #Discretized Function
    return F, g

def f(x, u, d, p, h):

    k1  = F(x, u, d, p)
    k2  = F(x + h/2*k1, u, d, p)
    k3  = F(x + h/2*k2, u, d, p)
    k4  = F(x + h*k3, u, d, p)
    x_new = x + h/6*(k1 + 2*k2 + 2*k3 + k4)
    return x_new



def F(x, u, d, p):
        ki =  np.array([
            p["alfaBeta"]*(
            (1-np.exp(-p["laiW"] * x[0])) * p["photI0"] * d[0] *
            (-p["photCO2_1"] * x[2]**2 + p["photCO2_2"] * x[2] - p["photCO2_3"]) * (x[1] - p["photGamma"]) 
            / (p["photI0"] * d[0] + (-p["photCO2_1"] * x[2]**2 + p["photCO2_2"] * x[2] - p["photCO2_3"]) * (x[1] - p["photGamma"])))
            - p["Wc_a"] * x[0] * 2**(0.1 * x[2] - 2.5)
            ,

            1 / p["CO2cap"] * (
            -((1 - np.exp(-p["laiW"] * x[0])) * p["photI0"] * d[0] *
            (-p["photCO2_1"] * x[2]**2 + p["photCO2_2"] * x[2] - p["photCO2_3"]) * (x[1] - p["photGamma"])
            / (p["photI0"] * d[0] + (-p["photCO2_1"] * x[2]**2 + p["photCO2_2"] * x[2] - p["photCO2_3"]) * (x[1] - p["photGamma"])))
            + p["CO2c_a"] * x[0] * 2**(0.1 * x[2] - 2.5) + u[0]/1e6 - (u[1] / 1e3 + p["leak"]) * (x[1] - d[1])
            ),

            1/p["aCap"] * (
            u[2] - (p["ventCap"] * u[1] / 1e3 + p["trans_g_o"]) * (x[2] - d[2]) + p["rad_o_g"] * d[0]
            ),

            1/p["H2Ocap"] * ((1 - np.exp(-p["laiW"] * x[0])) * p["evap_c_a"] * (p["satH2O1"]/(p["R"]*(x[2]+p["T"]))*
            np.exp(p["satH2O2"] * x[2] / (x[2] + p["satH2O3"])) - x[3]) - (u[1]/1e3 + p["leak"]) * (x[3] - d[3]))]
            )
        return ki
    
def g(x):
    y = np.array([
        1e3*x[0],
        co2dens2ppm(x[2], x[1]),
        x[2],
        vaporDens2rh(x[2], x[3])
    ])         
    return y

import matplotlib.pyplot as plt
from time import time
if __name__ == "__main__":
    ops = {}
    nDays = 1
    c = 86400
    ops['nd'] = 4
    ops["h"] = 15*60                                    # control interval (s)
    ops["L"]  = nDays*c                                 # final time simulation
    ops["t"]  = np.arange(
        0,
        ops["L"]+ops["h"],
        ops["h"]
    ) # initial time vector
    ops["N"]  = len(ops["t"])                           # number of samples in initial time vector
    ops["Np"] = 1
    xmin = np.array([0, 0, 0, 0])
    xmax = np.array([np.inf, 0.004, 40, 0.051])

    d = LoadDisturbances(ops)
    h = 900.
    model, output = define_model(h, xmin, xmax)
    p_dict = DefineParameters()
    p = get_parameters()
    x0     = np.array([0.0035, 1e-03, 15, 0.008])
    N = 96

    Xnp = np.zeros((4, N+1))
    Ynp = np.zeros((4, N))

    Xcas = np.zeros((4, N+1))
    Ycas = np.zeros((4, N))

    Xcas[:, 0] = x0
    Xnp[:, 0] = x0

    np_time = time()

    for i in range(N):
        u = np.random.rand(3)
        Xnp[:, i+1] = f(Xnp[:, i], u, d[:, i], p_dict, h)
        Ynp[:, i] = g(Xnp[:, i+1])

    cas_time = time()
    for i in range(N):
        Xcas[:, i+1] = np.array(model(Xcas[:, i], u, d[:, i], p)).flatten()
        Ycas[:, i] = np.array(output(Xcas[:, i+1])).flatten()

    print(f"np_time: {time()-np_time}")
    print(f"cas_time: {time()-cas_time}")

    fig, ax = plt.subplots(4, 1)
    for i in range(4):
        ax[i].plot(Ycas[i, :], label="Numpy")
        ax[i].plot(Ycas[i, :], linestyle='--', c="C3", label="Casadi")
    ax[i].legend()
    plt.savefig(f"numpy-casadi-y.jpg")
