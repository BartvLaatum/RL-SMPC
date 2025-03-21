from typing import Any, Dict
import yaml
import math

import casadi 
import numpy as np
import pandas as pd

def load_disturbances(
    file_name: str,
    L: float,
    start_day: int, 
    h: float,
    Np: int,
    nd: int
    ) -> np.ndarray:

    """
    Load and process disturbance data from a CSV file for a simulation.
    Arguments:
        file_name (str): Name of the CSV file containing the disturbance data.
        L (float): Total simulation time in seconds.
        start_day (int): Starting day of the simulation.
        h (float): Desired sample period for the simulation.
        Np (int): Prediction horizon.
        nd (int): Number of disturbance variables.
    Returns:
        numpy.ndarray: Processed disturbance data with shape (nd, ns), where ns is the number of samples.
    """

    c = 86400
    nDays = L/c      # number of days in the simulation
    D = pd.read_csv(f"weather/{file_name}", header=None, delimiter=",", names=["time", "Io", "To","RH","Vo","C02ppm"]).values
    # D = D["d"]
    t = D[:,0]                              # Time [days]
    dt = np.mean(np.diff(t-t[0]))           # sample period of data [s]
    Ns = math.ceil(nDays*c/dt)              # Number of samples we need
    N0 = int(np.ceil(start_day*c/dt))   # Start index                             # Start sample [TODO: set given starting date]

    # new sample period p times the original sample rate
    p = math.floor(1/(dt/h))

    if N0 + Ns + p*Np > len(t):
       print(" ") 
       print("Not enough samples in the data.")
       print(" ")
    if h < dt:
       print(" ") 
       print("Increase ops.h, sample period too small.")
       print(" ")

    # extract only data for current simulation
    t       = D[N0:N0+Ns-1 + p*Np, 0]      # Time [s]
    rad = D[N0:N0+Ns-1 + p*Np, 1]          # Outdoor Global radiation [W m^{-2}]
    tempOut = D[N0:N0+Ns-1 + p*Np, 2]          # Outdoor temperature [°C]
    co2Out = D[N0:N0+Ns-1 + p*Np, 5]          # Outdoor CO2 concentration [ppm]
    co2Out = co2ppm2dens(tempOut, co2Out)        # Outdoor CO2 concentration [kg/m3]
    vapOut = D[N0:N0+Ns-1 + p*Np, 3]          # Outdoor relative humidity [#]
    vapOut = rh2vaporDens(tempOut, vapOut)       # Outdoor humidity [kg/m3]

    # model: d(0) = rad, d(1) = co2Out, d(2) = tempOut, d(3) = vapOut
    # d0              = np.array([rad[0], co2Out[0], tempOut[0], vapOut[0]])

    ns              = math.ceil(len(rad)/p)
    original_d = np.array([rad, co2Out, tempOut, vapOut])

    ns              = math.ceil(len(rad)/p)
    d               = np.zeros((nd, ns))
    

    for i in range(nd):
        d[i, :] = original_d[i, ::p]
    
    d[0, d[0, :] < 1e-6] = 0

    return d

def transform_disturbances(d: np.ndarray) -> np.ndarray:
    """
    Transforms the disturbances array by converting CO2 density to ppm and 
    vapor density to relative humidity.

    Parameters
    d (np.ndarray)
        An array (nd, 1) of disturbances where:
        - nd is the number of disturbance variables.
        - ns is the number of samples.
        - d[0] is the radiation.
        - d[1] is the CO2 density to be converted to ppm.
        - d[2] is the temperature used in the conversion calculations.
        - d[3] is the vapor density to be converted to relative humidity.

    Returns (np.ndarray)
        The modified disturbances array with CO2 density converted to ppm 
        and vapor density converted to relative humidity.
    """
    d[1] = co2dens2ppm(d[2], d[1])
    d[3] = vaporDens2rh(d[2], d[3])
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
    rh = np.divide(100*R*(temp+C2K),(Mw*satP))*vaporDens

    return rh

def vaporDens2rh_hat(temp,vaporDens, p):

    # parameters used in the conversion
    # c = np.array( [610.78, 238.3, 17.2694, -6140.4, 273, 28.916])
    p0 = 610.78
    Mw = 18.01528e-3

    # Saturation vapor pressure of air in given temperature [Pa]
    satP = p0 * np.exp( p[29] *    (temp / (temp+p[28]))) 
    # satP = p_0*np.exp(params[26]*temp/   (temp+params[27]))

    # convert to relative humidity using the ideal gas law pV=nRT => n=pV/RT 
    # so n=p/RT is the number of moles in a m^3, and Mw*n=Mw*p/(R*T) is the 
    # number of kg in a m^3, where Mw is the molar mass of water.    
    rh = (100*p[23] * (temp+p[5]) / (Mw*satP))*vaporDens
#   rh = (100*params[11]*(temp+params[12])/(Mw*satP))*vaporDens

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

def co2dens2ppm_hat(temp, dens, p):
    # R = 8.3144598 # molar gas constant [J mol^{-1} K^{-1}]
    # C2K = 273.15 # conversion from Celsius to Kelvin [K]
    # M_CO2 = 44.01e-3 # molar mass of CO2 [kg mol^-{1}]
    # P = 101325 # pressure (assumed to be 1 atm) [Pa]

    ppm = 1e6*p[23]*(temp+p[5])*dens/(p[25]*p[24])
    return ppm

def get_parameters():
    return np.array(list(DefineParameters().values()))

def DefineParameters():
# Model parameters
    # parameter                         description                                     [unit]                  nominal value
    p = {}
    p["satH2O1"] = 9348         # 0     saturation water vapour parameter               [J m^{-3}] 				9348            0
    p["satH2O2"] = 17.4         # 1     saturation water vapour parameter               [-] 					17.4            1
    p["satH2O3"] = 239          # 2     saturation water vapour parameter               [°C] 					239             2
    p["satH2O4"] = 10998        # 3     saturation water vapour parameter               [J m^{-3}] 				10998           3
    p["R"] = 8314               # 4     ideal gas constant                              [J K^{-1} kmol^{-1}] 	8314            4
    p["T"] = 273.15             # 5     conversion from C to K                          [K]                     273.15          5

    p["leak"] = 0.75e-5         # 6     ventilation leakage through the cover           [m s^{-1}]              0.75e-4         6
    p["CO2cap"] = 4.1           # 7     CO2 capacity of the greenhouse                  [m^3{air} m^{-2}{gh}]   4.1             7
    p["H2Ocap"] = 4.1           # 8		Vapor capacity of the greenhouse                [m^3{air} m^{-2}{gh}]   4.1             8
    p["aCap"] = 3e4             # 9     effective heat capacity of the greenhouse air   [J m^{-2}{gh} °C^{-1}]  3e4             9
    p["ventCap"] = 1290         # 10    heat capacity per volume of greenhouse air 	    [J m^{-3}{gh} °C^{-1}]  1290            10
    p["trans_g_o"] = 6.1        # 11    overall heat transfer through the cover         [W m^{-2}{gh} °C^{-1}]  6.1             11
    p["rad_o_g"] = 0.2          # 12    heat load coefficient due to solar radiation    [-]                     0.2             12

    p["alfaBeta"] = 0.544       # 13    yield factor                                    [-]                     0.544           13
    p["Wc_a"] = 2.65e-7         # 14    respiration rate                                [s^{-1}]                2.65e-7         14
    p["CO2c_a"] = 4.87e-7       # 15    respiration coefficient                         [s^{-1}]                4.87e-7         15
    p["laiW"] = 53              # 16    effective canopy surface                        [m^2{leaf} kg^{-1}{dw}] 53              16
    p["photI0"] = 3.55e-9       # 17    light use efficiency                            [kg{CO2} J^{-1}]  		3.55e-9         17
    p["photCO2_1"] = 5.11e-6    # 18    temperature influence on photosynthesis         [m s^{-1} °C^{-2}]      5.11e-6         18
    p["photCO2_2"] = 2.3e-4		# 19    temperature influence on photosynthesis         [m s^{-1} °C^{-1}]      2.3e-4          19
    p["photCO2_3"] = 6.29e-4 	# 20    temperature influence on photosynthesis         [m s^{-1}]              6.29e-4         20
    p["photGamma"] = 5.2e-5 	# 21    carbon dioxide compensation point               [kg{CO2} m^{-3}{air}]   5.2e-5          21
    p["evap_c_a"] = 3.6e-3 		# 22    coefficient of leaf-air vapor flow              [m s^{-1}]              3.6e-3          22
    
    p["R2"] = 8.3144598         # 23    molar gas constant [J mol^{-1} K^{-1}]
    p["M_CO2"] = 44.01e-3       # 24    molar mass of CO2 [kg mol^-{1}]
    p["P"] = 101325             # 25    pressure (assumed to be 1 atm) [Pa]
    p["MW"] = 18.01528e-3       # 26    molar mass of water
    p["c0"] = 610.78            # 27    conversion factor for RH
    p["c1"] = 238.3             # 28    conversion factor for RH
    p["c2"] = 17.2694           # 29    conversion factor for RH

    p["energyCost"] = 0.1281    # 30     price of energy                          [€ J^{-1}]              6.35e-9 [Dfl J^{-1}] (division by 2.20371 represents currency conversion)   23
    p["co2Cost"] = 0.1906       # 31     price of CO2                                  [€ kg^{-1}{CO2}]        42e-2 [Dfl kg^{-1}{CO2}] (division by 2.20371 represents currency conversion)   24
    p["productPrice1"] = 1.8/2.20371    # 32    parameter for price of product                [€ m^{-2}{gh}]          1.8 [Dfl kg^{-1}{gh}] (division by 2.20371 represents currency conversion)  25
    p["productPrice2"] = 22.285125      # 33    parameter for price of product                [€ kg^{-1}{gh} m^{-2}{gh}] 16 (division by 2.20371 represents currency conversion)  26
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

        (1 / p[7]) * (
            -((1 - np.exp(-p[16] * x[0])) * (p[17] * d[0] *
            (-p[18] * x[2]**2 + p[19] * x[2] - p[20]) * (x[1] - p[21]))
            / (p[17] * d[0] + (-p[18] * x[2]**2 + p[19] * x[2] - p[20]) * (x[1] - p[21])))
            + p[15] * x[0] * 2 ** (0.1 * x[2] - 2.5) + u[0]/1e6 - (u[1] / 1e3 + p[6]) * (x[1] - d[1])
        ),

        (1 / p[9]) * (
            u[2] - (p[10] * u[1] * 1e-3 + p[11]) * (x[2] - d[2]) + p[12] * d[0]
        ),

        (1 / p[8]) * (
            (1 - np.exp(-p[16] * x[0])) * p[22] * (p[0] / (p[4] * (x[2] + p[5])) *
            np.exp((p[1] * x[2]) / (x[2] + p[2])) - x[3]) - 
            (u[1]*1e-3 + p[6]) * (x[3] - d[3])
        )
    )

    return ode

def g_measure(x):
    y = casadi.vertcat(
        x[0],
        co2dens2ppm(x[2], x[1]),
        x[2],
        casadi.mmin(casadi.horzcat(100, vaporDens2rh(x[2], x[3])))
    )   
    return y

def g_hat_measure(x, p):
    y = casadi.vertcat(
        x[0],
        co2dens2ppm_hat(x[2], x[1], p),
        x[2],
        casadi.mmin(casadi.horzcat(100, vaporDens2rh_hat(x[2], x[3], p)))
    )
    return y

def define_model(dt, xmin, xmax):
    # Convert xmin and xmax to CasADi constants (DM) with appropriate shape
    xmin_cas = casadi.DM(xmin).reshape((-1, 1))  # Ensure it"s a column vector
    xmax_cas = casadi.DM(xmax).reshape((-1, 1))  # Ensure it"s a column vector

    # Define the model
    x = casadi.SX.sym("x", 4, 1) # state vector: [dw, co2In, tempIn, vapIn]
    u = casadi.SX.sym("u", 3) # control vector: [co2, vent, heat]
    d = casadi.SX.sym("d", 4) # disturbance vector: [rad, co2Out, tempOut, vapOut]
    params = casadi.SX.sym("params", 34)

    # define the ode, and measurement function
    dxdt = ode(x, u, d, params)
    y = g_measure(x)

    # generate a casadi function
    f = casadi.Function("f", [x, u, d, params], [dxdt], ["x", "u", "d", "params"], ["dxdt"])
    g = casadi.Function("g", [x], [y], ["x"], ["y"])
    # expand casadi function for computational efficiency
    f.expand()
    g.expand()

    opts = {"simplify": True, "number_of_finite_elements": 4}
    input_args = casadi.vertcat(d, params)

    integrator_func = casadi.integrator("integrator", "rk", {"x": x, "u":u, "p": input_args, "ode": dxdt}, 0., dt, opts)
    res = integrator_func(x0=x, u=u, p=input_args)
    # Limit the state variables between xmin and xmax
    xnext_unbounded = res["xf"]
    xnext_limited = casadi.fmin(casadi.fmax(xnext_unbounded, xmin_cas), xmax_cas)
    # F = casadi.Function("F", [x, u, d, params], [res["xf"]], ["x", "u", "d", "p"], ["xnext"])    #Discretized Function
    F = casadi.Function("F", [x, u, d, params], [xnext_limited], ["x", "u", "d", "p"], ["xnext"])    #Discretized Function
    return F, g

def compute_economic_reward(delta_dw, params, dt, u):
    """
    Economic cost function. Can be used either by MPC or RL.

    Arguments:
        delta_dw (float): Change in dry weight.
        params (array-like): Array of parameters.
        h (float): Time step size.
        u (array-like): Control inputs.

    Returns:
        float: Economic reward.
    """
    # print(params[33], params[32])
    return params[33] * (delta_dw) \
        - 1e-6 * dt * params[31] * u[0] \
        - dt / 3600 * 1e-3 * params[30]  * u[2]

def load_env_params(env_id: str) -> Dict[str, Any]:
    """
    Load environment parameters from a yaml file.
    Arguments:
        file_name (str): Name of the MAT file containing the environment parameters.
    Returns:
        Dict[str, Any]: Dictionary of environment parameters.
    """    
    # load the config file
    with open(f"configs/envs/{env_id}.yml", "r") as file:
        env_params = yaml.safe_load(file)

    return env_params

def load_mpc_params(env_id: str) -> Dict[str, Any]:
    """
    Load MPC parameters from a yaml file.
    Arguments:
        file_name (str): Name of the MAT file containing the MPC parameters.
    Returns:
        Dict[str, Any]: Dictionary of MPC parameters.
    """    
    # load mpc parameters
    with open("configs/models/mpc.yml", "r") as file:
        mpc_params = yaml.safe_load(file)

    return mpc_params[env_id]
