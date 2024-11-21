import pandas as pd
import numpy as np
from utils import LoadDisturbances, DefineParameters, f, g
from matplotlib import pyplot as plt
from plots import plot_controls, plot_disturbances, plot_measurements

# Perform simulation
def simulate(x0, control_inputs, outdoor_weather, p, ops):
    x = x0
    Y = []
    for i, u in enumerate(control_inputs):
        d = outdoor_weather[i]
        h = None  # Replace with actual h if needed
        x = f(x, u, d, p, ops["h"])
        y = g(x, u, d, p, ops["h"])
        Y.append(y)
    return np.array(Y)


# Load control inputs from the CSV file
control_inputs = pd.read_csv('uopt_matlab.csv', header=None).values.T

ops = {}
nDays = 1
c = 86400
ops["h"] = 900                                      # control interval (s)
ops["L"]  = nDays*c                                 # final time simulation
ops["t"]  = np.arange(0,ops["L"]+ops["h"],ops["h"]) # initial time vector
ops["N"]  = len(ops["t"])                           # number of samples in initial time vector
ops["Np"] = 1
ops["nd"] = 4


# Load outdoor weather disturbances
outdoor_weather = LoadDisturbances(ops).T

# Set parameters
p = DefineParameters()

# Define initial condition
x0 = np.array([0.0035, 1e-03, 15, 0.008])  # Replace with the actual initial condition

# Run the simulation
y = simulate(x0, control_inputs, outdoor_weather, p, ops)
print("Final state:", y[-1])
ops["nx"] = 4
ops["nu"] = 3

plot_controls(ops, control_inputs, c, title='Controls-matlab')