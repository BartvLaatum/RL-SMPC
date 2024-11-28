from utils import define_model, get_parameters
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    df = pd.read_csv("data/mpc/murray.csv")
    F, g = define_model(h=900)
    p = get_parameters()
    u = df[["u_{i}".format(i=i) for i in range(3)]].values.T
    d = df[["d_{i}".format(i=i) for i in range(4)]].values.T
    y = np.zeros((4, len(df)))
    x = np.zeros((4, len(df) + 1))
    x[:,0] = np.array([0.0035, 1e-03, 15, 0.008])


    for i in range(len(df)):
        x[:,i+1] = F(x[:,i], u[:,i], d[:,i], p).toarray().ravel()
        y[:,i] = g(x[:,i]).toarray().ravel()

    fig, ax = plt.subplots(4, 1, figsize=(10, 12))
    for i in range(4):
        ax[i].plot(df['time'].iloc[:-1], y[i, 1:])
        ax[i].plot(df['time'], df["y_{i}".format(i=i)])
    plt.savefig("y.png")

    fig, ax = plt.subplots(3, 1, figsize=(10, 12))
    for i in range(3):
        ax[i].step(df['time'], u[i])
    plt.savefig("u.png")
