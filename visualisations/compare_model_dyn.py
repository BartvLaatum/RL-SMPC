import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plot_config
from common.utils import define_model, load_env_params, get_parameters, co2ppm2dens, rh2vaporDens

dt = 1800

env_id = "LettuceGreenhouse"
env_params = load_env_params(env_id)
constraints = env_params["constraints"]

x_min = np.array(
    [
        constraints["W_min"],
        constraints["state_co2_min"],
        constraints["state_temp_min"],
        constraints["state_vp_min"]
    ],
    dtype=np.float32
)
x_max = np.array(
    [
        constraints["W_max"],
        constraints["state_co2_max"],
        constraints["state_temp_max"],
        constraints["state_vp_max"]
    ],
)


F, g = define_model(dt, x_min, x_max)

input_path = f"data/matching-salim/stochastic/mpc/mpc-noise-correction-1H-0.1.csv"
df = pd.read_csv(input_path)
df = df[df["run"] == 0]
x0 = df.iloc[0, 1:5].values
(N, _) = df.shape
x = np.zeros((N, 4))
y = np.zeros((N, 4))
d = df.iloc[:, 12:16].values

d[:, 1]  = co2ppm2dens(d[:, 2], d[:, 1])
d[:, 3]  = rh2vaporDens(d[:, 2], d[:, 3])

u = df.iloc[:, 9:12].values
p = get_parameters()

x[0, :] = x0
y[0, :] = g(x0).toarray().ravel()

for i in range(N-1):
    x[i+1, :] = F(x[i, :], u[i, :], d[i, :], p).toarray().ravel()
    y[i+1, :] = g(x[i+1, :]).toarray().ravel()

WIDTH = 175 * 0.03937
HEIGHT = WIDTH * 0.75
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(WIDTH, HEIGHT), dpi=120)

fig.suptitle(r"Updated model dynamics with $\sigma=1e-8$")
ax1.plot(y[:, 0], label='Simulated')
ax1.plot(df["y_0"], label='Actual')
ax1.legend()
ax1.set_title('y0')

ax2.plot(y[:, 1], label='Simulated')
ax2.plot(df["y_1"], label='Actual')
ax2.set_title('y1')

ax3.plot(y[:, 2], label='Simulated')
ax3.plot(df["y_2"], label='Actual')
ax3.set_title('y2')

ax4.plot(y[:, 3], label='Simulated')
ax4.plot(df["y_3"], label='Actual')
ax4.set_title('y3')

plt.tight_layout()

fig.savefig("Y-sigma1e-8.png")
