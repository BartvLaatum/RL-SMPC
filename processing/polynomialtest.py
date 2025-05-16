import numpy as np
import casadi as ca

from common.utils import vaporDens2rh, rh2vaporDens, co2ppm2dens, co2dens2ppm

def poly(temp):
    p18 = 5.11e-6    # 18    temperature influence on photosynthesis         [m s^{-1} °C^{-2}]      5.11e-6         18
    p19 = 2.3e-4		# 19    temperature influence on photosynthesis         [m s^{-1} °C^{-1}]      2.3e-4          19
    p20 = 6.29e-4 	# 20    temperature influence on photosynthesis         [m s^{-1}]              6.29e-4         20

    return -p18 * ca.power(temp, 2) + p19 * temp - p20

x = np.linspace(5, 40, 100)
y = poly(x)
import matplotlib.pyplot as plt
# plt.plot(x, y)
plt.xlabel("Temperature [°C]")
# plt.title("Polynomial: $-p_{1, 8}x_{temp}(t)^2+p_{1,6}x_{temp}(t)-p_{1,7}$")
# plt.hlines(0, xmin=5, xmax=40, color="grey", linestyle="--")
# plt.savefig("temp-poly.png")

print(vaporDens2rh(5, 0.001))
print(vaporDens2rh(40, 0.001))

rhmin =  12
y = rh2vaporDens(x, rhmin)

# xmin = 0.001
# y = vaporDens2rh(x, xmin)
# plt.plot(x, y)
# plt.xlabel("Temperature")
# plt.ylabel("RH")
# plt.savefig("rh.png")
# plt.show()

xmin = 0.0002
y = co2dens2ppm(x, xmin)
plt.plot(x, y)
# plt.xlabel("Temperature")
plt.ylabel("CO2 (ppm)")
plt.savefig("co2.png")
plt.show()