import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.optimize import curve_fit
import mplhep as hep
from cycler import cycler
import matplotlib.colors as colors
import multiprocessing.pool
import pandas as pd


x = np.array([5.064, 5.164])
y = np.array([5.4, 5.36])

# fit
def fit_func(x, a, b):
    return a * x + b

fit_params, covariance = curve_fit(fit_func, x, y)
a, b = fit_params

# extrapolate where y = 5.37
ft_extrapolate = (5.37 - b) / a
ft_unc = np.abs(5.164 - ft_extrapolate)/np.sqrt(3)
print(f"Extrapolated ft value where V = 5.37: {ft_extrapolate}")
print(f"Uncertainty in ft: {ft_unc}")


plt.scatter(x, y, label='Data', color='black')
plt.plot(x, fit_func(x, a, b), label='Fit', color='red')
plt.vlines(ft_extrapolate, 5.36, 5.40, color='blue', linestyle='--', label='Extrapolated line')
plt.vlines(ft_extrapolate+ft_unc, 5.36, 5.40, color='blue', linestyle='--', label='Extrapolated line')
plt.vlines(ft_extrapolate-ft_unc, 5.36, 5.40, color='blue', linestyle='--', label='Extrapolated line')
plt.xlabel('X-axis label')
plt.ylabel('Y-axis label')

plt.title('Linear Fit Example')
plt.legend()
plt.grid()
plt.show()