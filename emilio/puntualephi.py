import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.optimize import curve_fit
import mplhep as hep
from cycler import cycler
import matplotlib.colors as colors
import multiprocessing.pool
import pandas as pd

inputname = 'RCF_PB.txt'

df = pd.read_csv(inputname, delimiter='\t', decimal=',')

f = np.array(df['f_gen_KHz'].values * 1e3)  # f in Hz
v_in = np.array(df['Vin'].values)
v_out = np.array(df['Vout'].values)
A = v_out / v_in  
phi = 2*np.pi*f*np.array(df['Fase_us'].values*1e-6)/(np.pi/2.) # norm to 1
v_fs = np.array(df['Scala_V'].values)
phi_fs = 2*np.pi*f*np.array(df['Scala_us'].values*1e-6)/(np.pi/2.)

Vin_errL = v_fs[0]/10*0.41
V_errL = v_fs/10*0.41
phi_errL = phi_fs/10*0.41*np.sqrt(2)
A_err = np.sqrt((V_errL)**2 + (v_out*0.03*0.41)**2)

sigma_A = A*np.sqrt((0.04*v_fs/v_in)**2 + (0.04*v_fs/v_out)**2)

x = np.array([v_out[19], v_out[20]])
y = np.array([phi[19]*90, phi[20]*90])
def fit_func(x, a, b):
    return a * x + b
fit_params, covariance = curve_fit(fit_func, x, y)
a, b = fit_params
ft_ext = (45-b)/a
ft_unc = np.abs(v_out[19] - ft_ext)/np.sqrt(3)

print(f"Extrapolated ft value where V = 45: {ft_ext}")
print(f"Uncertainty in ft: {ft_unc}")

plt.scatter(x, y, label='Data', color='black')
plt.plot(x, fit_func(x, a, b), label='Fit', color='red')
plt.vlines(ft_ext, phi[19]*90, phi[20]*90, color='blue', linestyle='--', label='Extrapolated line')
plt.vlines(ft_ext+ft_unc, phi[19]*90, phi[20]*90, color='blue', linestyle='--')
plt.vlines(ft_ext-ft_unc, phi[19]*90, phi[20]*90, color='blue', linestyle='--')
plt.xlabel('X-axis label')
plt.ylabel('Y-axis label')
plt.title('Linear Fit Example')
plt.grid()
plt.legend()
plt.show()