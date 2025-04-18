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

n = 9
f = np.array(df['f_gen_KHz'].values * 1e3)[:-n]  # f in Hz
v_in = np.array(df['Vin'].values)[:-n]
v_out = np.array(df['Vout'].values)[:-n]
A = v_out / v_in  
phi = 2*np.pi*f*np.array(df['Fase_us'].values[:-n]*1e-6)/(np.pi/2.) # norm to 1
v_fs = np.array(df['Scala_V'].values)[:-n]
phi_fs = 2*np.pi*f*np.array(df['Scala_us'].values[:-n]*1e-6)/(np.pi/2.)

Vin_errL = v_fs[0]/10*0.41
V_errL = v_fs/10*0.41
phi_errL = phi_fs/10*0.41*np.sqrt(2)
A_err = A*np.sqrt((0.04*v_fs/v_in)**2 + (0.04*v_fs/v_out)**2)

A_2 = A**(-2)
f_2 = f**(2)
err_A_2 = 2*A_2*A_err/A

# fit lineare
parameters, pcov = curve_fit(lambda x, m, q: m*x + q, f_2, A_2, sigma=err_A_2)
parameters2, pcov2 = curve_fit(lambda x, m: m*x + 1, f_2, A_2, sigma=err_A_2)

m = parameters[0]
q = parameters[1]
err = np.sqrt(np.diag(pcov))
err_m = err[0]
err_q = err[1]

m_2 = parameters2[0]
err_2 = np.sqrt(np.diag(pcov2))
err_m2 = err[0]

f_t = 1/np.sqrt(m)
err_f_t = 0.5*m**(-1.5)*err[0]

f_t_2 = 1/np.sqrt(m_2)
err_f_t_2 = 0.5*m_2**(-1.5)*err_m2

print(f"m={m} +/- {err_m}")
print(f"q={q} +/- {err_q}")
print(f"Compat con 1: ", (q-1)/err_q)
print(f"ft={f_t} +/- {err_f_t}")

print(f"m_2={m_2} +/- {err_m2}")
print(f"ft_2={f_t_2} +/- {err_f_t_2}")

x_vis = np.logspace(np.log10(min(f_2)), np.log10(max(f_2)), 10000)
y_vis = m*x_vis + q
y_vis2 = m_2*x_vis + 1
residui = A_2 - (m*f_2 + q)
residui2 = A_2 - (m_2*f_2 + 1)

chi = np.sum((residui/err_A_2)**2)
chi2 = np.sum((residui2/err_A_2)**2)
print(f"chi A={chi}")
print(f"chi A2={chi2}")


fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [3, 1]}, figsize=(8,6))

ax1.errorbar(f_2, A_2, yerr=err_A_2, fmt='o',ms=4, color='blue', label='$A^{-2}$ ')
ax1.plot(x_vis, y_vis, label='Fit a due parametri: $f_t=$ {:.0f} $\pm$ {:.0f}'.format(f_t, err_f_t), color='red')
ax1.plot(x_vis, y_vis2, label='Fit a un parametro: $f_t=$ {:.0f} $\pm$ {:.0f}'.format(f_t_2, err_f_t_2), color='green')
ax1.fill_between(x_vis, y_vis + err[0], y_vis - err[0], color='red', alpha=0.2)
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xlabel(r'$\nu^2$ [Hz]')
ax1.set_ylabel('$A^{-2}$ [V/V]')

ax2.errorbar(f_2, residui, yerr=err_A_2, fmt='o', color='blue', label='Residui fit a 2 parametri')
#ax2.errorbar(f_2, residui2, yerr=err_A_2, fmt='o', color='green', label='Residui a 1 param')
ax2.axhline(0, c='red', linestyle='-')
ax2.grid(True)
ax1.legend()
plt.legend()
plt.tight_layout(pad=0.2)
plt.show()
