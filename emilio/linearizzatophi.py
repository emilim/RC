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

n = -9
f = np.array(df['f_gen_KHz'].values * 1e3)[:n]  # f in Hz
v_in = np.array(df['Vin'].values)[:n]
v_out = np.array(df['Vout'].values)[:n]
A = v_out / v_in  
phi = 2*np.pi*f*np.array(df['Fase_us'].values[:n]*1e-6)/(np.pi/2.) # norm to 1
v_fs = np.array(df['Scala_V'].values)[:n]
phi_fs = 2*np.pi*f*np.array(df['Scala_us'].values[:n]*1e-6)/(np.pi/2.)

Vin_errL = v_fs[0]/10*0.41
V_errL = v_fs/10*0.41
phi_errL = phi_fs/10*0.41*np.sqrt(2)
A_err = A*np.sqrt((0.04*v_fs/v_in)**2 + (0.04*v_fs/v_out)**2)

tan_dphi = np.tan(phi*(np.pi/2.))
err_dphi = (np.pi/2.)*phi_errL/np.cos(phi*(np.pi/2.))**2

# fit lineare
parameters, pcov = curve_fit(lambda x, m: m*x, f, tan_dphi, sigma=err_dphi)
parameters2, pcov2 = curve_fit(lambda x, m, q: m*x + q, f, tan_dphi, sigma=err_dphi)

m = parameters[0]
err = np.sqrt(np.diag(pcov))
err_m = err[0]

f_t = 1/m
err_f_t = m**(-2)*err_m

print(f"m={m} +/- {err_m}")
print(f"ft={f_t} +/- {err_f_t}")

m_2 = parameters2[0]
q_2 = parameters2[1]
err_2 = np.sqrt(np.diag(pcov2))
err_m2 = err_2[0]
err_q2 = err_2[1]
f_t_2 = 1/m_2
err_f_t_2 = m_2**(-2)*err_m2

print(f"m_2={m_2} +/- {err_m2}")
print(f"q_2={q_2} +/- {err_q2}")
print("compat con 0: ", q_2/err_q2) 
print(f"ft_2={f_t_2} +/- {err_f_t_2}")



x_vis = np.linspace(min(f), max(f), 100)

y_vis = m*x_vis 
residui = tan_dphi - (m*f)
chi = np.sum((residui/err_dphi)**2)
print(f"chi fit a un param={chi}")

y_vis2 = m_2*x_vis + q_2
residui2 = tan_dphi - (m_2*f + q_2)
chi2 = np.sum((residui2/err_dphi)**2)
print(f"chi fit a due param={chi2}")


fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [3, 1]}, figsize=(8,6))

ax1.errorbar(f, tan_dphi, yerr=err_dphi, fmt='o',ms=6,color='blue')
ax1.plot(x_vis, y_vis, label='Fit a un parametro: $f_t=$ {:.0f} $\pm$ {:.0f}'.format(f_t, err_f_t), color='green')

ax1.plot(x_vis, y_vis2, label='Fit a due parametri: $f_t=$ {:.0f} $\pm$ {:.0f}'.format(f_t_2, err_f_t_2), color='red')

ax1.set_xlabel(r'$\nu^2$ [Hz]')
ax1.set_ylabel(r'$\tan(\Delta\phi)$ [rad]')

ax2.errorbar(f, residui, yerr=err_dphi, fmt='o', color='green', label='Residui fit a un parametro')
ax2.errorbar(f, residui2, yerr=err_dphi, fmt='o', color='red', label='Residui fit a 2 parametri')
ax2.axhline(0, c='orange', linestyle='-')
ax2.grid(True)
ax2.legend()

ax1.legend()
plt.legend()

plt.tight_layout(pad=0.2)
plt.show()
