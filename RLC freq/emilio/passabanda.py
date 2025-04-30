import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.optimize import curve_fit
import mplhep as hep
from cycler import cycler
import matplotlib.colors as colors
import multiprocessing.pool
import pandas as pd

inputname = 'passabanda.txt'

# f_0/deltaf = Q = sqrt(L/C)/R. R = R+R_L+R_G

# 

def fitf_R(x, A_p, B, C):
    omega = 2.0 * np.pi * x 
    fitval = A_p / np.sqrt(1+C**2*(omega**2/B**2-B**2/omega**2)**2)
    return fitval

df = pd.read_csv(inputname, delimiter='\t', decimal='.')

f = np.array(df['f_(kHz)'].values * 1e3)  # f in Hz
v_in = np.array(df['Vin_(V)'].values)
v_out = np.array(df['Vout_(mV)'].values*10**-3)
A = v_out / v_in  
phi = 2*np.pi*f*np.array(df['Dt_(ns)'].values[:]*1e-9)
v_fs_in = np.array(df['scala_Vin_(mV)'].values*10**-3)[:]
v_fs_out = np.array(df['scala_Vout_(mV)'].values*10**-3)[:]
phi_fs = 2*np.pi*f*np.array(df['scala_t_(ns)'].values[:]*1e-9)

letturaV = 0.1*0.41
errscalaV = 0.03*0.41

eVo = np.sqrt((letturaV * v_fs_out)**2 + (errscalaV * v_out)**2)
eVin = np.sqrt((letturaV * v_fs_in)**2 + (errscalaV * v_in)**2)

sigma_A = A * np.sqrt((eVo / v_out)**2 + (eVin / v_in)**2)
phi_errL = 0
f_0 = 235000
fdelta = max(A)/np.sqrt(2)

Ainit= 0.95
Binit =  2.0 * np.pi *233000.  # Hz
Cinit = 10. # Hz

popt, pcov = curve_fit(fitf_R, f, A, p0=[Ainit, Binit, Cinit], sigma=sigma_A, absolute_sigma=True)
ampiezza, f0_stima, Qvalue = popt
f0_stima = f0_stima / (2.0 * np.pi)
perr = np.sqrt(np.diag(pcov))
print( ' ampiezza = {a:.3f} +/- {b:.3f} \n f_0 = {c:.1f} +/- {d:.1f} kHz \n Q-valore = {e:.1f} +/- {f:.1f}'.format(a=ampiezza, b=perr[0],c=f0_stima/(1000),d=perr[1]/1000,e=Qvalue,f=perr[2]))

fig, ax = plt.subplots(2, 1, figsize=(5, 4),sharex=True, constrained_layout = True, height_ratios=[2, 1])
#plt.xscale('log')
#plt.yscale('log')
ax[0].errorbar(f, A, yerr=sigma_A, fmt='o',ms=5, color='blue')
ax[0].plot(f, fitf_R(f, *popt), color='red', lw=1.5, label='fit')
ax[0].vlines(f0_stima, 0, max(A), colors='red', linestyles='--', lw=1.5)
ax[0].hlines(fdelta, min(f), max(f), colors='green', linestyles='--', lw=1.5)
#plt.errorbar(f, phi, yerr=phi_errL, fmt='o',ms=2,color='green')


residui = A - fitf_R(f, *popt)
ax[1].errorbar(f, residui, yerr=sigma_A, fmt='o',ms=5, color='blue')
plt.show()

plt.errorbar(f, v_in, yerr=sigma_A, fmt='o',ms=5, color='blue', label='Vin')
plt.errorbar(f, v_out, yerr=sigma_A, fmt='o',ms=5, color='red', label='Vout')
plt.legend()
plt.show()

def fitphi_R(x, B, C):
    omega = 2.0 * np.pi * x 
    fitval = np.arctan(C*(-B/omega + omega/B))
    return fitval

poptphi, pcovphi = curve_fit(fitphi_R, f, phi, p0=[Binit, Cinit], sigma=sigma_A, absolute_sigma=True)


perrphi = np.sqrt(np.diag(pcovphi))

f0phi = poptphi[0] / (2.0 * np.pi)
err_f0phi = perrphi[0]
Qvaluephi = poptphi[1]
err_Qvaluephi = perrphi[1]
print(" ")
print( ' f0 = {a:.3f} +/- {b:.3f} kHz \n Q-valore = {c:.1f} +/- {d:.2f}'.format(a=f0phi/1000, b=err_f0phi/1000,c=Qvaluephi,d=err_Qvaluephi))


plt.errorbar(f, phi, yerr=sigma_A, fmt='o',ms=5, color='blue')
plt.plot(f, fitphi_R(f, *poptphi), color='red', lw=1.5, label='fit')
plt.show()
