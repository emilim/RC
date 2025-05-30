import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.optimize import curve_fit
import mplhep as hep
from cycler import cycler
import matplotlib.colors as colors
import multiprocessing.pool
import pandas as pd

inputname = 'passabasso.txt'

# f_0/deltaf = Q = sqrt(L/C)/R. R = R+R_L+R_G

# 

def fitf_C(x, Av, B, C):
    omega = 2.0 * np.pi * x
    fitval = Av / np.sqrt((1-omega**2/B**2)**2+1/C**2*omega**2/B**2)
    return fitval

df = pd.read_csv(inputname, delimiter='\t', decimal='.')

f = np.array(df['f_(kHz)'].values * 1e3)  # f in Hz
v_in = np.array(df['Vin_(V)'].values)
v_out = np.array(df['Vout_(V)'].values)
A = v_out / v_in  
#phi = 2*np.pi*f*np.array(df['Dt_(ns)'].values[:]*1e-9)
v_fs_in = np.array(df['scala_Vin_(mV)'].values*10**-3)[:]
v_fs_out = np.array(df['scala_Vout_(V)'].values)[:]

letturaV = 0.1*0.41
errscalaV = 0.03*0.41

eVo = np.sqrt((letturaV * v_fs_out)**2 + (errscalaV * v_out)**2)
eVin = np.sqrt((letturaV * v_fs_in)**2 + (errscalaV * v_in)**2)

sigma_A = A * np.sqrt((eVo / v_out)**2 + (eVin / v_in)**2)

f_0 = 235000
fdelta = max(A)/np.sqrt(2)

Avinit= 0.95
Binit =  2.0 * np.pi *233000.  # Hz
Cinit = 10. # Hz

popt, pcov = curve_fit(fitf_C, f, A, p0=[Avinit, Binit, Cinit], sigma=sigma_A, absolute_sigma=True)
ampiezza, f0_stima, Qvalue = popt
f0_stima = f0_stima / (2.0 * np.pi) 
perr = np.sqrt(np.diag(pcov))
ampiezza_err, f0_stima_err, Qvalue_err = perr
print( ' ampiezza = {a:.3f} +/- {b:.3f} \n f_0 = {c:.1f} +/- {d:.1f} kHz \n Q-valore = {e:.1f} +/- {f:.1f}'.format(a=ampiezza, b=ampiezza_err,c=f0_stima/(1000), d=f0_stima_err/1000, e=Qvalue, f=Qvalue_err))

fit_f = np.linspace(min(f), max(f), 1000)
plt.errorbar(f, A, yerr=sigma_A, fmt='o',ms=5, color='blue')
plt.xscale('log')
plt.yscale('log')
plt.plot(fit_f, fitf_C(fit_f, *popt), color='red', lw=1.5, label='fit')
plt.vlines(f0_stima, 0, max(A), colors='red', linestyles='--', lw=1.5)
plt.hlines(fdelta, min(f), max(f), colors='green', linestyles='--', lw=1.5)

plt.show()


plt.errorbar(f, v_in, yerr=sigma_A, fmt='o',ms=5, color='blue', label='Vin')
plt.errorbar(f, v_out, yerr=sigma_A, fmt='o',ms=5, color='red', label='Vout')
plt.legend()
plt.show()
