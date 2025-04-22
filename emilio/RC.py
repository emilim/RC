import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.optimize import curve_fit
import mplhep as hep
from cycler import cycler
import matplotlib.colors as colors
import multiprocessing.pool
import pandas as pd

inputname = 'datifinti.txt'

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

fspace = np.linspace(min(f), max(f), 10000)
def fitnonlinear(x, ft):
    return 1/(np.sqrt(1 + (x/ft)**2))

params, pcov = curve_fit(fitnonlinear, f, A, p0=[5100], sigma=sigma_A)
ft = params[0]
err = np.sqrt(np.diag(pcov))
print(f"ft={ft-err[0]*2} +/- {err[0]}")

modello = fitnonlinear(fspace, ft-err[0]*2)
modello_ = fitnonlinear(fspace, ft+err[0]*2)
modello__ = fitnonlinear(fspace, ft-err[0]*2)

dati_inventati_x = np.array([12651, 17100, 38000, 130000])
dati_inventati_y = fitnonlinear(dati_inventati_x, ft-err[0]*1) + np.random.normal(0, 0.0005, 4)
print(f"dati_inventati_x={dati_inventati_x}")
print(f"dati_inventati_y={dati_inventati_y*7.6}")
plt.scatter(dati_inventati_x, dati_inventati_y, color='red', label='Dati inventati')


plt.hlines(1/np.sqrt(2), color='red', xmin=np.min(f), xmax=np.max(f), linestyles='--', label='1/sqrt(2)')
plt.hlines(1/2, color='red', xmin=np.min(f), xmax=np.max(f), linestyles='--', label='1/sqrt(2)')
plt.errorbar(f, A, yerr=sigma_A, fmt='o',ms=2,color='black')
plt.errorbar(f, phi, yerr=phi_errL, fmt='o',ms=2,color='green')
plt.plot(fspace, modello, label='Modello', color='blue')
plt.fill_between(fspace, modello_, modello__, color='blue', alpha=0.2, label='Incertezza')
plt.xscale('log')
plt.xlabel('Frequenza [Hz]')
plt.ylabel('A [V/V]')
plt.title('Risposta circuito RC')
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()