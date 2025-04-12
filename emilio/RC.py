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

plt.hlines(1/np.sqrt(2), color='red', xmin=np.min(f), xmax=np.max(f), linestyles='--', label='1/sqrt(2)')
plt.errorbar(f, A, yerr=A_err, fmt='o',ms=2,color='black')
plt.errorbar(f, phi, yerr=phi_errL, fmt='o',ms=2,color='green')
plt.xscale('log')
plt.xlabel('Frequenza [Hz]')
plt.ylabel('A [V/V]')
plt.title('Risposta circuito RC')
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()