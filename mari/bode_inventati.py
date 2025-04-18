import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.optimize import curve_fit
import mplhep as hep
from cycler import cycler
import matplotlib.colors as colors
import multiprocessing.pool
import pandas as pd

inputname = 'dati_inventati.txt'

df = pd.read_csv(inputname, delimiter='\t', decimal=',')

f = np.array(df['f_gen_KHz'].values * 1e3)  # f in Hz
v_in = np.array(df['Vin'].values)
v_out = np.array(df['Vout'].values)
A = np.array(v_out / v_in)  
phi = 2*np.pi*f*np.array(df['Fase_us'].values*1e-6)/(np.pi/2.) # norm to 1
v_fs = np.array(df['Scala_V'].values)
#phi_fs = 2*np.pi*f*np.array(df['Scala_us'].values)*1e-6/(np.pi/2.)
Vin_errL = v_fs[0]/10*0.41
V_errL = v_fs/10*0.41
s_A=np.sqrt((0.041*v_fs/v_in)**2+(0.04*v_fs/v_out)**2) #3% per il cambio sonda ?????
#primi 6 dati e ultimi 9 fatti con due sonde

v_out_bode=v_out[35:45]
v_in_bode=v_in[35:45]
f_bode=f[35:45]
v_errL_bode=V_errL[35:45]
print(f"freq={f_bode}")

y=np.log10(v_out_bode/v_in_bode)
sy=np.sqrt(np.pow((v_errL_bode/v_out_bode), 2) + np.pow((1*0.41/(10*v_in_bode)),2) + np.pow((1.2/100*v_out_bode),2))
x=np.log10(f_bode)

print(f'y={y}')
print(f'x={x}')
print(f'sy={sy}')

def fitlineare(x, a, b):
   return( a*x + b)

def fit20(x, q):
   return(-x + q)

popt, ppcov= curve_fit(fitlineare, x, y, sigma=sy)
popt20, ppcov20= curve_fit(fit20, x, y, sigma=sy)

q=popt20

a, b=popt
err_param=np.sqrt(np.diag(ppcov))
sa, sb= err_param
print(f'a={a}+-{sa}, b={b}+-{sb}')


x_fit=np.linspace(min(x), max(x), 10)
y_fit=fitlineare(x_fit, *popt)
residui=y-fitlineare(x, *popt)
chi=np.sum(residui**2/(sy**2))
print(f'chi={chi}')

y_fit20=fit20(x_fit, popt20)

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [3, 1]}, figsize=(8,6))
ax1.scatter(x,y)
ax1.errorbar(x,y, yerr=sy, fmt='o', capsize=5)
ax1.plot(x_fit, y_fit)
ax1.plot(x_fit, y_fit20, c='red', label='-20')

ax2.scatter(x, residui)
ax2.errorbar(x, residui, yerr=sy, fmt='o')
ax2.axhline(0, c='red', linestyle='-')

plt.tight_layout(pad=0.2)
plt.show()

f_taglio=np.pow(10, -b/a)
sft=np.pow(10, -b/a)*np.log(10)*np.sqrt((sb/a)**2+ (b/(a**2)*sa)**2)
f_taglio_=np.pow(10, q)
print(f'f_taglio={f_taglio}+-{sft}')
print(f'f_taglio={f_taglio_}')


plt.scatter(np.log10(f), np.log10(v_out/v_in))
plt.show()