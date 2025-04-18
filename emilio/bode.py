import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.optimize import curve_fit
import mplhep as hep
from cycler import cycler
import matplotlib.colors as colors
import multiprocessing.pool
import pandas as pd

inputname = 'dati.txt'

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
s_A=A*np.sqrt((0.041*v_fs/v_in)**2+(0.04*v_fs/v_out)**2) #3% per il cambio sonda ?????
#primi 6 dati e ultimi 9 fatti con due sonde

v_out_bode=v_out[35:43]
v_in_bode=v_in[35:43]
f_bode=f[35:43]
v_errL_bode=V_errL[35:43]

y=np.log10(v_out_bode/v_in_bode)
sy=np.sqrt(np.power((v_errL_bode/v_out_bode), 2) + np.power((1*0.41/(10*v_in_bode) + (1.2/100*v_out_bode)),2))
x=np.log10(f_bode)

def fitadueparam(x, m, q):
   return( m*x + q)

def fita1param(x, q):
   return(-x + q)

popt2, ppcov2= curve_fit(fitadueparam, x, y, sigma=sy)
popt1, ppcov1= curve_fit(fita1param, x, y, sigma=sy)

m2, q2 = popt2
err_param=np.sqrt(np.diag(ppcov2))
sm2, sq2 = err_param
print(f'm2={m2}+-{sm2}, q2={q2}+-{sq2}')

q1 = popt1
err_param=np.sqrt(np.diag(ppcov1))
sq1 = err_param
print(f'q1={q1}+-{sq1}')

x_fit=np.linspace(min(x), max(x), 8)
y_fit2 = fitadueparam(x_fit, *popt2)
y_fit1 = fita1param(x_fit, *popt1)

residui2 = y - fitadueparam(x, *popt2)
chi2 = np.sum(residui2**2/(sy**2))
print(f'chi a due param={chi2}')

residui1 = y - fita1param(x, *popt1)
chi1 = np.sum(residui1**2/(sy**2))
print(f'chi a un param={chi1}')


fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [3, 1]}, figsize=(8,6))
#ax1.scatter(x,y1)
#ax1.errorbar(x,y1, yerr=sy, fmt='o', capsize=5)
#ax1.plot(x_fit, y_fit1)
ax1.scatter(x, y)
ax1.errorbar(x, y, yerr=sy, fmt='o', capsize=5)
ax1.plot(x_fit, y_fit1, label='Modello a 1 param', color='blue')
ax1.fill_between(x_fit, y_fit1 + sq1, y_fit1 - sq1, color='blue', alpha=0.2, label='Incertezza')
ax1.plot(x_fit, y_fit2, label='Modello a 2 param', color='green')
ax1.fill_between(x_fit, y_fit2 + sm2, y_fit2 - sm2, color='green', alpha=0.2, label='Incertezza')
#plt.plot(x_fit, y_fit20, c='red', label='-20')

ax2.errorbar(x, residui1, yerr=sy, fmt='o', label='Residui a 1 param')
ax2.errorbar(x, residui2, yerr=sy, fmt='o', label='Residui a 2 param')
ax2.axhline(0, c='red', linestyle='-')

plt.tight_layout(pad=0.2)
plt.show()

f_taglio=np.power(10, -q2/m2)
sft = 2.30258509299*10**(-q2/m2)*np.sqrt((sq2/m2)**2 + (q2/(m2**2)*sm2)**2)

print(f'f_taglio={f_taglio}+-{sft}')


'''
f_taglio=np.power(10, -b/a)
sft=np.power(10, -b/a)*np.log(10)*np.sqrt((sa/a)**2+ (b/(a**2)*sb)**2)
print(f'f_taglio={f_taglio}+-{sft}')

f_taglio1=np.power(10, -b1/a1)
sft1=np.power(10, -b1/a1)*np.log(10)*np.sqrt((sa1/a1)**2+ (b1/(a1**2)*sb1)**2)
print(f'f_taglio1={f_taglio1}+-{sft1}')
'''
