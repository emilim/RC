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
s_A=A*np.sqrt((0.041/v_in)**2+(0.041*v_fs/v_out)**2 + (3/100*v_in)**2 + (3/100*v_out)**2) #3% per il cambio sonda ?????
#primi 6 dati e ultimi 9 fatti con due sonde

v_out_bode=v_out[34:49]
v_in_bode=v_in[34:49]
f_bode=f[34:49]
v_errL_bode=V_errL[34:49]
print(f"freq={f_bode}")

y=np.log10(v_out_bode/v_in_bode)
sy=abs(y)*np.sqrt(np.pow((v_errL_bode/v_out_bode), 2) + np.pow((1*0.41/(10*v_in_bode)),2) + np.pow((1.2/100*v_in_bode),2) + np.pow((1.2/100*v_out_bode),2))
x=np.log10(f_bode)

print(f'y={y}')
print(f'x={x}')
print(f'sy={sy}')

def fitlineare(x, a, b):
   return( a*x + b)

def fit1(x, q):
   return(-x + q)

popt, ppcov= curve_fit(fitlineare, x, y, sigma=sy)
popt1, ppcov1= curve_fit(fit1, x, y, sigma=sy)

q=popt1
err_q=np.sqrt(np.diag(ppcov1))
sq=err_q
print(f'q={q}+-{sq}')

a, b=popt
err_param=np.sqrt(np.diag(ppcov))
sa, sb= err_param
print(f'a={a}+-{sa}, b={b}+-{sb}')


x_fit=np.linspace(min(x), max(x), 13)
y_fit=fitlineare(x_fit, *popt)
residui=y-fitlineare(x, *popt)
chi=np.sum(residui**2/(sy**2))
print(f'chi 2 param={chi}')

y_fit20=fit1(x_fit, popt1)
residui1=y-fit1(x, popt1)
chi1=np.sum(residui1**2/sy**2)
print(f'chi 1 param={chi1}')

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [3, 1]}, figsize=(8,6))
ax1.scatter(x,y)
ax1.errorbar(x,y, yerr=sy, fmt='o', capsize=5)
ax1.plot(x_fit, y_fit, label='fit lineare:y=-0.91x+3.34')
ax1.set_ylabel('log10(A)')
ax2.set_xlabel('log10(f) [Hz]')
ax1.plot(x_fit, y_fit20, c='green', label='fit 1 parametro: y=-x+3.7')
ax1.grid(True)
ax2.grid(True)

f_taglio=np.pow(10, -b/a)
sft=np.pow(10,-b/a)*np.sqrt((sb/a)**2+ (b/(a**2)*sa)**2)

f_taglio_=np.pow(10, q)
sft_=np.pow(10, q)*np.log(10)*sq
print(f'f_taglio={f_taglio}+-{sft}')
print(f'f_taglio 1 param={f_taglio_}+-{sft_}')

ax1.scatter(np.log10(f_taglio), 0, c='orange', label='(2 param) ft=(4600+-400)Hz')
#ax1.errorbar(np.log10(f_taglio), 0, xerr=np.log10(sft))
ax2.scatter(x, residui)
ax2.errorbar(x, residui, yerr=sy, c='orange', fmt='o')
ax1.scatter(np.log10(f_taglio_), 0, c='green', label='(1 param) ft=(5300+-200)Hz')
#ax1.errorbar(np.log10(f_taglio_), 0, xerr=np.log10(sft_))
ax2.scatter(x, residui1)
ax2.errorbar(x, residui1, yerr=sy, c='green', fmt='o')
ax2.axhline(0, c='red', linestyle='-')
ax1.legend()

plt.tight_layout(pad=0.2)
plt.legend()
plt.show()

s=abs(v_out/v_in)*np.sqrt(np.pow((0.041/v_out), 2) + np.pow((1*0.41/(10*v_in)),2) + np.pow((1.2/100*v_in),2) + np.pow((1.2/100*v_out),2))
plt.scatter(np.log10(f), np.log10(v_out/v_in), label='log10(A)')
plt.errorbar(np.log10(f), np.log10(v_out/v_in), yerr=s, fmt='o')
plt.plot(x_fit, y_fit, c='orange', label='fit lineare')
plt.xlabel('log10(f)[Hz]')
plt.ylabel('log10(A)')
plt.scatter(np.log10(877000), np.log10(0.096/7.6), c='red', label='dati anomali' )
plt.scatter(np.log10(577000), np.log10(0.122/7.6), c='red')
plt.scatter(np.log10(1177000), np.log10(0.078/7.6), c='red')
plt.legend()
plt.show()