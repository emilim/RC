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
inputname_i = 'dati_inventati.txt'

df = pd.read_csv(inputname, delimiter='\t', decimal=',')
df_i = pd.read_csv(inputname_i, delimiter='\t', decimal=',')

f = np.array(df['f_gen_KHz'].values * 1e3)  # f in Hz
v_in = np.array(df['Vin'].values)
v_out = np.array(df['Vout'].values)
A = np.array(v_out / v_in)  
phi = 2*np.pi*f*np.array(df['Fase_us'].values*1e-6)/(np.pi/2.) # norm to 1
v_fs = np.array(df['Scala_V'].values)
#phi_fs = 2*np.pi*f*np.array(df['Scala_us'].values)*1e-6/(np.pi/2.)
Vin_errL = v_fs[0]/10*0.41
V_errL = v_fs/10*0.41
s_A=np.sqrt((0.041*v_fs/v_in)**2+(0.041*v_fs/v_out)**2 + (3/100*v_in)**2 + (3/100*v_out)**2) #3% per il cambio sonda ?????
#primi 6 dati e ultimi 9 fatti con due sonde

v_out_bode=v_out[35:43]
v_in_bode=v_in[35:43]
f_bode=f[35:43]
v_errL_bode=V_errL[35:43]
print(f"freq={f_bode}")

y=np.log10(v_out_bode/v_in_bode)
y1=[-2.33674555, -2.39794001, -2.43365556, -2.51532561]#[-1.5797836,  -1.79445376, -1.89854236, -1.98871899]# -2.33674555, -2.39794001, -2.43365556, -2.51532561]
sy=np.sqrt(np.pow((v_errL_bode/v_out_bode), 2) + np.pow((1*0.41/(10*v_in_bode)),2) + np.pow((1.2/100*v_out_bode),2) + np.pow((1.2/100*v_in_bode),2) + np.pow((1.2/100*v_out_bode),2))
x=np.log10(f_bode)
x1=[6.19783169, 6.27346427, 6.33785843, 6.39392601]
sy1=[0.02404532, 0.02751029, 0.02978035, 0.03575524]

print(f'y={y}')
print(f'x={x}')
print(f'sy={sy}')

def fitlineare(x, a, b):
   return( a*x + b)

def fit20(x, q):
   return(-x + q)

popt, ppcov= curve_fit(fitlineare, x, y, sigma=sy)
popt20, ppcov20= curve_fit(fit20, x, y, sigma=sy)
popt1, ppcov1= curve_fit(fitlineare, x1, y1, sigma=sy1)

q=popt20

a, b=popt
err_param=np.sqrt(np.diag(ppcov))
sa, sb= err_param
print(f'a={a}+-{sa}, b={b}+-{sb}')

a1, b1=popt1
err_param1=np.sqrt(np.diag(ppcov1))
sa1, sb1= err_param1

x_fit=np.linspace(min(x), max(x), 8)
y_fit=fitlineare(x_fit, *popt)
residui=y-fitlineare(x, *popt)
chi=np.sum(residui**2/(sy**2))
print(f'chi={chi}')

x_fit1=np.linspace(min(x1), max(x1), 4)
y_fit1=fitlineare(x_fit1, *popt1)
#residui1=y1-fitlineare(x1, *popt1)
#chi1=np.sum(residui1**2/(sy1**2))
#print(f'chi1={chi1}')

y_fit20=fit20(x_fit, popt20)

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [3, 1]}, figsize=(8,6))
#ax1.scatter(x1,y1)
#ax1.errorbar(x1,y1, yerr=sy1, fmt='o', capsize=5)
#ax1.plot(x_fit1, y_fit1)
ax1.scatter(x,y)
ax1.errorbar(x,y, yerr=sy, fmt='o', capsize=5)
ax1.plot(x_fit, y_fit)
ax1.plot(x_fit, y_fit20, c='red', label='-20')

ax2.scatter(x, residui)
ax2.errorbar(x, residui, yerr=sy, fmt='o')
ax2.axhline(0, c='red', linestyle='-')

#ax2.scatter(x1, residui1)
#ax2.errorbar(x1, residui1, yerr=sy1, fmt='o')
#ax2.axhline(0, c='red', linestyle='-')

plt.tight_layout(pad=0.2)
plt.show()

f_taglio=np.pow(10, -b/a)
sft=np.pow(10, -b/a)*np.log(10)*np.sqrt((sb/a)**2+ (b/(a**2)*sa)**2)
f_taglio_=np.pow(10, q)
print(f'f_taglio={f_taglio}+-{sft}')
print(f'f_taglio={f_taglio_}')


f_taglio1=np.pow(10, -b1/a1)
sft1=np.pow(10, -b1/a1)*np.log(10)*np.sqrt((sb1/a1)**2+ (b1/(a1**2)*sa1)**2)
print(f'f_taglio1={f_taglio1}+-{sft1}')

plt.scatter(np.log10(f), np.log10(v_out/v_in))
plt.show()