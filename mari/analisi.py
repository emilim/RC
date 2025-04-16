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
# phi_fs = 2*np.pi*f*np.array(data[5])*1e-6/(np.pi/2.)
Vin_errL = v_fs[0]/10*0.41
V_errL = v_fs/10*0.41
s_A=np.sqrt((0.041*v_fs/v_in)**2+(0.04*v_fs/v_out)**2)
print(f"phi={phi}")

plt.hlines(1/np.sqrt(2), color='red', xmin=np.min(f), xmax=np.max(f), linestyles='--', label='1/sqrt(2)')
plt.scatter(f, A, c='black', s=10, label='Data', zorder=2)
plt.xscale('log')
plt.xlabel('Frequenza [Hz]')
plt.ylabel('A [V/V]')
plt.title('Risposta circuito RC')
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()

#fit lineare

inizio= 12
fine= 27

f_lin=f[inizio: fine]
f_lin1=f[inizio: (fine-2)]
phi_lin=phi[inizio: (fine-2)]
A_lin=A[inizio: fine]
s_A_lin=s_A[inizio: fine]
s_phi_lin=5*1e-6*2*np.pi*f_lin1/10*0.41*np.sqrt(2)/(np.pi/2) 
print(f"s_phi_lin={s_phi_lin}")


def fitlin(x, m, q):
    return m*x + q

popt1, ppcov1=curve_fit(fitlin, f_lin1, phi_lin, sigma=s_phi_lin)
popt, ppcov=curve_fit(fitlin, f_lin, A_lin, sigma=s_A_lin)
f_fit=np.linspace(min(f_lin), max(f_lin), 15)
f_fit1=np.linspace(min(f_lin1), max(f_lin1), 13)
A_fit=fitlin(f_fit, *popt)
phi_fit=fitlin(f_fit1, *popt1)

residui=A_lin-A_fit
residui1=phi_lin-phi_fit
err_param=np.sqrt(np.diag(ppcov))
err_param1=np.sqrt(np.diag(ppcov1))
chi=np.sum((residui/s_A_lin)**2)
chi1=np.sum((residui1/s_phi_lin)**2)

print(f"chi A={chi}")
print(f"chi phi={chi1}")

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [3, 1]}, figsize=(8,6))
ax1.scatter(f_lin, A_lin, label='A=V_out/V_in')
ax1.errorbar(f_lin, A_lin, yerr=s_A_lin, fmt='o')
ax1.axhline(1/np.sqrt(2), color='green', linestyle='--', label='A=1/sqrt(2)') 
ax1.set_ylabel('A')
ax2.set_xlabel('f [Hz]')
ax1.legend()
ax1.plot(f_fit, A_fit, label='fit lineare' )
ax1.grid(True)

ax2.errorbar(f_lin, residui, yerr=s_A_lin, fmt='o')
ax2.axhline(0, c='red', linestyle='-')
ax2.grid(True)


m, q= popt
sm, sq = err_param
f_tlin= (1/np.sqrt(2) - q)/m
s_ft=np.sqrt(sq**2/(m**2) + ((1/np.sqrt(2)-q)**2/(np.pow(m, 4))*sm**2))
print(f'fit lineare: y={m}x+{q}, sm={sm} e sq={sq}')
print(f'freq_taglio={f_tlin} +- {s_ft}')

ax1.scatter(f_tlin, 1/np.sqrt(2), c='green', label='frequenza di taglio: ft=(5180+-100)Hz' )
ax1.errorbar(f_tlin, 1/np.sqrt(2), xerr=s_ft, capsize=5)
ax1.legend()

plt.tight_layout(pad=0.2)
plt.show()

#stima grezza della tempi div
delta_T_grezzo = np.array(df['Fase_us'].values)/3.75 #microsecondi
print(f"scala fasi:{delta_T_grezzo}")
delta_T=np.trunc(delta_T_grezzo).astype(int)
print(f"scala fasi:{delta_T}")

fig, (bx1, bx2) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [3, 1]}, figsize=(8,6))
bx1.scatter(f_lin1, phi_lin, label='phi=2pi*f*deltaT')
bx1.errorbar(f_lin1, phi_lin, yerr=s_phi_lin, fmt='o')
bx1.axhline(0.5, color='green', linestyle='--', label='phi=0.5') 
bx1.set_ylabel('phi')
bx2.set_xlabel('f [Hz]')
bx1.legend()
bx1.plot(f_fit1, phi_fit, label='fit lineare' )
bx1.grid(True)

bx2.errorbar(f_lin1, residui1, yerr=s_phi_lin, fmt='o')
bx2.axhline(0, c='red', linestyle='-')
bx2.grid(True)


m1, q1= popt1
sm1, sq1 = err_param1
f_tlin1= (0.5 - q1)/m1
s_ft1=np.sqrt(sq1**2/(m1**2) + ((0.5-q1)**2/(np.pow(m1, 4))*sm1**2))
print(f'fit lineare phi: y={m1}x+{q1}, sm={sm1} e sq={sq1}')
print(f'freq_taglio={f_tlin1} +- {s_ft1}')

bx1.scatter(f_tlin1, 0.5 , c='green', label='frequenza di taglio: ft=(5500+-300)Hz' )
bx1.errorbar(f_tlin1, 0.5, xerr=s_ft1, capsize= True)
bx1.legend()

plt.tight_layout(pad=0.2)
plt.show()

fig, (cx1, cx2) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [3, 1]}, figsize=(8,6))
cx1.scatter(f_lin, A_lin, c="blue", label='A=V_out/V_in')
cx1.errorbar(f_lin, A_lin, yerr=s_A_lin, c="blue", fmt='o')
cx1.axhline(1/np.sqrt(2), color='green', linestyle='--', label='A=1/sqrt(2)') 
cx1.set_ylabel('A')
cx2.set_xlabel('f [Hz]')
cx1.legend()
cx1.plot(f_fit, A_fit, c="blue", label='fit lineare' )
cx1.grid(True)

cx2.errorbar(f_lin, residui, yerr=s_A_lin, fmt='o')
cx2.axhline(0, c='yellow', linestyle='-')
cx2.grid(True)

cx1.scatter(f_lin1, phi_lin, c="orange", label='phi=2pi*f*deltaT')
cx1.errorbar(f_lin1, phi_lin, c="orange", yerr=s_phi_lin, fmt='o')
cx1.axhline(0.5, color='red', linestyle='--', label='phi=0.5') 
cx1.set_ylabel('phi')
cx2.set_xlabel('f [Hz]')
cx1.legend()
cx1.plot(f_fit1, phi_fit, c="orange", label='fit lineare' )
cx1.grid(True)

cx2.errorbar(f_lin1, residui1, yerr=s_phi_lin, fmt='o')
cx2.axhline(0, c='yellow', linestyle='-')
cx2.grid(True)

cx1.scatter(f_tlin, 1/np.sqrt(2), c='green', label='da A: ft=(5180+-100)Hz' )
cx1.errorbar(f_tlin, 1/np.sqrt(2), c='green', xerr=s_ft, capsize= 5)
cx1.scatter(f_tlin1, 0.5 , c='red', label='da phi: ft=(5500+-300)Hz' )
cx1.errorbar(f_tlin1, 0.5, c='red', xerr=s_ft1, capsize= 5)
cx1.legend()

plt.tight_layout(pad=0.2)
plt.show()