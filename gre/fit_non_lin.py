import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


filename = 'RCF_PB.txt'

df = pd.read_csv(filename, delimiter='\t', decimal=',')

frequenze = df['f_gen_KHz'].values * 1e3 #Hz

#Tensioni
Vin = df['Vin'].values
Vout = df['Vout'].values
#Attenuazione
A = Vout / Vin

#Fase normalizzata tra 0 e 1
fase_us = df['Fase_us'].values * 1e-6  #sec
fase_norm = 2 * np.pi * frequenze * fase_us / (np.pi / 2.)

#Scale
scala_V = df['Scala_V'].values
scala_us = df['Scala_us'].values * 1e-6

#ERRORI
Vin_err = scala_V[0] / 10 * 0.41
Vout_err = scala_V / 10 * 0.41
fase_err = 2 * np.pi * frequenze * scala_us / 10 * 0.41 * np.sqrt(2) / (np.pi / 2)
#sigma_A = A * np.sqrt((0.04 * scala_V / Vin)**2 + (0.04 * scala_V / Vout)**2)
#sigma_A = A * np.sqrt((0.041 / Vin)**2 + (0.041 * scala_V / Vout)**2 + (1.2/100 * Vin)**2 + (1.2/100 * Vout)**2)

low_freq = frequenze < 2100           
high_freq = frequenze > 10000          
mid_freq = ~(low_freq | high_freq)  
sigma_A = np.zeros_like(A)

sigma_A_ext = A * np.sqrt((0.041 / Vin)**2 + (0.041 * scala_V / Vout)**2 + (1.2/100 * Vin)**2 + (1.2/100 * Vout)**2)
sigma_A_mid = A * np.sqrt((0.041 / Vin)**2 + (0.041 * scala_V / Vout)**2)

sigma_A[low_freq] = sigma_A_ext[low_freq]
sigma_A[high_freq] = sigma_A_ext[high_freq]
sigma_A[mid_freq] = sigma_A_mid[mid_freq]


#FIT NON LINEARE
f_fitspace = np.linspace(min(frequenze), max(frequenze), 10000)
def fit_attenuazione(f, ft):
    return 1 / np.sqrt(1 + (f / ft)**2)

#def fit_fase(f, ft_phi):
#    return -f / ft_phi
def fit_fase(f, ft_phi):
    return np.arctan(f / ft_phi) / (np.pi / 2)

parametri, pcov = curve_fit(fit_attenuazione, frequenze, A, sigma=sigma_A, absolute_sigma=True, p0=[5100])
ft_fit = parametri[0]
ft_err = np.sqrt(np.diag(pcov))[0]
print(f"Frequenza di taglio ft = {ft_fit:.2f} ± {ft_err:.2f} Hz")

parametri_phi, pcov_phi = curve_fit(fit_fase, frequenze, fase_norm, sigma=fase_err, absolute_sigma=True, p0=[1e5])
ft_phi = parametri_phi[0]
ft_phi_err = np.sqrt(np.diag(pcov_phi))[0]
print(f"Frequenza di taglio (fase) ft_phi = {ft_phi:.2f} ± {ft_phi_err:.2f} Hz")

#MODELLO FITTATO
modello_A = fit_attenuazione(f_fitspace, ft_fit)
modello_A_upper = fit_attenuazione(f_fitspace, ft_fit + ft_err)
modello_A_lower = fit_attenuazione(f_fitspace, ft_fit - ft_err)

modello_phi = fit_fase(f_fitspace, ft_phi)
modello_phi_upper = fit_fase(f_fitspace, ft_phi + ft_phi_err)
modello_phi_lower = fit_fase(f_fitspace, ft_phi - ft_phi_err)

#GRAFICO
plt.figure(figsize=(10, 6))
plt.axhline(1 / np.sqrt(2), color='red', linestyle='--', label='1/√2')
plt.axhline(0.5, color='brown', linestyle='--', label='π/4')
plt.errorbar(frequenze, A, yerr=sigma_A, fmt='o', ms=3, color='purple', label='Attenuazione A')
plt.plot(f_fitspace, modello_A, color='orange', label=(f"Fit A: ft= ({ft_fit:.2f} ± {ft_err:.2f}) Hz"))
plt.fill_between(f_fitspace, modello_A_lower, modello_A_upper, color='orange', alpha=0.2, label='Incertezza fit A')

plt.errorbar(frequenze, fase_norm, yerr=fase_err, fmt='o', ms=3, color='green', label='Fase φ normalizzata')
plt.plot(f_fitspace, modello_phi, color='blue', label=(f"Fit φ: ft_phi= ({ft_phi:.2f} ± {ft_phi_err:.2f}) Hz"))
plt.fill_between(f_fitspace, modello_phi_lower, modello_phi_upper, color='blue', alpha=0.2, label='Incertezza fit φ')

plt.xscale('log')
plt.xlabel('log10(f) [Hz]')
plt.ylabel('A [V/V] con φ-->1')
#plt.title('Fit non lineare attenuazione e fase circuito RC')
plt.grid(True, which='both', linestyle='--', alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()


#RESIDUI
residui = A - fit_attenuazione(frequenze, ft_fit)
residui_phi = fase_norm - fit_fase(frequenze, ft_phi)

plt.figure(figsize=(10, 4))
plt.axhline(0, color='gray', linestyle='--')
plt.errorbar(frequenze, residui, yerr=sigma_A, fmt='o', ms=3, color='purple', label='Residui attenuazione')
plt.errorbar(frequenze, residui_phi, yerr=fase_err, fmt='o', ms=3, color='green', label='Residui fase')
plt.xscale('log')
plt.xlabel('Frequenza [Hz]')
plt.ylabel('Residui')
plt.title('Residui dei fit')
plt.grid(True, which='both', linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()
for f, phi in zip(frequenze, fase_norm):
    print(f"{f:.0f} Hz --> fase_norm = {phi:.2f}")