import matplotlib.pyplot as plt

# Dati (ultimi due valori convertiti in kHz)
methods = [
    'metrix', 'A, loc', 'phi, loc', 
    'Bode 1 param', 'Bode 2 param', 
    'A, lin 2 param', 'phi, lin 2 param', 'A, lin 1 param', 'phi, lin 1 param', 
    'A, nl', 'phi, nl'
]

values = [
    5.52, 5.2, 5.5, 
    5.3, 4.6, 
    5.205, 5.297, 5.173, 5.490,
    5.190, 5.500
]

errors = [
    0.03, 0.1, 0.3, 
    0.2, 0.4, 
    0.009, 0.004, 0.009, 0.040,
    20 / 1000, 20 / 1000
]

x_medio1 = 5.275

# Creazione del grafico con barre di errore
fig, ax = plt.subplots()
x = range(len(methods))
ax.errorbar(x, values, yerr=errors, fmt='o', capsize=5)
ax.errorbar(4, 4.6, yerr=0.4, fmt='o', capsize=5, color='red')
ax.axhline(y=x_medio1, color='green', linestyle='--', label='Media 5.287 ± 0.003 kHz')
ax.set_xticks(x)
ax.set_xticklabels(methods, rotation=45, ha='right')
ax.set_ylabel('Frequenza di taglio (kHz)')
ax.set_title('Confronto delle frequenze di taglio')
ax.legend()
plt.tight_layout()
plt.show()
