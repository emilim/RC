import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.optimize import curve_fit
import mplhep as hep
from cycler import cycler
import matplotlib.colors as colors
import multiprocessing.pool


DEB = False



## PARAMATERS

# Number of bins for parameter scan
NI = 20
NJ = 20
NK = 20

# Input file name
# file = 'RLC_Cres'   # seleziono per fit su C
file = 'capiRes'   # seleziono per fit su R
inputname = file+'.txt'

# Frequency limits for the fit function (in kHz)
frfit0 = 10.0
frfit1 = 100.0

# Initial parameter values
Ainit= 0.3
Binit =  2.0 * np.pi *233000.  # Hz
Cinit = 20. # Hz

# Assumed reading errors
letturaV = 0.1*0.41
errscalaV = 0.03*0.41


#### LOAD DATA
    

# Read data from the input file
data = np.loadtxt(inputname)
fr = data[:, 0] #frequenze
Vin = data[:, 1] #Vin
Vo = data[:, 3]*10**-3 #Vout
Vdiv_in = data[:, 2]*10**-3 #divisioni-FS del Vin
VdivR = data[:, 4]*10**-3 #divisioni-FS del Vout

# Number of points to fit
# va a contare il numero di frequenze nel vettore fr che siano maggiori di zero
N = len(fr[fr > 0])

# Calculate errors on x and y
eVo = np.sqrt((letturaV * VdivR)**2 + (errscalaV * Vo)**2)
eVin = np.sqrt((letturaV * Vdiv_in)**2 + (errscalaV * Vin)**2)

# Calculate the transfer function
TR = Vo / Vin
print('TR',TR)
eTR = TR * np.sqrt((eVo / Vo)**2 + (eVin / Vin)**2+ 2 * (errscalaV**2))

# Plot Vin and Vout vs. f e the transfer function vs. f

fig, ax = plt.subplots(1, 2, figsize=(6, 4),sharex=True, constrained_layout = True, width_ratios=[1, 1])
ax[0].errorbar(fr,Vin,yerr=eVin, fmt='o', label=r'$V_{in}$',ms=2)
ax[0].errorbar(fr,Vo,yerr=eVo, fmt='o', label=r'$V_{out}$',ms=2)
ax[0].scatter(234.9, np.min(Vin),c='greenyellow', label=r'$V_{in,min}$')
ax[0].legend(prop={'size': 10}, loc='best')
ax[0].set_ylabel(r'Voltaggio (V)')
ax[0].grid(True)
ax[1].errorbar(fr,TR,yerr=eTR, fmt='o', label=r'$T=\frac{V_{out}}{V_{in}}$',ms=2,color='red')
ax[1].legend(prop={'size': 10}, loc='best')
ax[1].scatter(234.9, np.max(TR), label=r'$f_0$',c='deepskyblue')
ax[1].scatter(240.33, np.max(TR)/np.sqrt(2), label=r'$f_1, f_2$',c='orange', s=20)
ax[1].scatter(229.78, np.max(TR)/np.sqrt(2),c='orange', s=20)
ax[1].set_ylabel(r'Funzione di trasferimento $T_R$')
ax[1].set_xlabel(r'Frequenza (kHz)')
ax[1].yaxis.set_ticks_position('right')
ax[1].yaxis.set_label_position('right')
ax[1].axhline(y=np.max(TR)/np.sqrt(2), color='orange', linestyle='--', label=r'$T=\frac{T_{max}}{\sqrt{2}}$')
ax[1].legend(prop={'size': 10}, loc='best')

plt.grid(True)
plt.savefig(file+'_1'+'.png',
            bbox_inches ="tight",
            pad_inches = 1,
            transparent = True,
            facecolor ="w",
            edgecolor ='w',
            orientation ='Portrait',
            dpi = 100)

plt.show()
