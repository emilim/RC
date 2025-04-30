import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.optimize import curve_fit
import mplhep as hep
from cycler import cycler
import matplotlib.colors as colors
import multiprocessing.pool


# settaggio globale grafici    
#print(plt.style.available)
#plt.style.use('classic')
plt.style.use(hep.style.ROOT)
params = {'legend.fontsize': '10',
         'legend.loc': 'upper right',
          'legend.frameon':       'True',
          'legend.framealpha':    '0.8',      # legend patch transparency
          'legend.facecolor':     'w', # inherit from axes.facecolor; or color spec
          'legend.edgecolor':     'w',      # background patch boundary color
          'figure.figsize': (6, 4),
         'axes.labelsize': '10',
         'figure.titlesize' : '14',
         'axes.titlesize':'12',
         'xtick.labelsize':'10',
         'ytick.labelsize':'10',
         'lines.linewidth': '1',
         'text.usetex': False,
#         'axes.formatter.limits': '-5, -3',
         'axes.formatter.min_exponent': '2',
#         'axes.prop_cycle': cycler('color', 'bgrcmyk')
         'figure.subplot.left':'0.125',
         'figure.subplot.bottom':'0.125',
         'figure.subplot.right':'0.925',
         'figure.subplot.top':'0.925',
         'figure.subplot.wspace':'0.1',
         'figure.subplot.hspace':'0.1',
#         'figure.constrained_layout.use' : True
          }
plt.rcParams.update(params)
plt.rcParams['axes.prop_cycle'] = cycler(color=['b','g','r','c','m','y','k'])

# Enable debug mode
DEB = False


# Function definition


def fitf(x, A, delta, OM):
    omega = 2.0 * np.pi * x * 1e3  # input in kHz
    fitval = (A * (OM**2)) / np.sqrt(omega**4 - 2.0 * omega**2 * (OM**2 - 2.0 * delta**2) + OM**4)
    return fitval

def fitchi2(i,j,k):
    x = fr
    y = TR
    y_err = eTR
    A,OM,delta = A_chi[i],OM_chi[j],DEL_chi[k]
    omega = 2.0 * np.pi * x * 1e3  # input in kHz
    fitval = (A * (OM**2)) / np.sqrt(omega**4 - 2.0 * omega**2 * (OM**2 - 2.0 * delta**2) + OM**4)
    residuals = (y - fitval) / y_err
    chi2 = np.sum(residuals**2)
    mappa[i,j,k] = chi2

def profi2D(axis,matrix3D):
    if axis == 1 :
        mappa2D = np.array([[np.min(mappa[:,b,c]) for b in range(step)] for c in range(step)])
    if axis == 2 :
        mappa2D = np.array([[np.min(mappa[a,:,c]) for a in range(step)] for c in range(step)])
    if axis == 3 :
        mappa2D = np.array([[np.min(mappa[a,b,:]) for a in range(step)] for b in range(step)])
    return mappa2D

def profi1D(axis, mappa):
    if 1 in axis :
        mappa2D = np.array([[np.min(mappa[:,b,c]) for b in range(step)] for c in range(step)])
#        print('1')
        if 2 in axis:
            mappa1D = np.array([np.min(mappa2D[b,:]) for b in range(step)])
#            print('2')
        if 3 in axis:
            mappa1D = np.array([np.min(mappa2D[:,c]) for c in range(step)])
#            print('3')
    else :
#        print('2-3')
        mappa2D = np.array([[np.min(mappa[a,:,c]) for a in range(step)] for c in range(step)])
        mappa1D = np.array([np.min(mappa2D[a,:]) for a in range(step)])
    return mappa1D



## PARAMATERS

# Number of bins for parameter scan
NI = 20
NJ = 20
NK = 20

# Input file name
file = 'RLC_Cres'
inputname = file+'.txt'

# Number of points to fit
N = 23

# Frequency limits for the fit function (in kHz)
frfit0 = 10.0
frfit1 = 100.0

# Initial parameter values
Ainit= 0.9
OMinit = 300000.  # Hz
DELinit = 15000. # Hz

# Assumed reading errors
letturaV = 0.041
errscalaV = 0.012


#### LOAD DATA
    

# Read data from the input file
data = np.loadtxt(inputname)
fr = data[:, 0] #frequenze
Vin = data[:, 1] #Vin
Vo = data[:, 2] #Vout
VdivR = data[:, 4] #divisioni-FS del Vin

# Number of points to fit
# va a contare il numero di frequenze nel vettore fr che siano maggiori di zero
n = len(fr[fr > 0])

# Calculate errors on x and y
eVo = np.sqrt((letturaV * VdivR)**2 + (errscalaV * Vo)**2)
eVin = np.sqrt((letturaV * data[:, 3])**2 + (errscalaV * Vin)**2)

# Calculate the transfer function
TR = Vo / Vin
eTR = TR * np.sqrt((eVo / Vo)**2 + (eVin / Vin)**2) ######## ??????? + 2 * (errscalaV**2))

# Plot Vin and Vout vs. f e the transfer function vs. f

fig, ax = plt.subplots(1, 2, figsize=(5, 4),sharex=True, constrained_layout = True, width_ratios=[1, 1])
ax[0].errorbar(fr,Vin,yerr=eVin, fmt='o', label=r'$V_{in}$',ms=2)
ax[0].errorbar(fr,Vo,yerr=eVo, fmt='o', label=r'$V_{out}$',ms=2)
ax[0].legend(prop={'size': 10}, loc='best')
ax[0].set_ylabel(r'Voltaggio (V)')

ax[1].errorbar(fr,TR,yerr=eTR, fmt='o', label=r'$T=\frac{V_{out}}{V{in}}$',ms=2,color='red')
ax[1].legend(prop={'size': 10}, loc='best')
ax[1].set_ylabel(r'Funzione di trasferimento')
ax[1].set_xlabel(r'Frequenza (kHz)')
ax[1].yaxis.set_ticks_position('right')
ax[1].yaxis.set_label_position('right')

plt.savefig(file+'_1'+'.png',
            bbox_inches ="tight",
            pad_inches = 1,
            transparent = True,
            facecolor ="w",
            edgecolor ='w',
            orientation ='Portrait',
            dpi = 100)

plt.show()


# Perform the fit

popt, pcov = curve_fit(fitf, fr, TR, p0=[Ainit, DELinit, OMinit], method='trf', sigma=eTR, bounds=([0.6, 8000, 250000], [1.1, 25000, 350000]))

"""
POPT: Vettore con la stima dei parametri dal fit
PCOV: Matrice delle covarianze
bounds sono i limiti inferiori e superiori dei parametri (si richiede positività delle stime in questo caso)
"""

perr = np.sqrt(np.diag(pcov))
print( ' ampiezza = {a:.3f} +/- {b:.3f} \n delta = {c:.1f} +/- {d:.1f} kHz \n Omega = {e:.1f} +/- {f:.1f} kHz '.format(a=popt[0], b=perr[0],c=popt[1]/1000,d=perr[1]/1000,e=popt[2]/1000,f=perr[2]/1000))

residuA = TR - fitf(fr, *popt)

x_fit = np.linspace(min(fr), max(fr), 1000)

"""
fit tracciato con mille punti fra la freq min e max
"""

# Plot the fit
fig, ax = plt.subplots(2, 1, figsize=(5, 4),sharex=True, constrained_layout = True, height_ratios=[2, 1])
ax[0].plot(x_fit, fitf(x_fit, *popt), label='Fit', linestyle='--', color='black')
ax[0].plot(x_fit,fitf(x_fit,Ainit,DELinit,OMinit), label='init guess', linestyle='dashed', color='green')
ax[0].errorbar(fr,TR,yerr=eTR, fmt='o', label=r'$T=\frac{V_{out}}{V{in}}$',ms=2,color='red')
ax[0].legend(loc='upper left')
ax[0].set_ylabel(r'Funzione di trasferimento')
#ax[0].set_xticks([20,30,40,50])

ax[1].errorbar(fr,residuA,yerr=eTR, fmt='o', label=r'Residui$',ms=2,color='red')
ax[1].set_ylabel(r'Residui')
ax[1].set_xlabel(r'Frequenza (kHz)')
ax[1].plot(fr,np.zeros(len(fr)))

plt.savefig(file+'_2'+'.png',
            bbox_inches ="tight",
            pad_inches = 1,
            transparent = True,
            facecolor ="w",
            edgecolor ='w',
            orientation ='Portrait',
            dpi = 100)

plt.show()


# Extract and print best fit parameters and errors
A_BF, DEL_BF, OM_BF = popt #parametri del best fit
eA_BF, eDEL_BF, eOM_BF = np.sqrt(np.diag(pcov))

print("============== BEST FIT with SciPy ====================")
print(f"A = ({A_BF} +/- {eA_BF})")
print(f"OM = ({OM_BF * 1e-3} +/- {eOM_BF * 1e-3}) kHz")
print(f"DEL = ({DEL_BF * 1e-3} +/- {eDEL_BF * 1e-3}) kHz")
print("=======================================================")


# Define the interval for parameter limits
NSI = 2
A0, A1 = A_BF - NSI * eA_BF, A_BF + NSI * eA_BF
OM0, OM1 = OM_BF - NSI * eOM_BF, OM_BF + NSI * eOM_BF
DEL0, DEL1 = DEL_BF - NSI * eDEL_BF, DEL_BF + NSI * eDEL_BF

print(f"A0 = ({A0}, A1 = {A1})")
print(f"OM0 = ({OM0}, OM1 = {OM1}) Hz")
print(f"DEL0 = ({DEL0}, DEL1 = {DEL1}) Hz")

# Perform the fit with parameter limits
popt_limited, pcov_limited = curve_fit(
    fitf, fr, TR, method='trf', sigma=eTR, bounds=([A0, DEL0, OM0], [A1, DEL1, OM1])
)

# Extract and print best fit parameters and errors after parameter limits
A_BF, DEL_BF, OM_BF = popt_limited
eA_BF, eDEL_BF, eOM_BF = np.sqrt(np.diag(pcov_limited))

print("============== BEST FIT with SciPy (par limits) =========")
print(f"A = ({A_BF} +/- {eA_BF})")
print(f"OM = ({OM_BF * 1e-3} +/- {eOM_BF * 1e-3}) kHz")
print(f"DEL = ({DEL_BF * 1e-3} +/- {eDEL_BF * 1e-3}) kHz")
print("=======================================================")


# Calcolo mappa Chi2 3D

step = 100

A_chi = np.linspace(A0,A1,step)
OM_chi = np.linspace(OM0,OM1,step)
DEL_chi = np.linspace(DEL0,DEL1,step)

mappa = np.zeros((step,step,step))
item = [(i,j,k) for i in range(step) for j in range(step) for k in range(step)]

# assign the global pool
pool = multiprocessing.pool.ThreadPool(100)
# issue top level tasks to pool and wait
pool.starmap(fitchi2, item, chunksize=10)
# close the pool
pool.close()

mappa = np.asarray(mappa)            
print(mappa.shape)
print(np.argmin(mappa))

chi2_min = np.min(mappa)
#print(mappa.shape,np.argmin(mappa),chi2_min)
print(chi2_min)

#Calcolo Profilazione 2D
chi2D = profi2D(1,mappa)
Achi2D = profi2D(2,mappa) 

#Calcolo Profilazioni 1D
prof_OM = profi1D([1,2],mappa)
prof_DEL = profi1D([1,3],mappa)   
prof_A = profi1D([2,3],mappa)    

# trovo l'errore sui parametri

lvl = chi2_min+1. # 2.3 # 3.8
diff_OM = abs(prof_OM-lvl)
diff_DEL = abs(prof_DEL-lvl)
diff_A = abs(prof_A-lvl)

#print(diff_OM, diff_OM[OM_chi<OM_BF])

OM_dx = np.argmin(diff_OM[OM_chi<OM_BF])
OM_sx = np.argmin(diff_OM[OM_chi>OM_BF])+len(diff_OM[OM_chi<OM_BF])
DEL_dx = np.argmin(diff_DEL[DEL_chi<DEL_BF])
DEL_sx = np.argmin(diff_DEL[DEL_chi>DEL_BF])+len(diff_DEL[DEL_chi<DEL_BF])
A_dx = np.argmin(diff_A[A_chi<A_BF])
A_sx = np.argmin(diff_A[A_chi>A_BF])+len(diff_A[A_chi<A_BF])

print(OM_dx,OM_sx,DEL_dx,DEL_sx,A_dx,A_sx)

cmap = mpl.colormaps['plasma'].reversed()
#print(chi2D.shape)
level = np.linspace(47,70,100)
line_c = 'gray'

fig, ax = plt.subplots(2, 2, figsize=(5, 4),constrained_layout = True, height_ratios=[3, 1], width_ratios=[1,3], sharex='col', sharey='row')
im = ax[0,1].contourf(OM_chi,DEL_chi,chi2D, levels=level, cmap=cmap) 
cbar = fig.colorbar(im, extend='both', shrink=0.9, ax=ax[0,1], ticks = [50,55,60,65]) 
cbar.set_label(r'$\chi^2$',rotation=360)

CS = ax[0,1].contour(OM_chi,DEL_chi,chi2D, levels=[chi2_min+0.001,chi2_min+1,chi2_min+2.3,chi2_min+3.8],linewidths=1, colors='k',alpha=0.5,linestyles='dotted')
ax[0,1].clabel(CS, inline=True, fontsize=9, fmt='%1.1f')
print(np.argmin(prof_OM),np.argmin(prof_DEL),OM_chi[np.argmin(prof_OM)],DEL_chi[np.argmin(prof_DEL)])
ax[0,1].text(OM_chi[np.argmin(prof_OM)],DEL_chi[np.argmin(prof_DEL)]+50.,str(np.around(chi2_min,1)), color='k',alpha=0.5, fontsize=9)
ax[0,1].plot([OM0,OM1],[DEL_chi[OM_sx],DEL_chi[OM_sx]],color=line_c, ls='dashed')
ax[0,1].plot([OM0,OM1],[DEL_chi[OM_dx],DEL_chi[OM_dx]],color=line_c, ls='dashed')
ax[0,1].plot([OM_chi[DEL_sx],OM_chi[DEL_sx]],[DEL0,DEL1],color=line_c, ls='dashed')
ax[0,1].plot([OM_chi[DEL_dx],OM_chi[DEL_dx]],[DEL0,DEL1],color=line_c, ls='dashed')

ax[0,0].plot(prof_OM,DEL_chi) 
ax[0,0].plot([47,60],[DEL_chi[OM_sx],DEL_chi[OM_sx]], color=line_c, ls='dashed') 
ax[0,0].plot([47,60],[DEL_chi[OM_dx],DEL_chi[OM_dx]], color=line_c, ls='dashed') 
ax[0,0].text(49,DEL_chi[np.argmin(prof_DEL)],str(np.around(DEL_chi[np.argmin(prof_DEL)],0)), color='k',alpha=0.5, fontsize=9)
ax[0,0].text(52,DEL_chi[OM_sx]+50.,str(np.around(DEL_chi[OM_sx]-DEL_chi[np.argmin(prof_DEL)],0)), color='b',alpha=0.5, fontsize=9)
ax[0,0].text(52,DEL_chi[OM_dx]-120.,str(np.around(DEL_chi[OM_dx]-DEL_chi[np.argmin(prof_DEL)],0)), color='r',alpha=0.5, fontsize=9)

ax[1,1].plot(OM_chi,prof_DEL) 
ax[1,1].plot([OM_chi[DEL_sx],OM_chi[DEL_sx]],[47,60], color=line_c, ls='dashed') 
ax[1,1].plot([OM_chi[DEL_dx],OM_chi[DEL_dx]],[47,60], color=line_c, ls='dashed') 
ax[1,1].text(OM_chi[np.argmin(prof_OM)]-200,50,str(np.around(OM_chi[np.argmin(prof_OM)],0)), color='k',alpha=0.5, fontsize=9)
ax[1,1].text(OM_chi[DEL_sx]+50.,54.5,str(np.around(OM_chi[DEL_sx]-OM_chi[np.argmin(prof_OM)],0)), color='b',alpha=0.5, fontsize=9)
ax[1,1].text(OM_chi[DEL_dx]-270.,54.5,str(np.around(OM_chi[DEL_dx]-OM_chi[np.argmin(prof_OM)],0)), color='r',alpha=0.5, fontsize=9)

ax[1,0].set_axis_off()
ax[0,0].set_ylabel(r'$\delta\left(Hz\right)$') 
ax[1,1].set_xlabel(r'$\Omega\left(Hz\right)$') 
ax[0,0].set_xlim(47,60) 
ax[1,1].set_ylim(47,60)

plt.savefig(file+'_3'+'.png',
            bbox_inches ="tight",
            pad_inches = 1,
            transparent = True,
            facecolor ="w",
            edgecolor ='w',
            orientation ='Portrait',
            dpi = 100)

plt.show()

fig, ax = plt.subplots(2, 2, figsize=(5, 4),constrained_layout = True, height_ratios=[3, 1], width_ratios=[1,3], sharex='col', sharey='row')
im = ax[0,1].contourf(A_chi,DEL_chi,chi2D, levels=level, cmap=cmap) 
cbar = fig.colorbar(im, extend='both', shrink=0.9, ax=ax[0,1], ticks = [50,55,60,65]) 
cbar.set_label(r'$\chi^2$',rotation=360)

CS = ax[0,1].contour(A_chi,DEL_chi,chi2D, levels=[chi2_min+0.001,chi2_min+1,chi2_min+2.3,chi2_min+3.8],linewidths=1, colors='k',alpha=0.5,linestyles='dotted')
ax[0,1].clabel(CS, inline=True, fontsize=9, fmt='%1.1f')
print(np.argmin(prof_A),np.argmin(prof_DEL),A_chi[np.argmin(prof_A)],DEL_chi[np.argmin(prof_DEL)])
ax[0,1].text(A_chi[np.argmin(prof_A)],DEL_chi[np.argmin(prof_DEL)]+50.,str(np.around(chi2_min,1)),color='k',alpha=0.5, fontsize=9)
ax[0,1].plot([A0,A1],[DEL_chi[A_sx],DEL_chi[A_sx]],color=line_c, ls='dashed')
ax[0,1].plot([A0,A1],[DEL_chi[A_dx],DEL_chi[A_dx]],color=line_c, ls='dashed')
ax[0,1].plot([A_chi[DEL_sx],A_chi[DEL_sx]],[DEL0,DEL1],color=line_c, ls='dashed')
ax[0,1].plot([A_chi[DEL_dx],A_chi[DEL_dx]],[DEL0,DEL1],color=line_c, ls='dashed')

ax[0,0].plot(prof_A,DEL_chi) 
ax[0,0].plot([47,60],[DEL_chi[A_sx],DEL_chi[A_sx]], color=line_c, ls='dashed') 
ax[0,0].plot([47,60],[DEL_chi[A_dx],DEL_chi[A_dx]], color=line_c, ls='dashed') 

ax[1,1].plot(A_chi,prof_DEL) 
ax[1,1].plot([A_chi[DEL_sx],A_chi[DEL_sx]],[47,60], color=line_c, ls='dashed') 
ax[1,1].plot([A_chi[DEL_dx],A_chi[DEL_dx]],[47,60], color=line_c, ls='dashed') 
ax[1,1].text(A_chi[np.argmin(prof_A)]-0.001,50,str(np.around(A_chi[np.argmin(prof_A)],3)), color='k',alpha=0.5, fontsize=9)
ax[1,1].text(A_chi[DEL_sx]+0.001,54.5,str(np.around(A_chi[DEL_sx]-A_chi[np.argmin(prof_A)],3)), color='b',alpha=0.5, fontsize=9)
ax[1,1].text(A_chi[DEL_dx]-0.006,54.5,str(np.around(A_chi[DEL_dx]-A_chi[np.argmin(prof_A)],3)), color='r',alpha=0.5, fontsize=9)

ax[1,0].set_axis_off()
ax[0,0].set_ylabel(r'$\delta\left(Hz\right)$') 
ax[1,1].set_xlabel(r'A') 
ax[0,0].set_xlim(47,60) 
ax[1,1].set_ylim(47,60)

plt.savefig(file+'_4'+'.png',
            bbox_inches ="tight",
            pad_inches = 1,
            transparent = True,
            facecolor ="w",
            edgecolor ='w',
            orientation ='Portrait',
            dpi = 100)

plt.show()



