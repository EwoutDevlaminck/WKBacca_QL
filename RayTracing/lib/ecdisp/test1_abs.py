# plot the absorption coefficient for given values of R 
# and realistic tokamak plasma and so on.


# Import statements
from pylab import *
import math
from farinaECabsorption import warmdamp
import sys
from CommonModules.PlasmaEquilibrium import TokamakEquilibrium
from CommonModules.input_data import InputData   # where parameters are given
from RayTracing.modules.dispersion_matrix_cfunctions import *


input_file = 'StandardCases/ITER/ITERtest_raytracing.txt'
idata = InputData(input_file)

# initialise magnetic field object
MagField = TokamakEquilibrium(idata)



# Set numerical parameter
nptx = 100
nptn = 100

xmin = 162.5
xmax = 164.5
Deltax = (xmax-xmin)/nptx

z = 0.

nmin = -0.5
nmax = +0.5
Deltan = (nmax-nmin)/nptn


# Set physical parameters
f = idata.freq                    # (beam frequency in GHz)

                   # (electron mass in kg)
emass = 9.11*1e-31            # electron mass in kg
echarge = 1.602*1e-19         # electron charge in C
epsilon0 = 8.85*1e-12         # dielectric constant in V/Cm


# Set numerical grid
x = linspace(xmin, xmax, nptx)
Nll = linspace(nmin, nmax, nptn)


# Set mode label and propagation angle 
mode = idata.sigma

# Set derived profiles
omega = 2. * pi * f * 1.e9        # (beam frequency in rad/sec)


# Define the array for Im(N_perp)
ImNperp = zeros([nptx, nptn])
ImNperpModel = zeros([nptx, nptn])
magfield = zeros([nptx])
Temperature = zeros([nptx])
eldens = zeros([nptx])

# Span the x-N space and compute Ni
for ix in range(0,nptx):
    
    R = xmin + ix * Deltax

    Psi = MagField.PsiInt.eval(R,z)
    ne = MagField.NeInt.eval(R,z)
    Bt = MagField.BtInt.eval(R,z)
    BR = MagField.BRInt.eval(R,z)
    Bz = MagField.BzInt.eval(R,z)
    B = math.sqrt(Bt**2+BR**2+Bz**2)

    if ne < 0.:
##        print("negative density: ne = ", ne)
        ne = 0.

    omega_p = echarge*sqrt(ne*1e19/epsilon0/emass)
    omega_c = B*echarge / emass 
   
    Te = MagField.TeInt(Psi)

    alpha = (omega_p / omega)**2
    beta = (omega_c / omega)**2
    Te_loc = Te                                 # (Thermal speed in keV)
   
    eldens[ix] = ne
    Temperature[ix] = Te
    magfield[ix] = B
    
    for iN in range(0, nptn):
        Nll = nmin + iN * Deltan
        Nr = disNperp(omega, B, ne, Nll, mode, idata.epsilonRegS) 
        if abs(Nll / Nr) <= 1.:
            theta = arccos(Nll / Nr)
            ImNperp[ix,iN] = warmdamp(alpha, beta, Nr, theta, Te_loc, mode)
            
            
        Nref = 0.15
        a1 = 40.
        gamma1 = 1.
        Rabs = 163.5

        a = a1 * Nll**2 + Rabs
        if R < a:
            gamma = gamma1*(1.-Nll**2)**2 / (1.+abs(Nll)**3/Nref**3) * (a-R)
        else:
            gamma = 0.
        
        ImNperpModel[ix,iN] = gamma


# Plotting
figure(1, figsize=(12,10))
subplots_adjust(bottom=0.1, top=0.95, left=0.1, right=0.9, hspace=0.3)

xlist = np.linspace(xmin,xmax,nptx)
Nlllist = np.linspace(nmin,nmax,nptn)


subplot(321)
plot(xlist,eldens)
title('electron density')
xlabel('R (cm)')
ylabel(r'$n_e$ (10$^13$ / cm$^3$')

subplot(322)
plot(xlist,magfield)
title('magnetic field')
xlabel('R (cm)')
ylabel(r'B (T)')

subplot(323)
plot(xlist,Temperature)
title('temperature')
xlabel('R (cm)')
ylabel('T ( keV)')

# Contours of the imaginary part of the refractive index
subplot(324)
contour(xlist, Nlllist, transpose(ImNperp))
xlabel('$x = R - R_0$ [cm]')
ylabel('$N_\parallel$')
title('absorption contours')
colorbar()
grid('on')

subplot(326)
contour(xlist, Nlllist, transpose(ImNperpModel))
xlabel('$x = R - R_0$ [cm]')
ylabel('$N_\parallel$')
title('absorption contours (model)')
colorbar()
grid('on')

# show
show()
