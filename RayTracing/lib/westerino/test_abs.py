# Testing the absorption coefficient.

# Import statements
from pylab import *
from westerinoECabsorption import dampbq

# Set numerical parameter
nptx = 300
nptn = 100
xmax = 150.0

# Set physical parameters
f = 170.                          # (beam frequency in GHz)
ne0 = 1.e14                       # (max. density in cm^-3)
Lne = 151.0                       # (density scale in cm)
B0 = 5.5                          # (max. magnetic field in Tesla)
Te0 = 15.                         # (max. temperature in KeV)
LTe = 160.                        # (temperature scale in cm)
R0 = 621.                         # (Major radius in cm)
me = 9.1e-31                      # (electron mass in kg)

# Set numerical grid
x = linspace(-xmax, +xmax, nptx)
Nll = linspace(-1., +1., nptn)

# Set mode label and propagation angle 
mode = +1                         # (O-mode)

# Set plasma profiles
ne = ne0 * (1 - (x / Lne)**4)**.1 # (density in cm^-3)
B = 1.e4 * B0 / (1 + x / R0)      # (magnetic field in Gauss)
Te = Te0 * (1 - (x / LTe)**2)     # (temperature in KeV)

# Set derived profiles
omega = 2. * pi * f * 1.e9        # (beam frequency in rad/sec)
omega_p = 5.64e4 * sqrt(ne)       # (plasma frequency in rad/sec)
omega_c = 1.76e7 * B              # (cyclotron frequency in rad/sec)

# Define the array for Im(N_perp)
ImNperp = zeros([nptx, nptn])

# Span the x-N space and compute Ni
icalled = 0
for ix in range(0,nptx):
    alpha = (omega_p[ix] / omega)**2
    beta = (omega_c[ix] / omega)**2
    vTe = sqrt(3.2e-16 * Te[ix] / me) / 3.e8    # (Thermal speed)
    Nr = sqrt(1. - alpha)                       # (O-mode dispersion)
    for iN in range(0, nptn):
        if abs(Nll[iN] / Nr) <= 1.:
            theta = arccos(Nll[iN] / Nr)
            ImNperp[ix,iN] = dampbq(theta, Nr, alpha, beta, vTe, mode, icalled)
            if icalled == 0: icalled = 1

# Plotting
figure(1, figsize=(12,10))
subplots_adjust(bottom=0.1, top=0.95, left=0.1, right=0.9, hspace=0.3)

# density
subplot(221)
plot(x, ne)
xlabel('$x = R - R_0$ [cm]')
ylabel('cm^-3')
title('electron density')
grid('on')

# temperature
subplot(222)
plot(x, Te)
xlabel('$x = R - R_0$ [cm]')
ylabel('KeV')
title('electron temperature')
grid('on')

# magnetic field
subplot(223)
plot(x, B)
xlabel('$x = R - R_0$ [cm]')
ylabel('Gauss')
title('magnetic field')
grid('on')

# Contours of the imaginary part of the refractive index
subplot(224)
contour(x, Nll, transpose(ImNperp))
xlabel('$x = R - R_0$ [cm]')
ylabel('$N_\parallel$')
title('absorption contours')
colorbar()
grid('on')

# show
show()
