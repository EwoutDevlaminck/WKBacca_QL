from QL_bounce_calc_v2 import *

import warnings
warnings.filterwarnings("ignore") # Fuck around until I find out
p_norm = np.linspace(0, 30, 20)
ksi = np.linspace(-.999, .999, 30)

# We need a binning of the data according to the original liuqe/LUKE grids!
# For now the liuqe grid, but later ideally we'll have the LUKE one
# Look into this!
filename_WKBeam = '/home/devlamin/Documents/WKBeam_related/WKBacca_dev_v1/WKBacca_cases/TCV72644/t_1.05/output/L1_binned_QL.hdf5'
WhatToResolve, FreqGHz, mode, Wfct, Absorption, EnergyFlux, rho, theta, Npar, Nperp = read_h5file(filename_WKBeam)

filename_Eq = '/home/devlamin/Documents/WKBeam_related/WKBacca_QL/WKBacca_cases/TCV72644/t_1.05/L1_raytracing.txt'

idata = InputData(filename_Eq)
Eq = TokamakEquilibrium(idata)

DRF0_wh, DRF0D_wh, DRF0F_wh, DRF0_hw, DRF0D_hw, DRF0F_hw, DRF0_hh, DRF0D_hh = \
    D_RF(rho**2, theta, p_norm, ksi, Npar, Nperp, Wfct, Eq, n=[2, 3], FreqGHz=FreqGHz)


Pw, Kh = np.meshgrid(p_norm, ksi[:-1])

PP, PPer = Pw * Kh, Pw * np.sqrt(1 - Kh**2)

fig, axs = plt.subplots(4, 2, figsize=(12, 18))

for i, ax in enumerate(axs.flatten()):
    ax.pcolormesh(PP, PPer, DRF0_wh[2*i].T, cmap='plasma')
    ax.set_title(f'rho = {rho[2*i]:.2f}')
    ax.set_xlabel(r'$p\{\|}$')
    ax.set_ylabel(r'$p_{\perp}$')
    ax.set_aspect('equal')

plt.show()