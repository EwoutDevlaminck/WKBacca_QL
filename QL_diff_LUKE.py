# General imports
import warnings
warnings.filterwarnings("ignore") # Fuck around until I find out
import sys
import os
import time

# Imports for python
import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy.special as sp
from scipy.spatial import KDTree
from scipy.interpolate import interp1d
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import minimize, fsolve

# MPI import
from mpi4py import MPI

# WKBeam-specific imports
from CommonModules.input_data import InputData 
from CommonModules.PlasmaEquilibrium import TokamakEquilibrium
import RayTracing.modules.dispersion_matrix_cfunctions as disp
import CommonModules.physics_constants as phys

# WKBacca functions import
from QL_bounce_calc_v3 import *

#---------------------------#
#---Define physics constants---#
#---------------------------#

eps = np.finfo(np.float32).eps

# electron mass
m_e         = 9.10938356e-31 # kg
# speed of light
c           = 299792458 # m/s
# electron charge
e           = 1.60217662e-19 # C
# momentum conversion
e_over_me_c2 = e / (m_e * c**2) 


tic = time.time()

#------------------------------#
#---Computation setup----------#
#------------------------------#

# WKBeam results, binned in appropriate dimensions
# TCV74302 case
#filename_WKBeam = '/home/devlamin/Documents/WKBeam_related/Cases_ran_before/TCV74302/output/L1_binned_QL.hdf5'
#filename_Eq = '/home/devlamin/Documents/WKBeam_related/WKBacca_QL/WKBacca_cases/TCV74302/L1_raytracing.txt'
#outputname = 'QL_bounce_TCV74302_test.h5'

#TCV72644 case
filename_WKBeam = '/home/devlamin/Documents/WKBeam_related/Cases_ran_before/TCV72644_1.25/No_fluct/output/L4_binned_QL.hdf5'
filename_Eq = '/home/devlamin/Documents/WKBeam_related/WKBacca_QL/WKBacca_cases/TCV72644_1.25/L4_raytracing.txt'
outputname = 'QL_bounce_TCV72644_1.25_test.h5'

# Momentum grids
p_norm = np.linspace(0, 15, 100)
anglegrid = np.linspace(-np.pi, 0, 300)
ksi0 = np.cos(anglegrid)
#ksi0 = np.linspace(-1, 1, 100)

#Harmonics to take into account
harmonics = [2]

plot_option = 1

#------------------------------#
#---MPI implementation----------#
#------------------------------#

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Initialize shared data
if rank == 0:
    # Initialize variables, load data, and pre-process as needed
    WhatToResolve, FreqGHz, mode, Wfct, Absorption, EnergyFlux, rho, theta, Npar, Nperp = read_h5file(filename_WKBeam)



    psi = rho**2 # Psi_p, the poloidal flux definition

    # For the calculation, we'll need to have the volume element
    # Psi is already a half-grid by definition, so we calculate dpsi as such
    d_psi = 1/2* (np.diff(psi)[:-1] + np.diff(psi)[1:])
    d_psi = np.concatenate(([np.diff(psi)[0]], d_psi, [np.diff(psi)[-1]]))
    #d_psi = [0.37**2-0.35**2]

    idata = InputData(filename_Eq)
    Eq = TokamakEquilibrium(idata)

    # For Npar, Nperp and theta, we assume that the grid is uniform
    d_npar = Npar[1] - Npar[0]
    d_nperp = Nperp[1] - Nperp[0]
    d_theta = theta[1] - theta[0]
    dV_N = 2 * np.pi * Nperp * d_nperp * d_npar
    ptV = np.zeros((len(rho), len(theta)))
    R = np.zeros((len(rho), len(theta)))
    Z = np.zeros((len(rho), len(theta)))

    for l, psi_val in enumerate(psi):
        for t, theta_val in enumerate(theta):
            ptV[l, t] = 2*np.pi * 1e-6 * d_psi[l] * d_theta * Eq.volume_element_J(theta_val, psi_val)
            R[l, t], Z[l, t] = Eq.flux_to_grid_coord(psi_val, theta_val)

    # As in notes, (called W_KB in the notes), we need Wfct * 4pi/c* 1/2piR * 1/(dV * dV_N)
    Edens =  Wfct[:,:,:,:,0] /ptV[:,:, None, None] 
    Edens /= 2*np.pi # Average over the toroidal angle
    Edens /= dV_N[None, None, None, :]
    Edens *= 4*np.pi /c * 1e6 
    # With this, Edens is the toroidally averaged energy density in J/m^3/N_volume, integrated over the refractive index angle


    if plot_option:
        plt.figure()
        ax = plt.subplot(111)
        Edens_2D_tor_avg = np.sum(Edens*dV_N[None, None, None, :], axis=(2, 3)) # Integrated over N_par, N_perp 
        Absorption_2D_tor_avg = np.sum(Absorption[:,:,:,:, 0]/ptV[:,:,None, None]*dV_N[None, None, None, :]/(2*np.pi*(R[:,:, None, None]/100)), axis=(2, 3))
        beam= ax.contourf(R, Z, Edens_2D_tor_avg, levels=50)
        absorb = ax.contour(R, Z, Absorption_2D_tor_avg, levels=10, cmap='hot')
        flux_surf = ax.contour(R, Z, np.tile(rho, (len(theta), 1)).T, levels=10, colors='black', linestyles='dashed', linewidths=0.5)
        ax.clabel(flux_surf, flux_surf.levels, inline=True, fontsize=10)
        ax.set_aspect('equal')
        plt.colorbar(beam)
        plt.title('Beam in poloidal plane, resolved in rho, theta')
    plt.show()
        
    # TEST, adding a factor V/2R0
    # Somewhere, a factor quite similar to this (propertional to either psi or the area of a flux surface) is needed
    Rp = Eq.magn_axis_coord_Rz[0] / 100
    flux_volumes = np.zeros_like(psi)
    for l, psi_val in enumerate(psi):
        flux_volumes[l] = Eq.compute_volume2(psi_val)
    Edens *= flux_volumes[:, None, None, None]/(2*Rp)

    # Calculate the reference quantities needed for the normalisation
    omega = phys.AngularFrequency(FreqGHz)
    _, _, _, _, _, _, ptNe, ptTe, _, _, _, _, _ = config_quantities(psi, theta, omega, Eq)
    Ne_ref = np.amax(ptNe)
    Te_ref = np.amax(ptTe)

    # Variables to hold results
    DRF0_wh = np.zeros((len(psi), len(p_norm), len(ksi0)-1, len(harmonics)))
    DRF0D_wh = np.zeros((len(psi), len(p_norm), len(ksi0)-1, len(harmonics)))
    DRF0F_wh = np.zeros((len(psi), len(p_norm), len(ksi0)-1, len(harmonics)))
    DRF0_hw = np.zeros((len(psi), len(p_norm)-1, len(ksi0), len(harmonics)))
    DRF0D_hw = np.zeros((len(psi), len(p_norm)-1, len(ksi0), len(harmonics)))
    DRF0F_hw = np.zeros((len(psi), len(p_norm)-1, len(ksi0), len(harmonics)))
    DRF0_hh = np.zeros((len(psi), len(p_norm)-1, len(ksi0)-1, len(harmonics)))
    DRF0D_hh = np.zeros((len(psi), len(p_norm)-1, len(ksi0)-1, len(harmonics)))
    Trapksi0_h = np.zeros(len(psi))
    Trapksi0_w = np.zeros(len(psi))

    task_queue = [(i, psi_val, d_psi[i], Edens[i]) for i, psi_val in enumerate(psi)] # (index, psi, Wfct slice)

    # Sort task queue by descending psi values
    # Higher psi values have more trapped particles, which are more expensive to calculate
    task_queue.sort(key=lambda x: x[1], reverse=True)

else:
    mode = None
    FreqGHz = None
    theta = None
    p_norm = None
    ksi0 = None
    Npar = None
    Nperp = None
    Eq = None
    Ne_ref = None
    Te_ref = None

# Broadcast shared data
mode = comm.bcast(mode, root=0)
FreqGHz = comm.bcast(FreqGHz, root=0)
theta = comm.bcast(theta, root=0)
p_norm = comm.bcast(p_norm, root=0)
ksi0 = comm.bcast(ksi0, root=0)
Npar = comm.bcast(Npar, root=0)
Nperp = comm.bcast(Nperp, root=0)
Eq = comm.bcast(Eq, root=0)
Ne_ref = comm.bcast(Ne_ref, root=0)
Te_ref = comm.bcast(Te_ref, root=0)

if rank == 0:
    # Master process
    num_tasks = len(task_queue)
    num_workers = size - 1
    task_idx = 0
    active_workers = 0
    finished_tasks = 0

    # Start distributing initial tasks
    for i in range(1, min(num_workers + 1, num_tasks + 1)):
        comm.send(task_queue[task_idx], dest=i, tag=1)
        task_idx += 1
        active_workers += 1
        print(f'\rActive workers: {active_workers}', end='', flush=True)

    while task_idx < num_tasks or active_workers > 0:
        # Receive results
        result = comm.recv(source=MPI.ANY_SOURCE, tag=2)
        worker_id, idx, result_data = result
        active_workers -= 1
        finished_tasks += 1
        print(f'\rActive workers: {active_workers}, Progress: {finished_tasks}/{num_tasks}', end='', flush=True)

        # Update results arrays with the data from worker
        DRF0_wh[idx], DRF0D_wh[idx], DRF0F_wh[idx], DRF0_hw[idx], \
        DRF0D_hw[idx], DRF0F_hw[idx], DRF0_hh[idx], DRF0D_hh[idx], Trapksi0_h[idx], Trapksi0_w[idx] = result_data


        if task_idx < num_tasks:
            # Send new task to this worker
            comm.send(task_queue[task_idx], dest=worker_id, tag=1)
            task_idx += 1
            active_workers += 1
            print(f'\rActive workers: {active_workers}, Progress: {finished_tasks}/{num_tasks}', end='', flush=True)

    # Send termination signal
    for i in range(1, size):
        comm.send(None, dest=i, tag=0)

else:
    # Worker process
    while True:
        task = comm.recv(source=0, tag=MPI.ANY_TAG)
        if task is None:
            break

        idx, psi_value, d_psi_value, Edens_slice = task

        #Expand dimension of Wfct to have len=1 in the first dimension
        Edens_slice = np.expand_dims(Edens_slice, axis=0)

        # Perform the calculation
        DRF0_wh_loc, DRF0D_wh_loc, DRF0F_wh_loc, DRF0_hw_loc,\
        DRF0D_hw_loc, DRF0F_hw_loc, DRF0_hh_loc, DRF0D_hh_loc, Trapksi0_h_loc, Trapksi0_w_loc = \
        D_RF([psi_value], [d_psi_value], theta, p_norm, ksi0, Npar, Nperp, Edens_slice, Eq, Ne_ref, Te_ref, n=harmonics, FreqGHz=FreqGHz)

        result_data = (DRF0_wh_loc[0], DRF0D_wh_loc[0], DRF0F_wh_loc[0], DRF0_hw_loc[0],
                    DRF0D_hw_loc[0], DRF0F_hw_loc[0], DRF0_hh_loc[0], DRF0D_hh_loc[0], Trapksi0_h_loc, Trapksi0_w_loc)

        # Send result back to master
        comm.send((rank, idx, result_data), dest=0, tag=2)


if rank == 0:
    toc = time.time()
    print(f'\rTime taken: {toc-tic:.2f} s', flush=True)

    # Save the data
    with h5py.File(outputname, 'w') as file:
        file.create_dataset('harmonics', data=harmonics)
        file.create_dataset('psi', data=psi)
        file.create_dataset('theta', data=theta)
        file.create_dataset('ksi0', data=ksi0)
        file.create_dataset('p_norm', data=p_norm)
        file.create_dataset('FreqGHz', data=FreqGHz)
        file.create_dataset('mode', data=mode)
        file.create_dataset('DRF0_wh', data=DRF0_wh)
        file.create_dataset('DRF0D_wh', data=DRF0D_wh)
        file.create_dataset('DRF0F_wh', data=DRF0F_wh)
        file.create_dataset('DRF0_hw', data=DRF0_hw)
        file.create_dataset('DRF0D_hw', data=DRF0D_hw)
        file.create_dataset('DRF0F_hw', data=DRF0F_hw)
        file.create_dataset('DRF0_hh', data=DRF0_hh)
        file.create_dataset('DRF0D_hh', data=DRF0D_hh)
        file.create_dataset('Trapksi0_h', data=Trapksi0_h)
        file.create_dataset('Trapksi0_w', data=Trapksi0_w)

    if plot_option:
        Pw, Kh = np.meshgrid(p_norm, ksi0[:-1])

        PP, PPer = Pw * Kh, Pw * np.sqrt(1 - Kh**2)

        fig, axs = plt.subplots(3, 2, figsize=(18, 12))

        maxDrf = np.amax(DRF0_wh)

        for i, ax in enumerate(axs.flatten()):
            ax.pcolormesh(PP, PPer, np.sum(DRF0_wh[2*i], axis=-1).T, cmap='plasma', vmax=maxDrf)
            ax.set_title(f'psi = {rho[2*i]**2:.2f}')
            ax.set_xlabel(r'$p\{||}$')
            ax.set_ylabel(r'$p_{\perp}$')
            ax.set_aspect('equal')
        plt.colorbar(axs[0, 0].pcolormesh(PP, PPer, np.sum(DRF0_wh[0], axis=-1).T, cmap='plasma', vmax=maxDrf), ax=axs, orientation='vertical')
        plt.show()


sys.exit(0)