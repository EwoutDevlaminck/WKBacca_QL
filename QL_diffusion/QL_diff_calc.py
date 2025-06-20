# General imports
import sys
import os
import time
import warnings
warnings.filterwarnings("ignore") # Fuck around until I find out

# Imports for python
import numpy as np
import matplotlib.pyplot as plt
import h5py
from scipy.io import loadmat

# MPI import
from mpi4py import MPI

# WKBeam-specific imports
from CommonModules.input_data import InputData 
from CommonModules.PlasmaEquilibrium import TokamakEquilibrium
import CommonModules.physics_constants as phys
from Tools.PlotData.PlotAbsorptionProfile.plotabsprofile import compute_deposition_profiles

# WKBacca functions import
from QL_diffusion.QL_diff_aux import *

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

def call_QLdiff(input_file):
    """ Driver for the QL diffusion calculation. """
    # Record time for performance measurement
    start_time = time.time()

    #------------------------------#
    #---MPI implementation----------#
    #------------------------------#

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Initialize shared data
    if rank == 0:
        # Initialize variables, load data, and pre-process as needed

        # The file for the Ql calculation
        idata           = InputData(input_file)
        # The file for the equilibrium and raytracing
        configfile      = idata.configfile
        eqdata          = InputData(configfile)   
        Eq              = TokamakEquilibrium(eqdata)
        harmonics       = np.array(idata.harmonics)
        DKE_calc        = idata.DKE_calc


        if idata.manual_grids:
            # Load the grids from the input file
            pmin        = idata.pmin
            pmax        = idata.pmax
            n_p         = idata.np
            p_norm_w    = np.linspace(pmin, pmax, n_p)
            n_ksi       = idata.nksi
            anglegrid   = np.linspace(-np.pi, 0, n_ksi)
            ksi0_w      = np.cos(anglegrid)

            # Calculate the normalised momentum and pitch angle on the half grid
            p_norm_h = 0.5 * (p_norm_w[1:] + p_norm_w[:-1])
            ksi0_h = 0.5 * (ksi0_w[1:] + ksi0_w[:-1])

        else:
            #Load the grids from the provided filename
            try:
                grid_file = idata.gridfile
                grids = loadmat(grid_file)['WKBacca_grids']
                ksi0_h = grids['ksi0_h'][0,0][0]
                ksi0_w = grids['ksi0_w'][0,0][0]
                p_norm_h = grids['p_norm_h'][0,0][0]
                p_norm_w = grids['p_norm_w'][0,0][0]

            except:
                print("Error: Could not load the grids from the provided file.")
                sys.exit(1)

        try:
            filename_RhoThetaN = idata.outputdirectory+'RhoThetaN_binned.hdf5'
            
            WhatToResolve, FreqGHz, mode, Wfct, Absorption, EnergyFlux, rhobins, thetabins, Nparallelbins, Nperpbins = read_h5file(filename_RhoThetaN)

        except:
            print("Error: Could not find the binned data for the quasilinear calculation.")
            sys.exit(1)
        
        # From the bins, calculate the central values and the bin widths

        d_rho = np.diff(rhobins)
        rho = 0.5 * (rhobins[1:] + rhobins[:-1])

        psibins = rhobins**2
        d_psi = np.diff(psibins)
        psi = 0.5 * (psibins[1:] + psibins[:-1])

        d_theta = np.diff(thetabins)[0]
        theta = 0.5 * (thetabins[1:] + thetabins[:-1])

        d_npar = np.diff(Nparallelbins)[0]
        Npar = 0.5 * (Nparallelbins[1:] + Nparallelbins[:-1])

        d_nperp = np.diff(Nperpbins)[0]
        Nperp = 0.5 * (Nperpbins[1:] + Nperpbins[:-1])


        dV_N = 2 * np.pi * Nperp * d_nperp * d_npar
        ptV = np.zeros((len(rho), len(theta)))
        R = np.zeros((len(rho), len(theta)))
        Z = np.zeros((len(rho), len(theta)))

        for l, psi_val in enumerate(psi):
            for t, theta_val in enumerate(theta):
                ptV[l, t] = 2*np.pi * 1e-6 * d_psi[l] * d_theta * Eq.volume_element_J(theta_val, psi_val)
                R[l, t], Z[l, t] = Eq.flux_to_grid_coord(psi_val, theta_val)
        # As in notes, (called W_KB in the notes), we need Wfct * 4pi/c /(dV_N)
        Edens =  Wfct[:,:,:,:,0] / np.sum(ptV, axis=1)[:, None, None, None]
        Edens /= dV_N[None, None, None, :] # Energy density in k-space
        Edens *= 4*np.pi /c * 1e6 
        # With this, Edens_N is k-space energy density in J/N^2
        # No factor 2 comes in anywhere anymore. The energy density is derived from the full complex electric field so in fact
        # The normalisation receives a factor 2 already.

            
        # Calculate the reference quantities needed for the normalisation
        omega = phys.AngularFrequency(FreqGHz)
        _, _, _, _, _, _, ptNe, ptTe, _, _, _, _, _ = config_quantities(psi, theta, omega, Eq)
        Ne_ref = np.amax(ptNe)
        Te_ref = np.amax(ptTe)

        # Variables to hold results
        DRF0_wh = np.zeros((len(psi), len(p_norm_w), len(ksi0_h), len(harmonics)))
        DRF0_hw = np.zeros((len(psi), len(p_norm_h), len(ksi0_w), len(harmonics)))
        DRF0_hh = np.zeros((len(psi), len(p_norm_h), len(ksi0_h), len(harmonics)))
        if DKE_calc:
            DRF0D_wh = np.zeros((len(psi), len(p_norm_w), len(ksi0_h), len(harmonics)))
            DRF0D_hw = np.zeros((len(psi), len(p_norm_h), len(ksi0_w), len(harmonics)))
            DRF0D_hh = np.zeros((len(psi), len(p_norm_h), len(ksi0_h), len(harmonics)))

            DRF0F_wh = np.zeros((len(psi), len(p_norm_w), len(ksi0_h), len(harmonics)))
            DRF0F_hw = np.zeros((len(psi), len(p_norm_h), len(ksi0_w), len(harmonics)))
        else:
            DRF0D_wh = np.zeros_like(psi)
            DRF0D_hw = np.zeros_like(psi)
            DRF0D_hh = np.zeros_like(psi)

            DRF0F_wh = np.zeros_like(psi)
            DRF0F_hw = np.zeros_like(psi)
            
            
        Trapksi0_h = np.zeros(len(psi))
        Trapksi0_w = np.zeros(len(psi))

        task_queue = [(i, psi_val, Edens[i]) for i, psi_val in enumerate(psi)] # (index, psi, Wfct slice)

        # Sort task queue by descending psi values
        # Higher psi values have more trapped particles, which are more expensive to calculate
        task_queue.sort(key=lambda x: x[1], reverse=True)

    else:
        mode = None
        FreqGHz = None
        harmonics = None
        thetabins = None
        p_norm_h = None
        p_norm_w = None
        ksi0_h = None
        ksi0_w = None
        Npar = None
        Nperp = None
        Eq = None
        Ne_ref = None
        Te_ref = None
        DKE_calc = None

    # Broadcast shared data
    mode = comm.bcast(mode, root=0)
    FreqGHz = comm.bcast(FreqGHz, root=0)
    harmonics = comm.bcast(harmonics, root=0)
    thetabins = comm.bcast(thetabins, root=0)
    p_norm_w = comm.bcast(p_norm_w, root=0)
    p_norm_h = comm.bcast(p_norm_h, root=0)
    ksi0_w = comm.bcast(ksi0_w, root=0)
    ksi0_h = comm.bcast(ksi0_h, root=0)
    Npar = comm.bcast(Npar, root=0)
    Nperp = comm.bcast(Nperp, root=0)
    Eq = comm.bcast(Eq, root=0)
    Ne_ref = comm.bcast(Ne_ref, root=0)
    Te_ref = comm.bcast(Te_ref, root=0)
    DKE_calc = comm.bcast(DKE_calc, root=0)

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

            idx, psi_value, Edens_slice = task

            #Expand dimension of Wfct to have len=1 in the first dimension
            Edens_slice = np.expand_dims(Edens_slice, axis=0)

            # Perform the calculation
            DRF0_wh_loc, DRF0D_wh_loc, DRF0F_wh_loc, DRF0_hw_loc,\
            DRF0D_hw_loc, DRF0F_hw_loc, DRF0_hh_loc, DRF0D_hh_loc, Trapksi0_h_loc, Trapksi0_w_loc = \
            D_RF([psi_value], thetabins, p_norm_w, p_norm_h, ksi0_w, ksi0_h, Npar, Nperp, Edens_slice, Eq, Ne_ref, Te_ref, n=harmonics, FreqGHz=FreqGHz, DKE_calc=DKE_calc)

            result_data = (DRF0_wh_loc[0], DRF0D_wh_loc[0], DRF0F_wh_loc[0], DRF0_hw_loc[0],
                        DRF0D_hw_loc[0], DRF0F_hw_loc[0], DRF0_hh_loc[0], DRF0D_hh_loc[0], Trapksi0_h_loc, Trapksi0_w_loc)

            # Send result back to master
            comm.send((rank, idx, result_data), dest=0, tag=2)


    # Store the data at the end

    if rank == 0:
        outputname = idata.outputdirectory + idata.outputfilename + '.h5'

        # Transpose the arrays to have [np, nksi, npsi, nharmonics] as used in LUKE
        DRF0_wh = np.transpose(DRF0_wh, (1, 2, 0, 3))
        DRF0_hw = np.transpose(DRF0_hw, (1, 2, 0, 3))
        DRF0_hh = np.transpose(DRF0_hh, (1, 2, 0, 3))
        
        if DKE_calc:
            DRF0D_wh = np.transpose(DRF0D_wh, (1, 2, 0, 3))
            DRF0D_hw = np.transpose(DRF0D_hw, (1, 2, 0, 3))
            DRF0D_hh = np.transpose(DRF0D_hh, (1, 2, 0, 3))

            DRF0F_wh = np.transpose(DRF0F_wh, (1, 2, 0, 3))
            DRF0F_hw = np.transpose(DRF0F_hw, (1, 2, 0, 3))

        #--------------------------------#
        #---Efficient data storage-------#
        #--------------------------------#

        # Save the data
        with h5py.File(outputname, 'w') as file:
            file.create_dataset('harmonics', data=harmonics)
            file.create_dataset('psi', data=psi)
            file.create_dataset('ksi0_w', data=ksi0_w)
            file.create_dataset('ksi0_h', data=ksi0_h)
            file.create_dataset('p_norm_w', data=p_norm_w)
            file.create_dataset('p_norm_h', data=p_norm_h)
            file.create_dataset('FreqGHz', data=FreqGHz)
            file.create_dataset('mode', data=mode)
            file.create_dataset('Trapksi0_h', data=Trapksi0_h)
            file.create_dataset('Trapksi0_w', data=Trapksi0_w)

            # Add absorption data
            filename_abs = idata.absorption_file
            filename_abs_dat = idata.absorption_data_file
            absorption_data = compute_deposition_profiles(InputData(filename_abs), filename_abs_dat)
            file.create_dataset('rho_abs', data=absorption_data['rho'])
            file.create_dataset('dP_dV', data=absorption_data['dP_dV'])
            file.create_dataset('dP_drho', data=absorption_data['dP_drho'])
            file.create_dataset('dV_drho', data=absorption_data['dV_drho'])

            # Create sparse matrices for the bounce integrals
            # And accompanying masks, all ordered in the fortran style
            # This, because reshaping and ordering in matlab is done according to the fortran convention of row-first ordering
            mask_DRF0_wh = np.where(DRF0_wh.flatten(order='F') > 1e-5)
            DRF0_wh_sparse = DRF0_wh.flatten(order='F')[mask_DRF0_wh]

            mask_DRF0_hw = np.where(DRF0_hw.flatten(order='F') > 1e-5)
            DRF0_hw_sparse = DRF0_hw.flatten(order='F')[mask_DRF0_hw]

            mask_DRF0_hh = np.where(DRF0_hh.flatten(order='F') > 1e-5)
            DRF0_hh_sparse = DRF0_hh.flatten(order='F')[mask_DRF0_hh]


            if DKE_calc:
                mask_DRF0D_wh = np.where(DRF0D_wh.flatten(order='F') > 1e-5)
                DRF0D_wh_sparse = DRF0D_wh.flatten(order='F')[mask_DRF0D_wh]

                mask_DRF0D_hw = np.where(DRF0D_hw.flatten(order='F') > 1e-5)
                DRF0D_hw_sparse = DRF0D_hw.flatten(order='F')[mask_DRF0D_hw]

                mask_DRF0D_hh = np.where(DRF0D_hh.flatten(order='F') > 1e-5)
                DRF0D_hh_sparse = DRF0D_hh.flatten(order='F')[mask_DRF0D_hh]

                mask_DRF0F_wh = np.where(DRF0F_wh.flatten(order='F') > 1e-5)
                DRF0F_wh_sparse = DRF0F_wh.flatten(order='F')[mask_DRF0F_wh]

                mask_DRF0F_hw = np.where(DRF0F_hw.flatten(order='F') > 1e-5)
                DRF0F_hw_sparse = DRF0F_hw.flatten(order='F')[mask_DRF0F_hw]

            else:
                mask_DRF0D_wh = np.zeros(1)
                DRF0D_wh_sparse = np.zeros(1)
                mask_DRF0D_hw = np.zeros(1)
                DRF0D_hw_sparse = np.zeros(1)
                mask_DRF0D_hh = np.zeros(1)
                DRF0D_hh_sparse = np.zeros(1)
                mask_DRF0F_wh = np.zeros(1)
                DRF0F_wh_sparse = np.zeros(1)
                mask_DRF0F_hw = np.zeros(1)
                DRF0F_hw_sparse = np.zeros(1)


            # save the data
            file.create_dataset('DRF0_wh_sparse', data=DRF0_wh_sparse)
            file.create_dataset('mask_DRF0_wh', data=mask_DRF0_wh)
            file.create_dataset('DRF0_hw_sparse', data=DRF0_hw_sparse)
            file.create_dataset('mask_DRF0_hw', data=mask_DRF0_hw)
            file.create_dataset('DRF0_hh_sparse', data=DRF0_hh_sparse)
            file.create_dataset('mask_DRF0_hh', data=mask_DRF0_hh)

            file.create_dataset('DRF0D_wh_sparse', data=DRF0D_wh_sparse)
            file.create_dataset('mask_DRF0D_wh', data=mask_DRF0D_wh)
            file.create_dataset('DRF0D_hw_sparse', data=DRF0D_hw_sparse)
            file.create_dataset('mask_DRF0D_hw', data=mask_DRF0D_hw)
            file.create_dataset('DRF0D_hh_sparse', data=DRF0D_hh_sparse)
            file.create_dataset('mask_DRF0D_hh', data=mask_DRF0D_hh)

            file.create_dataset('DRF0F_wh_sparse', data=DRF0F_wh_sparse)
            file.create_dataset('mask_DRF0F_wh', data=mask_DRF0F_wh)
            file.create_dataset('DRF0F_hw_sparse', data=DRF0F_hw_sparse)
            file.create_dataset('mask_DRF0F_hw', data=mask_DRF0F_hw)

        file.close()

        print(f'\nResults saved to {outputname}')
        # Print the elapsed time
        elapsed_time = time.time() - start_time
        print(f"Elapsed time: {elapsed_time:.2f} seconds")
    # Finalize MPI
    MPI.Finalize()


    return 0
#---------------------------#
#---Main function-------------#
#---------------------------#

if __name__ == "__main__":
    # Check if the script is being run as the main module
    if len(sys.argv) < 2:
        print("Usage: python QL_diff_calc.py <input_file>")
        sys.exit(1)

    # Get the input file from the command line arguments
    input_file = sys.argv[1]

    # Call the function to perform the QL diffusion calculation
    call_QLdiff(input_file)
    # Finalize MPI
    MPI.Finalize()
    sys.exit(0)
# End of script
#---------------------------#
