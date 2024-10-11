import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import ticker
from scipy.spatial import KDTree
import h5py


from CommonModules.input_data import InputData 
from CommonModules.PlasmaEquilibrium import TokamakEquilibrium
import RayTracing.modules.dispersion_matrix_cfunctions as disp
import CommonModules.physics_constants as phys


from numba import jit
from mpi4py import MPI

##################"

def sum_over_dimensions(data, dims):
    for dim in dims:
        l = data.shape[dim]
        data = np.nansum(data, axis=dim)
        data /= l
    return data

def read_h5file(filename):
    """
    Read the h5 file and return the data
    """
    file            = h5py.File(filename, 'r')

    WhatToResolve   = file.get('WhatToResolve')[()]
    FreqGHz         = file.get('FreqGHz')[()]
    mode            = file.get('Mode')[()]
    Wfct            = file.get('BinnedTraces')[()]
    try:
        Absorption = file.get('Absorption')[()]
    except:
        Absorption = None

    try:
        EnergyFlux = file.get('VelocityField')[()]

    except:
        EnergyFlux = None

    
    rhomin          = file.get('rhomin')[()]
    rhomax          = file.get('rhomax')[()]
    nmbrrho         = file.get('nmbrrho')[()]
    rho             = np.linspace(rhomin, rhomax, nmbrrho)

    Thetamin        = file.get('Thetamin')[()]
    Thetamax       = file.get('Thetamax')[()]
    nmbrTheta        = file.get('nmbrTheta')[()]
    Theta           = np.linspace(Thetamin, Thetamax, nmbrTheta)

    Nparallelmin    = file.get('Nparallelmin')[()]
    Nparallelmax    = file.get('Nparallelmax')[()]
    nmbrNparallel   = file.get('nmbrNparallel')[()]
    Nparallel      = np.linspace(Nparallelmin, Nparallelmax, nmbrNparallel)

    Nperpmin        = file.get('Nperpmin')[()]
    Nperpmax        = file.get('Nperpmax')[()]
    nmbrNperp       = file.get('nmbrNperp')[()]
    Nperp          = np.linspace(Nperpmin, Nperpmax, nmbrNperp)

    file.close()

    return WhatToResolve, FreqGHz, mode, Wfct, Absorption, EnergyFlux, rho, Theta, Nparallel, Nperp

##################
  

# Calculate the product of both for every point in the grid, by summing over the k-space

# boltzmann constant
kB          = 1.38064852e-23 # 
# electron mass
m_e         = 9.10938356e-31

kBm_e       = kB * m_e

# speed of light
c           = 299792458
# Used in the calculation of the relativistic factor
cm_e        = c * m_e

# electron charge
e           = 1.60217662e-19 # C

# momentum conversion

e_over_me_c2 = e / (m_e * c**2)

# conversion eV to K
eV2K        = 11604.52500617

# epsilon_0
epsilon_0   = 8.854187817e-12

# plasma frequency conversion
wp_conv    = e**2 / (m_e * epsilon_0)

###############################

###############################

@jit(nopython=True)
def pTe_from_Te(Te):
    """
    Thermal momentum from temperature, normalised to m_e*c
    Te in keV
    factor e/(m_e*c**2) is precalculated
    """
    return np.sqrt(1e3 * Te* e_over_me_c2)

@jit(nopython=True)
def gamma(p, pTe):
    """
    Relativistic factor, for p a grid of momenta, normalized to the thermal momentum.
    pTe is the thermal momentum, normalised to m_e*c itself, making the calculation easy
    """
    return np.sqrt(1 + (p*pTe)**2)

@jit(nopython=True)
def N_par_resonant(P_norm, pTe, Ksi, Gamma, X, harm):
    """
    Calculate the resonant n_par. P_norm, Ksi and Gamma are of shape (n_p x n_ksi), StixY is a scalar.
    Returns a matrix of the same shape as P_norm.
    """
    return (Gamma - harm*X)/(pTe * P_norm * Ksi)
    

@jit(nopython=True)
def polarisation(N2, K_angle, P, R, L, S):
    PlusOverMinus = (N2 - R)/(N2 - L)
    ParOverMinus = - (N2 - S)/(N2 - L) * (N2*np.cos(K_angle)*np.sin(K_angle))/(P - N2*np.sin(K_angle)**2)

    emin2 = 1/(1 + PlusOverMinus**2 + ParOverMinus**2)
    eplus2 = PlusOverMinus**2 * emin2
    epar2 = ParOverMinus**2 * emin2

    return np.array([eplus2, emin2, epar2])

@jit(nopython=True)
def A_perp(nperp, p_norm, pTe, ksi, X):
    return nperp * p_norm * pTe * np.sqrt(1-ksi**2) * X

# Import the bessel functions
import scipy.special as sp

def bessel_integrand(n, x):
    return sp.jn(n, x)**2


def QL_diff(E_sq, psi, theta, ksi, p_norm, npar, nperp, Eq, n=[2, 3], freq=82.7):
    # Calculate the product in the integrand and sum over the k-space
    # we do this straight away in the full phase space. 
    # So rho is a 1D array, P_norm and Ksi are 1D arrays, npar and nperp are 1D arrays

    #Lowercase indicate 1D arrays, uppercase 2D arrays

    # Attemped implementation of the QL operator
    #For now, just a code skeleton

    
    # the momentum is normalised to the thermal momentum, which is flux surface dependent. We always look at the same normalised range.

    omega = phys.AngularFrequency(FreqGHz)

    P_norm, Ksi = np.meshgrid(p_norm, ksi)
    P_par, P_perp = P_norm * Ksi, P_norm * np.sqrt(1 - Ksi**2)


    # Something else we can already do, is make the KDtree for Npar. This will help us to quickly perform the Kronecker delta function.

    npar_tree = KDTree(npar.reshape(-1, 1))
    dnpar = npar[1] - npar[0]
    dpsi = psi[1] - psi[0]
    dtheta = theta[1] - theta[0]

    Npar, Nper = np.meshgrid(npar, nperp, indexing='ij')
    N2 = Npar**2 + Nper**2
    K_angle = np.arctan2(Nper, Npar)
    

    QL_nobounce = np.zeros([len(psi),len(theta), len(ksi), len(p_norm)])
    # Get the R,Z coordinates for the grid points
    R2d, Z2d = np.zeros([len(psi), len(theta)]), np.zeros([len(psi), len(theta)])

    for i_psi, psi_val in enumerate(psi):
        for i_theta, theta_val in enumerate(theta):
            # For reasons discussed in the notes, the iterations over rho and theta are done explicitly.

            # STILL MISSING DV 
            Jacobian = Eq.volume_element_J(theta_val, psi_val)
            V = Jacobian * dpsi * dtheta # Volume element in the grid

            # Can safely be taken as 2D, as the binning already averages out over the toroidal dimension.

            R, Z = Eq.flux_to_grid_coord(psi_val, theta_val)
            R2d[i_psi, i_theta], Z2d[i_psi, i_theta] = R, Z
            # Calculate the Stix parameters at this point in the grid
            Ne = Eq.NeInt.eval(R, Z) #1e19  m^-3
            Te = Eq.TeInt.eval(R, Z) #keV
            Bt = Eq.BtInt.eval(R, Z) #T
            BR = Eq.BRInt.eval(R, Z)
            Bz = Eq.BzInt.eval(R, Z)

            B = np.sqrt(Bt**2 + BR**2 + Bz**2)

            # Calculate the Stix parameters
            omegaP = disp.disParamomegaP(Ne)
            omegaC = disp.disParamOmega(B)
            P = 1 - omegaP**2/(omega**2)
            X = omegaC/omega

            R = (P + X)/(1 + X)
            L = (P - X)/(1 - X)
            S = .5 * (R + L)
        
            # Calculate the relativistic factor, given the thermal momentum at this location
            pTe = pTe_from_Te(Te)
            Gamma = gamma(P_norm, pTe)
                
                # resonance_N_par gives a 2D array with the same shape as P_norm and Ksi
                # This whole calculation is done dimensionless now.

                #P_norm is normalised to P_base, so P_norm/p_Te_factor is P/P_Te

            for harm in n:
                
                resonance_N_par = N_par_resonant(P_norm, pTe, Ksi, Gamma, X, harm) 

                # We can now query the KDTree to get the indices of the resonant values in Npar.
                # The code below efficiently looks for the index of the value in Npar that is closest to the resonant value.
                # This is done for every point in the grid. If it is not within dNpar/2 of any value in Npar, the index is set to -1.
                # This in fact replaces the integral over Npar, as we now just have a 2D array indicating what value of Npar is resonant, if any.

                dist_N_par, ind_N_par = npar_tree.query(np.expand_dims(resonance_N_par, axis=-1), distance_upper_bound=dnpar/2)
                res_condition_N_par = np.where(np.isinf(dist_N_par), -1, ind_N_par)

                
                if i_psi == 0 and i_theta == 100:
                    fig = plt.figure()
                    pl = plt.contour(P_par, P_perp, resonance_N_par, levels=np.linspace(-1, 1, 11), cmap='coolwarm')
                    if np.any(res_condition_N_par == -1):
                        norescon = plt.contour(P_par, P_perp, res_condition_N_par, colors='k', levels=[-1])
                    rescon = plt.contourf(P_par, P_perp, res_condition_N_par, cmap='Greens', levels=np.arange(0, len(Npar)))
                    plt.clabel(pl, inline=True, fontsize=8)
                    plt.colorbar(rescon, label='Npar index')
                    plt.xlabel(r'$p_\|/p_{Te}$')
                    plt.ylabel(r'$p_\perp/_{Te}$')
                    plt.title(f'Npar required for resonance at psi = {psi_val:.2f} and theta = {theta_val:.2f}')
                    

                res_mask_Pspace = np.where(res_condition_N_par != -1, True, False) # Mask for the resonant values in P_norm and Ksi
                i_Ksi_res, i_P_res = np.where(res_mask_Pspace) # Tuple containing 2 arrays.
                # The first array contains the indices of the resonant values in P_norm, the second in Ksi.

                # We can now use the mask to select the resonant values in P_norm and Ksi, and calculate the integrand. 
                # Where there is no Npar value resonant, we can skip the calculation.


                for i_P, i_Ksi in zip(i_P_res, i_Ksi_res):
                    i_npar = res_condition_N_par[i_Ksi, i_P]
                    # At this point, we check |E|² to see if at rho_val, theta_val, Npar[i_npar] (for this P,Ksi), the beam is present.
                    # And if so, for what value of Nperp.

                    mask_beam_present = np.where(E_sq[i_psi, i_theta, i_npar, :, 0] > 0, True, False)
                    # Make a list of the indices of the Nperp values where the beam is present
                    indeces_Nperp_beam_present = np.where(mask_beam_present)[0]
            

                    if len(indeces_Nperp_beam_present) > 0:

                        # Calculate the polarisation for the cells where the beam is present
                        # Now that we've pruned all the data so heavily, finally we can calculate the polarisation for the beam cells.
                        
                        for i_nperp in indeces_Nperp_beam_present:
                            pol = polarisation(N2[i_npar, i_nperp], K_angle[i_npar, i_nperp], P, R, L, S)

                            a_perp = A_perp(Nperp[i_nperp], P_norm[i_Ksi, i_P], pTe, Ksi[i_Ksi, i_P], X)
 
                            #Take the bessel functions...
                            #
                            #And combine to get polarisation term
                            #
                            Pol_term = .5* (pol[0] * bessel_integrand(harm-1, a_perp) + \
                                            pol[1] * bessel_integrand(harm+1, a_perp) )+ \
                                                pol[2] * bessel_integrand(harm, a_perp)

                            # And finally, the integrand is calculated.
                            QL_nobounce[i_psi, i_theta, i_Ksi, i_P] += Nperp[i_nperp] * Pol_term * E_sq[i_psi, i_theta, i_npar, i_nperp, 0]
                    
            # Multiply by the correct prefactor
            QL_nobounce[i_psi, i_theta] *= e**2 /(4*np.pi * V)

    return QL_nobounce, P_par, P_perp, R2d, Z2d
                        
# Start the execution of the code

#Only if we are running this script as the main script

if __name__ == '__main__':
    import sys
    import os
    import time

    tic = time.time()

    # For now, restricted to having the a number of processes that is a divisor of the number of rho values!

    comm = MPI.COMM_WORLD
    prank = comm.Get_rank()
    psize = comm.Get_size()

    psi_size = None
    psi_seg = None
    Wfct_seg = None
    WhatToResolve = None
    mode = None
    FreqGHz = None
    theta = None
    p_norm = None
    ksi = None
    Npar = None
    Nperp = None
    Eq = None

    if prank == 0:


        filename = '/home/devlamin/Documents/WKBeam_related/WKBacca_dev_v1/WKBacca_cases/TCV72644/t_1.05/output/L1_binned_QL_highres.hdf5'
        WhatToResolve, FreqGHz, mode, Wfct, Absorption, EnergyFlux, psi, theta, Npar, Nperp = read_h5file(filename)


        ###
        # For now, just take the central angle, ignore the rest

        Wfct_maxdens = np.amax(Wfct)

        beam_mask = np.where(Wfct < 1e-4*Wfct_maxdens, False, True)

        Wfct = np.where(beam_mask, Wfct, 0)

        input_file = '/home/devlamin/Documents/WKBeam_related/WKBacca_dev_v1/WKBacca_cases/TCV72644/t_1.05/L1_raytracing.txt'

        idata = InputData(input_file)
        Eq = TokamakEquilibrium(idata)

        # Define the grid for the calculation
        p_norm = np.linspace(0, 15, 300)
        ksi = np.linspace(-1, 1, 200)

        # Easily split the work by splitting the psi values
        psi_size = len(psi)
        psi_seg = np.array_split(psi, psize)
        Wfct_seg = np.array_split(Wfct, psize, axis=0)




        
    # Broadcast the shared data

    mode = comm.bcast(mode, root=0)
    FreqGHz = comm.bcast(FreqGHz, root=0)

    theta = comm.bcast(theta, root=0)

    p_norm = comm.bcast(p_norm, root=0)
    ksi = comm.bcast(ksi, root=0)

    Npar = comm.bcast(Npar, root=0)
    Nperp = comm.bcast(Nperp, root=0)
    
    Eq = comm.bcast(Eq, root=0)

    # Scatter the data that is split

    local_psi = comm.scatter(psi_seg, root=0)
    local_Wfct = comm.scatter(Wfct_seg, root=0) 






    local_QL, P_par, P_perp, local_R2d, local_Z2d = QL_diff(local_Wfct, local_psi, theta, ksi, p_norm, Npar, Nperp, Eq,  n=[2], freq=FreqGHz)

    # Gather the results

    local_sizes = comm.gather(len(local_psi), root=0)

    QL = None
    R2d = None
    Z2d = None

    if prank == 0:
        QL = np.empty((len(psi), len(theta), len(ksi), len(p_norm)))
        R2d = np.empty((len(psi), len(theta)))
        Z2d = np.empty((len(psi), len(theta)))
    comm.Gather(local_QL, QL, root=0)
    comm.Gather(local_R2d, R2d, root=0)
    comm.Gather(local_Z2d, Z2d, root=0)


    if prank == 0:

        QL_bounce = np.sum(QL, axis=1)/len(theta)

        toc = time.time()
        print('Time elapsed:', toc - tic)
        print('Done')
        QL_bounce_max = np.amax(QL_bounce)

        QL_spatial =  sum_over_dimensions(QL, [-2, -1])
        Wfct_spatial = sum_over_dimensions(Wfct, [-3, -2])
        Wfct_spatial = np.where(Wfct_spatial > 0, Wfct_spatial, 1e-10)
        Absorption_spatial = sum_over_dimensions(Absorption, [-3, -2])
        RHO, THETA = np.meshgrid(psi, theta)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        #RFfield = ax.pcolormesh(R2d, Z2d, Wfct_spatial[:, :, 1], norm = LogNorm())
        #RFfield = ax.contourf(R2d, Z2d, Wfct_spatial[:, :, 1], levels=100, locator=ticker.LogLocator(subs='auto'))
        RFfield = ax.contourf(R2d, Z2d, Wfct_spatial[:, :, 1], levels=100)
        spatial = ax.contour(R2d, Z2d, QL_spatial, cmap='hot', levels=10)
        ax.contour(R2d, Z2d, Absorption_spatial[:, :, 0], levels=10, cmap='bone', linewidths=1)
        fluxsurf = ax.contour(R2d, Z2d, RHO.T, levels=np.linspace(0, 1, 11), colors='grey', linestyles='dashed', linewidths=1)

        plt.clabel(fluxsurf, inline=True, fontsize=8)
        ax.set_aspect('equal')
        plt.colorbar(RFfield, label='RF field')
        

        fig = plt.figure(figsize=(10, 10))
        for i in range(20):
            ax = fig.add_subplot(5, 4, i+1)
            ql_plot = ax.contourf(P_par, P_perp, QL_bounce[i]/QL_bounce_max, levels=100)
            #ax.set_title(f'psi = {psi[i]:.2f}')
            ax.set_title(f'rho = {np.sqrt(psi[i]):.2f}')
            ax.set_xlabel(r'$p_\|/p_{Te}$')
            ax.set_ylabel(r'$p_\perp/_{Te}$')
            if i == 0:
                plt.colorbar(ql_plot, label='QL')
        plt.tight_layout()
            
        plt.show()

        # Save the data to an h5 file

        with h5py.File('QL_bounce.h5', 'w') as file:
            file.create_dataset('QL_bounce', data=QL_bounce)
            file.create_dataset('RHO', data=RHO)
            file.create_dataset('theta', data=theta)
            file.create_dataset('ksi', data=ksi)
            file.create_dataset('p_norm', data=p_norm)
            file.create_dataset('FreqGHz', data=FreqGHz)
            file.create_dataset('mode', data=mode)

    sys.exit(0)

