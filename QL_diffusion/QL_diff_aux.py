import sys
import os
import time

import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy.special as sp
from scipy.spatial import KDTree
from scipy.interpolate import interp1d
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import minimize, fsolve


from CommonModules.input_data import InputData 
from CommonModules.PlasmaEquilibrium import TokamakEquilibrium
import RayTracing.modules.dispersion_matrix_cfunctions as disp
import CommonModules.physics_constants as phys

eps = np.finfo(np.float32).eps

# electron mass
m_e         = 9.10938356e-31 # kg
# speed of light
c           = 299792458 # m/s
# electron charge
e           = 1.60217662e-19 # C
# momentum conversion
e_over_me_c2 = e / (m_e * c**2) 


#-------------------------------#
#---Import the QL input from WKBeam---#
#-------------------------------#

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

    try:
        uniform_bins = file.get('uniform_bins')[()]
    except: 
        uniform_bins = True

    if uniform_bins:
        rhomin          = file.get('rhomin')[()]
        rhomax          = file.get('rhomax')[()]
        nmbrrho         = file.get('nmbrrho')[()]
        rhobins         = np.linspace(rhomin, rhomax, nmbrrho+1)

        Thetamin        = file.get('Thetamin')[()]
        Thetamax        = file.get('Thetamax')[()]
        nmbrTheta       = file.get('nmbrTheta')[()]
        Thetabins       = np.linspace(Thetamin, Thetamax, nmbrTheta+1)

        Nparallelmin    = file.get('Nparallelmin')[()]
        Nparallelmax    = file.get('Nparallelmax')[()]
        nmbrNparallel   = file.get('nmbrNparallel')[()]
        Nparallelbins   = np.linspace(Nparallelmin, Nparallelmax, nmbrNparallel+1)

        Nperpmin        = file.get('Nperpmin')[()]
        Nperpmax        = file.get('Nperpmax')[()]
        nmbrNperp       = file.get('nmbrNperp')[()]
        Nperpbins       = np.linspace(Nperpmin, Nperpmax, nmbrNperp+1)


    else:
        rhobins         = file.get('rhobins')[()]
        Thetabins       = file.get('Thetabins')[()]
        Nparallelbins   = file.get('Nparallelbins')[()]
        Nperpbins       = file.get('Nperpbins')[()]

    file.close()

    return WhatToResolve, FreqGHz, mode, Wfct, Absorption, EnergyFlux, rhobins, Thetabins, Nparallelbins, Nperpbins

#-------------------------------#
#--- Calculate configuration space quantities on psi, theta grid---#
#-------------------------------#

def config_quantities(psi, theta, omega, Eq):

    ptR, ptZ = np.zeros([len(psi), len(theta)]), np.zeros([len(psi), len(theta)])
    ptBt = np.zeros_like(ptR)
    ptBR = np.zeros_like(ptR)
    ptBz = np.zeros_like(ptR)
    ptB = np.zeros_like(ptR)
    ptNe = np.zeros_like(ptR)
    ptTe = np.zeros_like(ptR)

    P, X, R, L, S = np.zeros_like(ptR), np.zeros_like(ptR), np.zeros_like(ptR), np.zeros_like(ptR), np.zeros_like(ptR)

    for l, psi_l in enumerate(psi):
        for t, theta_t in enumerate(theta):
            ptR[l, t], ptZ[l, t] = Eq.flux_to_grid_coord(psi_l, theta_t)

            ptNe[l, t] = Eq.NeInt.eval(ptR[l, t], ptZ[l, t]) # 1e19 m⁻3
            ptTe[l, t] = Eq.TeInt.eval(ptR[l, t], ptZ[l, t]) # keV

            ptBt[l, t] = Eq.BtInt.eval(ptR[l, t], ptZ[l, t]) # T
            ptBR[l, t] = Eq.BRInt.eval(ptR[l, t], ptZ[l, t])
            ptBz[l, t] = Eq.BzInt.eval(ptR[l, t], ptZ[l, t])

            ptB[l, t] = np.sqrt(ptBt[l, t]**2 + ptBR[l, t]**2 + ptBz[l, t]**2)

            omega_pe = disp.disParamomegaP(ptNe[l, t])
            omega_ce = disp.disParamOmega(ptB[l, t])

            P[l, t] = 1 - omega_pe**2 / (omega**2)
            X[l, t] = omega_ce / omega

            R[l, t] = (P[l, t] + X[l, t])/(1 + X[l, t])
            L[l, t] = (P[l, t] - X[l, t])/(1 - X[l, t])
            S[l, t] = .5 * (R[l, t] + L[l, t])

    #Everything inside the WKbeam code is in cm, so we needed to keep R and Z in cm, but now we want it in m!
    return ptR/100, ptZ/100, ptBt, ptBR, ptBz, ptB, ptNe, ptTe, P, X, R, L, S

#-------------------------------#
#---Functions for trapping boundary---#
#-------------------------------#

#   - B_bounce(psi, ksi0): at what field a particle will bounce
#   - Ksi_trapping(psi): The boundary value for given psi. Particles with smaller ksi will be trapped
#   - theta_T,m and theta_T,M(psi,ksi0): minimal and maximal angle reached by particles (where they meet B_bounce)

def minmaxB(BInt_at_psi, theta):

    minusB_at_psi = -BInt_at_psi(theta)
    # Artificial shift to avoid problems of minimization, we shift the first half of the values and put them at the end.
    # If the optimum happens to lie beyond the original max value, we know it is actually at the theta value that was
    # shifted by 2*pi
    minusB_at_psi_shift = np.concatenate((minusB_at_psi[len(theta)//2:], minusB_at_psi[:len(theta)//2]))
    theta_shift = np.concatenate((theta[len(theta)//2:], theta[:len(theta)//2] + 2*np.pi))

    minusB_at_psiInt = interp1d(theta_shift, minusB_at_psi_shift, kind='cubic')

    minimum = minimize(BInt_at_psi, 0.)
    maximum = minimize(minusB_at_psiInt, np.pi)

    # Return the minimum and maximum
    return BInt_at_psi(minimum.x), -minusB_at_psiInt(maximum.x)

#-------------------------------#

def Trapping_boundary(ksi0, BInt_at_psi, theta_grid=[], eps = np.finfo(np.float32).eps):
    TrapB = np.zeros_like(ksi0)
    theta_roots = np.zeros((len(ksi0), 2))

    B0, Bmax = minmaxB(BInt_at_psi, theta_grid)
    TrapB = B0/(1-ksi0**2)# + eps) # Might revision, is eps needed?
    Trapksi0 = np.sqrt(1-B0/Bmax)

    for j, ksi0_val in enumerate(ksi0):
        if abs(ksi0_val) <= Trapksi0:

            def deltaB(x):
                return BInt_at_psi(x) - TrapB[j]

            theta_roots[j] = fsolve(deltaB, [-np.pi/2, np.pi/2])
        else:
            theta_roots[j, 0] = -np.pi
            theta_roots[j, 1] = np.pi

    return TrapB, Trapksi0, theta_roots

#-------------------------------#
#---Helper functions for the calculation of D_RF(psi, theta, p, ksi)---#
#-------------------------------#

#@jit(nopython=True)
def pTe_from_Te(Te):
    """
    Thermal momentum from temperature, normalised to m_e*c
    Te in keV
    factor e/(m_e*c**2) is precalculated
    """
    return np.sqrt(1e3 * Te* e_over_me_c2)

#@jit(nopython=True)
def gamma(p, pTe):
    """
    Relativistic factor, for p a grid of momenta, normalized to the thermal momentum.
    pTe is the thermal momentum, normalised to m_e*c itself, making the calculation easy
    """
    return np.sqrt(1 + (p*pTe)**2)

#@jit(nopython=True)
def N_par_resonant(inv_kp, p_Te, Gamma, X, harm):
    """
    Calculate the resonant n_par. p_norm and Gamma are of shape (n_p), StixY is a scalar.
    Returns an array of the same shape as p_norm
    """
    return (Gamma - harm*X)/p_Te *inv_kp

#@jit(nopython=True)
def polarisation(N2, K_angle, P, R, L, S):
    PlusOverMinus = (N2 - R)/(N2 - L)
    ParOverMinus = - (N2 - S)/(N2 - L) * (N2*np.cos(K_angle)*np.sin(K_angle))/(P - N2*np.sin(K_angle)**2)

    emin2 = 1/(1 + PlusOverMinus**2 + ParOverMinus**2)
    eplus2 = PlusOverMinus**2 * emin2
    epar2 = ParOverMinus**2 * emin2

    return np.array([eplus2, emin2, epar2])

#@jit(nopython=True)
def A_perp(nperp, p_norm, pTe, ksi, X):
    return - nperp * p_norm * pTe * np.sqrt(1-ksi**2) / X

# Import the bessel functions
import scipy.special as sp

def bessel_integrand(n, x):
    return sp.jn(n, x)**2

#-------------------------------#
#---Functions for the prefactor of bounce averaged D_RF matrices---#
#-------------------------------#

def D_RF_prefactor(p_norm, ksi0, Ne_ref, Te_ref, omega, eps):

    p_Te = pTe_from_Te(Te_ref)
    Gamma_Te = gamma(1, p_Te)
    coulomb_log = 31.3 - 0.5 * np.log(1e19*Ne_ref) + np.log(1e3*Te_ref) # DKE 6.50 with n_e in 1e19 m⁻3 and T_e in keV
    Gamma = gamma(p_norm, p_Te)
    P_norm, Ksi0 = np.meshgrid(p_norm, ksi0)
    inv_kabsp = 1 /(abs(Ksi0)* P_norm + eps)
    omega_pe = disp.disParamomegaP(Ne_ref)

    prefac =  8*np.pi**2 * Gamma * inv_kabsp / (m_e * omega_pe**2 * coulomb_log * Gamma_Te**3)  * (c/omega)

    return prefac.T

#-------------------------------#
#---Function for the calculation of D_RF(psi, theta, p, ksi)---#
#----------------------------

def D_RF_nobounce(p_norm, ksi, npar, nperp, Wfct, Te, P, X, R, L, S, harm, eps, plot=False):
        
    # The subfunction, that for a given psi_l and ksi, calculates the 
    # contribution to D_RF for a given momentum grid [i].
    # This is done for all harmonics in n.

    npar_tree = KDTree(npar.reshape(-1, 1))
    d_npar = npar[1] - npar[0] # Assume a constant grid for now
    d_nperp = nperp[1] - nperp[0] # Assume a constant grid for now
    

    # Precalculate the inverse ksi*p_norm grid, with a small offset to avoid division by zero
    inv_kp = 1 / (ksi * p_norm + eps)

    # Calculate the relativistic factor, given the thermal momentum at this location
    p_Te = pTe_from_Te(Te) # Normalised to m_e*c
    Gamma = gamma(p_norm, p_Te)

    # Initialise the integrand
    D_RF_integrand = np.zeros_like(p_norm)


    # resonance_N_par gives a 1D array of the resonant Npar values [i]
    # This whole calculation is done dimensionless now. 

    resonance_N_par = N_par_resonant(inv_kp, p_Te, Gamma, X, harm)

    # We can now query the KDTree to get the indices of the resonant values in Npar.
    # The code below efficiently looks for the index of the value in Npar that is closest to the resonant value.
    # This is done for every point in the grid. If it is not within dNpar/2 of any value in Npar, the index is set to -1.
    # This in fact replaces the integral over Npar, as we now just have a 2D array indicating what value of Npar is resonant, if any.

    dist_bound = d_npar/2 # How far away a point can be to be considered resonant
    dist_N_par, ind_N_par = npar_tree.query(np.expand_dims(resonance_N_par, axis=-1), k=2, distance_upper_bound=dist_bound) #/2
    res_condition_N_par = np.where(np.isinf(dist_N_par), -1, ind_N_par)
    
    i_res, n_par_res = np.where(res_condition_N_par != -1)

    # The array i_res contains the indices of the resonant values in the p_norm grid for this psi, ksi pair.
    # n_par_res has the indeces of the resonant Npar values in the res_condition_N_par array.

    # We can now use the mask to select the resonant values in p_norm and calculate the integrand. 
    # Where there is no Npar value resonant, we can skip the calculation.
    
    for i, m in zip(i_res, n_par_res):

        i_npar = res_condition_N_par[i, m]
        # At this point, we check |E|² to see if at Npar[i_npar] (for this p), the beam is present.
        # And if so, for what value of Nperp.

        #TEST Gaussian prefactor based on the resonance condition
        #prefac = np.exp(-dist_N_par[i,m]**2/(1/2*d_npar**2)) # Gaussian weight with sigma = d_npar/2
        prefac = 1/ (2*dist_bound) # Uniform weight
        mask_beam_present = Wfct[i_npar, :] > 0 
        indeces_Nperp_beam_present = np.where(mask_beam_present)[0]
        

        if np.any(mask_beam_present):
            # Calculate the polarisation for the cells where the beam is present
            # Now that we've pruned all the data so heavily, finally we can calculate the polarisation for the beam cells.
            for i_nperp in indeces_Nperp_beam_present:
                # Calculate local quantities
                N2 = nperp[i_nperp]**2 + npar[i_npar]**2
                K_angle = np.arctan2(npar[i_npar], nperp[i_nperp])
                pol = polarisation(N2, K_angle, P, R, L, S)

                a_perp = A_perp(nperp[i_nperp], p_norm[i], p_Te, ksi, X)

                #Take the bessel functions...
                #And combine to get polarisation term
                
                Pol_term = .5* (pol[0] * bessel_integrand(harm-1, a_perp) + \
                                pol[1] * bessel_integrand(harm+1, a_perp) )+ \
                                    pol[2] * bessel_integrand(harm, a_perp)

                # Now we can calculate the integrand for this point in the grid
                D_RF_integrand[i] += d_nperp * prefac * d_npar * nperp[i_nperp] * Pol_term * Wfct[i_npar, i_nperp]

    return D_RF_integrand


#-------------------------------#
#---Function for a bounce sum---#
#-------------------------------#

def bounce_sum(d_theta_grid_j, CB_j, Func, passing, sigma_dep=False):
    if passing or not sigma_dep:
        # Even if it is trapped, when there's no explicit sigma dependence, we can just sum over the grid
        # Instead of summing over both signs of ksi
        return np.nansum(d_theta_grid_j/(2*np.pi) * CB_j * Func)
    else:
        return 1/2* (np.nansum(d_theta_grid_j/(2*np.pi) * CB_j * Func) + np.sum(d_theta_grid_j/(2*np.pi) * CB_j * -Func))
    
#-------------------------------#
#---Function for the calculation of D_RF---#
#---THIS IS THE MAIN FUNCTION---#
#-------------------------------#

def D_RF(psi, theta_w, p_norm_w, p_norm_h, ksi0_w, ksi0_h, npar, nperp, Edens, Eq, Ne_ref, Te_ref, n=[2, 3], FreqGHz=82.7, DKE_calc=False, eps=np.finfo(np.float32).eps):

    """
    The main function to calculate the RF diffusion coefficients.
    It takes in the following arguments:

        psi: np.array [l]
            The radial coordinate, already on the half grid as required
        theta_w: np.array [t]
            The poloidal coordinate, on the whole grid made of the edges of the bins
        p_norm_w: np.array [i]
            The normalised (to the thermal momentum) momentum grid (whole grid)
        ksi0_w: np.array [j]
            The pitch angle grid (whole grid)
        npar: np.array [length given by WKBeam binning]
            The parallel refractive index
        nperp: np.array [length given by WKBeam binning]
            The perpendicular refractive index
        Edens: np.array [l, t, npar, nperp]
            The electric power density in [J/m^3/[N-volume]] from WKbeam
        Eq: equilibrium object
            The equilibrium object, containing the equilibrium quantities
            From WKBeam too
        n: list of harmonics to take into account
        FreqGHZ: float [GHZ]
        DK_calc: bool, wether to calculate the DKE diffusion coefficients

    Returns:
    
            DRF0_wh: np.array [l, i_w, j_h]
                The FP diffusion coefficient D_RF0 on the whole-half grid
            DRF0D_wh: np.array [l, i_w, j_h]
                The DKE diffusion coefficient D_RF0D on the whole-half grid
            DRF0F_wh: np.array [l, i_w, j_h]
                The DKE convection coefficient D_RF0F on the whole-half grid
            DRF0_hw: np.array [l, i_h, j_w] 
                The FP diffusion coefficient D_RF0 on the half-whole grid
            DRF0D_hw: np.array [l, i_h, j_w]
                The DKE diffusion coefficient D_RF0D on the half-whole grid
            DRF0F_hw: np.array [l, i_h, j_w]
                The DKE convection coefficient D_RF0F on the half-whole grid
            DRF0_hh: np.array [l, i_h, j_h]
                The FP diffusion coefficient D_RF0 on the half-half grid
            DRF0D_hh: np.array [l, i_h, j_h]
                The DKE diffusion coefficient D_RF0D on the half-half grid
    """

    # Timekeeping
    tic_internal = time.time()
    #---------------------------------#
    #---Calculate the quantities needed for the calculation---#
    #---------------------------------#

    omega = phys.AngularFrequency(FreqGHz)
    
    # Precaution to not have exactly 0 values in the grid for ksi, as this would be
    # infintely trapped particles
    # Hopefully will not be needed in final implementation, if LUKE grids are ok.
    ksi0_h[abs(ksi0_h)<1e-4] = 1e-4
    ksi0_w[abs(ksi0_w)<1e-4] = 1e-4

    
    # For theta. Psi is a half grid already, theta is a full grid

    d_theta = np.diff(theta_w)
    theta_h = theta_w[:-1] + d_theta/2

    # Precalculate quantities in configuration space
    # Careful, these are only to be used for passign particles, that keep the full theta grid!
    Rp, Zp = Eq.magn_axis_coord_Rz /100 #m
    ptR, ptZ, ptBt, ptBR, ptBz, ptB, ptNe, ptTe, P, X, R, L, S = config_quantities(psi, theta_h, omega, Eq)


    #--------------------------------#
    #---Initialision of grids---#
    #--------------------------------#

    # Trapping boundaries on psi, ksi grids
    # Trapksi [l, -], theta_T [l, j, 2]
    Trapksi0_w, theta_T_w = np.zeros((len(psi), 1)), np.zeros((len(psi), len(ksi0_w), 2))
    Trapksi0_h, theta_T_h = np.zeros((len(psi), 1)), np.zeros((len(psi), len(ksi0_h), 2))

    # Initialise the final 8 matrices, the D_RF matrices [l, i, j]
    # Also the lambda*q matrices [l, j]
    lambda_q_h = np.zeros((len(psi), len(ksi0_h)))
    lambda_q_w = np.zeros((len(psi), len(ksi0_w)))
                          
    DRF0_wh = np.zeros((len(psi), len(p_norm_w), len(ksi0_h), len(n)))
    DRF0_hw = np.zeros((len(psi), len(p_norm_h), len(ksi0_w), len(n)))
    DRF0_hh = np.zeros((len(psi), len(p_norm_h), len(ksi0_h), len(n)))

    if DKE_calc:
        DRF0D_wh = np.zeros((len(psi), len(p_norm_w), len(ksi0_h), len(n)))
        DRF0D_hw = np.zeros((len(psi), len(p_norm_h), len(ksi0_w), len(n)))
        DRF0D_hh = np.zeros((len(psi), len(p_norm_h), len(ksi0_h), len(n))) 

        DRF0F_wh = np.zeros((len(psi), len(p_norm_w), len(ksi0_h), len(n)))
        DRF0F_hw = np.zeros((len(psi), len(p_norm_h), len(ksi0_w), len(n)))
    else:
        DRF0D_wh, DRF0D_hw, DRF0D_hh = np.zeros_like(psi), np.zeros_like(psi), np.zeros_like(psi)
        DRF0F_wh, DRF0F_hw = np.zeros_like(psi), np.zeros_like(psi)



    #--------------------------------#
    # ---Calculation split into the psi grid---#
    #--------------------------------#

    for l, psi_l in enumerate(psi):
        # The calculation is split completely into the psi grid,
        # as the calculation is independent for every psi value

        ptB_Int_at_psi = interp1d(theta_h, ptB[l, :],fill_value=np.amax(ptB[l, :]), bounds_error=False)

        _, Trapksi0_w[l], theta_T_w[l] = Trapping_boundary(ksi0_w, ptB_Int_at_psi, theta_h)
        _, Trapksi0_h[l], theta_T_h[l] = Trapping_boundary(ksi0_h, ptB_Int_at_psi, theta_h)
        # First find the normalisation prefactor for the D_RF matrices [i, j]
        C_RF_wh =  D_RF_prefactor(p_norm_w, ksi0_h, Ne_ref, Te_ref, omega, eps)
        C_RF_hw =  D_RF_prefactor(p_norm_h, ksi0_w, Ne_ref, Te_ref, omega, eps)
        C_RF_hh =  D_RF_prefactor(p_norm_h, ksi0_h, Ne_ref, Te_ref, omega, eps)


        #--------------------------------#
        #---Bounce averaging calculation-#
        #--------------------------------#

        # Precalculate an interpolation function for Edens at psi_l
        # This is only needed for ksi values that correspond to trapped particles
        # The actual interpolation will differ for each ksi value, but the function is the same
        Edens_at_psi = Edens[l,:,:,:]
        Edens_Int_at_psi = RegularGridInterpolator((theta_h, npar, nperp), Edens_at_psi, bounds_error=False, fill_value=None)


        #--------------------------------#
        #---ksi0 half grid calculation---#
        #--------------------------------#

        for j, ksi0_val in enumerate(ksi0_h):  
            # Now, we iterate over ksi first, defining the accessible 
            # theta grid and calculating the D_RF_nobounce matrices for these points
            # 
  
            if abs(ksi0_val) > Trapksi0_h[l]:
                # Passing!
                # Then it is rather easy, all theta values are accessible
                passing = True      

                theta_grid_j_h = theta_h
                d_theta_grid_j_h = d_theta


                B0_h, _               = minmaxB(ptB_Int_at_psi, theta_grid_j_h)
                B_at_psi_j_h          = ptB[l, :]
                BR_at_psi_j_h         = ptBR[l, :]
                Bz_at_psi_j_h         = ptBz[l, :]
                R_axis_at_psi_j_h     = ptR[l, :] - Rp
                Z_axis_at_psi_j_h     = ptZ[l, :] - Zp

                # Calculate the internal factor of the bounce integral [t]
                CB_j_h = B_at_psi_j_h * (R_axis_at_psi_j_h**2 + Z_axis_at_psi_j_h**2)\
                  / (Rp * abs(BR_at_psi_j_h*Z_axis_at_psi_j_h - Bz_at_psi_j_h*R_axis_at_psi_j_h))# + eps)

                # And the B/B0 ratio [t]
                B_ratio_h = B_at_psi_j_h/B0_h
                #B_ratio_h = np.where(B_ratio_h < 1., 1. + eps, B_ratio_h)

                # Precalculate ksi over the theta grid [t]

                ksi_vals = np.sign(ksi0_val) * np.sqrt(1 - (B_ratio_h)*(1 - ksi0_val**2))
                ksi0_over_ksi_j_h = ksi0_val/ksi_vals
           
                # Calculate the lambda*q factor [t]
                lambda_q_h[l, j] = bounce_sum(d_theta_grid_j_h, CB_j_h, ksi0_over_ksi_j_h, passing, False)
                    
                # Define the integrands for the different bounce integrals [t]
                # See the notes for the derivation of these 
                DRF0_integrand = ksi0_over_ksi_j_h**2 * B_ratio_h
                if DKE_calc:
                    DRF0D_integrand = ksi0_over_ksi_j_h
                    DRF0F_integrand = (B_ratio_h -1) * ksi0_over_ksi_j_h**3

                # All is set up now, we just need to perform the expensive D_RF_nobounce calculation

                D_rf_lj_wh = np.zeros((len(theta_h), len(p_norm_w), len(n)))
                D_rf_lj_hh = np.zeros((len(theta_h), len(p_norm_h), len(n)))

                # No new theta grid
                Edens_lj_h = Edens_at_psi

                for n_idx, harm in enumerate(n):
                    # Split over the harmonics here because it will be important for the final quantities
                    for t, theta_val in enumerate(theta_grid_j_h):
                        if Edens_lj_h[t].max() > 0: # Only calculate if there is a beam present
                            D_rf_lj_wh[t, :, n_idx] = \
                                D_RF_nobounce(p_norm_w, ksi_vals[t], npar, nperp, \
                                    Edens_lj_h[t, :,:], Te_ref, \
                                        P[l, 0], X[l, t], R[l, t], L[l, t], S[l, t], harm, eps).T
                            
                            D_rf_lj_hh[t, :, n_idx] = \
                                D_RF_nobounce(p_norm_h, ksi_vals[t], npar, nperp, \
                                    Edens_lj_h[t, :,:], Te_ref, \
                                        P[l, 0], X[l, t], R[l, t], L[l, t], S[l, t], harm, eps).T
                        else:
                            D_rf_lj_wh[t, :, n_idx] = np.zeros_like(p_norm_w)
                            D_rf_lj_hh[t, :, n_idx] = np.zeros_like(p_norm_h)
            
                    # With the calculated values for all [t] at given psi, ksi_O,
                    # we can now calculate the bounce integrals
                    # As D_rf_lj is [t, i], we need to loop over the momentum values
                    # to calculate the bounce integral for each i
                    for i, _ in enumerate(p_norm_w):

                        DRF0_wh[l, i, j, n_idx] = bounce_sum(d_theta_grid_j_h, CB_j_h, DRF0_integrand*D_rf_lj_wh[:, i, n_idx] , passing, False)
                        if DKE_calc:
                            DRF0D_wh[l, i, j, n_idx] = np.sign(ksi0_val) * bounce_sum(d_theta_grid_j_h, CB_j_h, DRF0D_integrand*D_rf_lj_wh[:, i, n_idx] , passing, True)
                            DRF0F_wh[l, i, j, n_idx] = np.sign(ksi0_val) * bounce_sum(d_theta_grid_j_h, CB_j_h, DRF0F_integrand*D_rf_lj_wh[:, i, n_idx] , passing, True)

                    for i, _ in enumerate(p_norm_h):
                        DRF0_hh[l, i, j, n_idx] = bounce_sum(d_theta_grid_j_h, CB_j_h, DRF0_integrand*D_rf_lj_hh[:, i, n_idx] , passing, False)
                        if DKE_calc:
                            DRF0D_hh[l, i, j, n_idx] = np.sign(ksi0_val) * bounce_sum(d_theta_grid_j_h, CB_j_h, DRF0D_integrand*D_rf_lj_hh[:, i, n_idx] , passing, True)

                    # Finish off by adding the prefactor and the lambda*q factor
                    DRF0_wh[l, :, j, n_idx] *= C_RF_wh[:, j] / lambda_q_h[l, j] 
                    DRF0_hh[l, :, j, n_idx] *= C_RF_hh[:, j] / lambda_q_h[l, j] 
                    if DKE_calc:
                        DRF0D_wh[l, :, j, n_idx] *= C_RF_wh[:, j]  / lambda_q_h[l, j]
                        DRF0D_hh[l, :, j, n_idx] *= C_RF_hh[:, j] / lambda_q_h[l, j]
                    
                        DRF0F_wh[l, :, j, n_idx] *= C_RF_wh[:, j]  / lambda_q_h[l, j]


            else:
                # Trapped!
                passing = False
                

                # In this case, we have to shift to a different theta grid

                # The theta roots are the boundaries of the region where the particles are trapped
                theta_T_m, theta_T_M = theta_T_h[l, j]
                theta_w_aux= theta_w[(theta_w >= theta_T_m) & (theta_w <= theta_T_M)]
                # Add the boundaries to the whole grid
                theta_w_aux = np.concatenate(([theta_T_m], theta_w_aux, [theta_T_M]))

                d_theta_grid_j_h = np.diff(theta_w_aux) # The grid is now the full grid, so we can just take the dif

                #Update: Try to follow DKEp134 on the numerical integration, we need the half grid!
                # Otherwise we inevitably run into issues where B_ratio_h*(1-ksi_val**2) > 1

                theta_grid_j_h = theta_w_aux[:-1] + d_theta_grid_j_h/2

                if theta_grid_j_h[-1] > theta_h[-1]:
                    theta_grid_j_h[-1] = theta_h[-1]
                if theta_grid_j_h[0] < theta_h[0]:
                    theta_grid_j_h[0] = theta_h[0]

                # From here, most is the same as the passing case, but we have to interpolate the Wfct

                B0_h, _               = minmaxB(ptB_Int_at_psi, theta_h)
                B_at_psi_j_h          = interp1d(theta_h, ptB[l, :])(theta_grid_j_h)
                BR_at_psi_j_h         = interp1d(theta_h, ptBR[l, :])(theta_grid_j_h)
                Bz_at_psi_j_h         = interp1d(theta_h, ptBz[l, :])(theta_grid_j_h)
                R_axis_at_psi_j_h     = interp1d(theta_h, ptR[l, :])(theta_grid_j_h) - Rp
                Z_axis_at_psi_j_h     = interp1d(theta_h, ptZ[l, :])(theta_grid_j_h) - Zp

                # Get configuration quantities on the newly defined theta grid
                # Only the ones that depend on theta
                _, _, _, _, _, _, _, _, _, X_h, R_h, L_h, S_h = config_quantities([psi_l], theta_grid_j_h, omega, Eq)

                # Calculate the internal factor of the bounce integral [t]
                CB_j_h = B_at_psi_j_h * (R_axis_at_psi_j_h**2 + Z_axis_at_psi_j_h**2)\
                  / (Rp * abs(BR_at_psi_j_h*Z_axis_at_psi_j_h - Bz_at_psi_j_h*R_axis_at_psi_j_h))
                
                # And the B/B0 ratio [t]
                B_ratio_h = B_at_psi_j_h/B0_h
                #B_ratio_h = np.where(B_ratio_h < 1., 1. + eps, B_ratio_h)

                # Precalculate ksi over the theta grid [t]
                ksi_vals = np.sign(ksi0_val) * np.sqrt(1 - (B_ratio_h)*(1 - ksi0_val**2))
                ksi0_over_ksi_j_h = ksi0_val/ksi_vals

                # Calculate the lambda*q factor [t]
                lambda_q_h[l, j] = bounce_sum(d_theta_grid_j_h, CB_j_h, ksi0_over_ksi_j_h, passing, False)

                # Define the integrands for the different bounce integrals [t]
                # See the notes for the derivation of these
                DRF0_integrand = ksi0_over_ksi_j_h**2 * B_ratio_h

                # All is set up now, we just need to perform the expensive D_RF_nobounce calculation

                D_rf_lj_wh = np.zeros((len(theta_grid_j_h), len(p_norm_w), len(n)))
                D_rf_lj_hh = np.zeros((len(theta_grid_j_h), len(p_norm_h), len(n)))

                # Interpolate the Wfct to the new theta grid
                # We need to meshgrid the theta and npar, nperp grids
                Theta_grid_j_h, Npar_grid_j_h, Nperp_grid_j_h = np.meshgrid(theta_grid_j_h, npar, nperp, indexing='ij')
                Edens_interp_lj_h = Edens_Int_at_psi((Theta_grid_j_h, Npar_grid_j_h, Nperp_grid_j_h))

                for n_idx, harm in enumerate(n):
                    # Split over the harmonics here because it will be important for the final quantities
                    for t, theta_val in enumerate(theta_grid_j_h):
                        if Edens_interp_lj_h[t].max() > 0: # Only calculate if there is a beam present

                            D_rf_lj_wh[t, :, n_idx] = \
                                D_RF_nobounce(p_norm_w, ksi_vals[t], npar, nperp, \
                                    Edens_interp_lj_h[t, :, :], Te_ref, \
                                        P[l, 0], X_h[0, t], R_h[0, t], L_h[0, t], S_h[0, t], harm, eps).T

                            D_rf_lj_hh[t, :, n_idx] = \
                                D_RF_nobounce(p_norm_h, ksi_vals[t], npar, nperp, \
                                    Edens_interp_lj_h[t, :, :], Te_ref, \
                                        P[l, 0], X_h[0, t], R_h[0, t], L_h[0, t], S_h[0, t], harm, eps).T
                        else:
                            D_rf_lj_wh[t, :, n_idx] = np.zeros_like(p_norm_w)
                            D_rf_lj_hh[t, :, n_idx] = np.zeros_like(p_norm_h)
                        
                    # With the calculated values for all [t] at given psi, ksi_O,
                    # we can now calculate the bounce integrals
                    # As D_rf_lj is [t, i], we need to loop over the momentum values

                    for i, _ in enumerate(p_norm_w):

                        DRF0_wh[l, i, j, n_idx] = bounce_sum(d_theta_grid_j_h, CB_j_h, DRF0_integrand*D_rf_lj_wh[:, i, n_idx] , passing, False)
                        if DKE_calc:
                            DRF0D_wh[l, i, j, n_idx] = np.sign(ksi0_val) * bounce_sum(d_theta_grid_j_h, CB_j_h, DRF0D_integrand*D_rf_lj_wh[:, i, n_idx] , passing, True)
                            DRF0F_wh[l, i, j, n_idx] = np.sign(ksi0_val) * bounce_sum(d_theta_grid_j_h, CB_j_h, DRF0F_integrand*D_rf_lj_wh[:, i, n_idx] , passing, True)

                    for i, _ in enumerate(p_norm_h):

                        DRF0_hh[l, i, j, n_idx] = bounce_sum(d_theta_grid_j_h, CB_j_h, DRF0_integrand*D_rf_lj_hh[:, i, n_idx] , passing, False)
                        if DKE_calc:
                            DRF0D_hh[l, i, j, n_idx] = np.sign(ksi0_val) * bounce_sum(d_theta_grid_j_h, CB_j_h, DRF0D_integrand*D_rf_lj_hh[:, i, n_idx] , passing, True)

                    # Finish off by adding the prefactor and the lambda*q factor
                    DRF0_wh[l, :, j, n_idx] *= C_RF_wh[:, j] / lambda_q_h[l, j]
                    DRF0_hh[l, :, j, n_idx] *= C_RF_hh[:, j] / lambda_q_h[l, j]
                    if DKE_calc:
                        DRF0D_wh[l, :, j, n_idx] *= C_RF_wh[:, j] / lambda_q_h[l, j]
                        DRF0D_hh[l, :, j, n_idx] *= C_RF_hh[:, j] / lambda_q_h[l, j]

                        DRF0F_wh[l, :, j, n_idx] *= C_RF_wh[:, j] / lambda_q_h[l, j]
                
                
        #--------------------------------#
        #---ksi0 whole grid calculation--#
        #--------------------------------#
        for j, ksi0_val in enumerate(ksi0_w):
            if abs(ksi0_val) > Trapksi0_w[l]:

                passing = True

                theta_grid_j_w = theta_h
                d_theta_grid_j_w = d_theta


                B0_w, _               = minmaxB(ptB_Int_at_psi, theta_grid_j_w)
                B_at_psi_j_w          = ptB[l, :]
                BR_at_psi_j_w         = ptBR[l, :]
                Bz_at_psi_j_w         = ptBz[l, :]
                R_axis_at_psi_j_w     = ptR[l, :] - Rp
                Z_axis_at_psi_j_w     = ptZ[l, :] - Zp

                # Calculate the internal factor of the bounce integral [t]
                CB_j_w = B_at_psi_j_w * (R_axis_at_psi_j_w**2 + Z_axis_at_psi_j_w**2)\
                  / (Rp * abs(BR_at_psi_j_w*Z_axis_at_psi_j_w - Bz_at_psi_j_w*R_axis_at_psi_j_w))# + eps)
            
                # And the B/B0 ratio [t]
                B_ratio_w = B_at_psi_j_w/B0_w
                #B_ratio_w = np.where(B_ratio_w < 1., 1. + eps, B_ratio_w)

                # Precalculate ksi over the theta grid [t]

                ksi_vals = np.sign(ksi0_val) * np.sqrt(1 - (B_ratio_w)*(1 - ksi0_val**2))
                ksi0_over_ksi_j_w = ksi0_val/ksi_vals

                # Calculate the lambda*q factor [t]
                lambda_q_w[l, j] = bounce_sum(d_theta_grid_j_w, CB_j_w, ksi0_over_ksi_j_w, passing, False)
                    
                # Define the integrands for the different bounce integrals [t]
                # See the notes for the derivation of these 
                DRF0_integrand = ksi0_over_ksi_j_w**2 * B_ratio_w
                if DKE_calc:
                    DRF0D_integrand = ksi0_over_ksi_j_w
                    DRF0F_integrand = (B_ratio_w -1) * ksi0_over_ksi_j_w**3

                # All is set up now, we just need to perform the expensive D_RF_nobounce calculation

                D_rf_lj_hw = np.zeros((len(theta_h), len(p_norm_h), len(n)))

                Edens_lj_w = Edens_at_psi

                for n_idx, harm in enumerate(n):
                    # Split over the harmonics here because it will be important for the final quantities
                    for t, theta_val in enumerate(theta_grid_j_w):
                        if Edens_lj_w[t].max() > 0: # Only calculate if there is a beam present
                            D_rf_lj_hw[t, :, n_idx] = \
                                D_RF_nobounce(p_norm_h, ksi_vals[t], npar, nperp, \
                                    Edens_lj_w[t,:,:], Te_ref, \
                                        P[l, 0], X[l, t], R[l, t], L[l, t], S[l, t], harm, eps).T
                        else:
                            D_rf_lj_hw[t, :, n_idx] = np.zeros_like(p_norm_h)

                    # With the calculated values for all [t] at given psi, ksi_O,
                    # we can now calculate the bounce integrals
                    # As D_rf_lj is [t, i], we need to loop over the momentum values
                    # to calculate the bounce integral for each i
                    for i, _ in enumerate(p_norm_h):

                        DRF0_hw[l, i, j, n_idx] = bounce_sum(d_theta_grid_j_w, CB_j_w, DRF0_integrand*D_rf_lj_hw[:, i, n_idx] , passing, False)
                        if DKE_calc:
                            DRF0D_hw[l, i, j, n_idx] = np.sign(ksi0_val) * bounce_sum(d_theta_grid_j_w, CB_j_w, DRF0D_integrand*D_rf_lj_hw[:, i, n_idx] , passing, True)
                            DRF0F_hw[l, i, j, n_idx] = np.sign(ksi0_val) * bounce_sum(d_theta_grid_j_w, CB_j_w, DRF0F_integrand*D_rf_lj_hw[:, i, n_idx] , passing, True)

                    # Finish off by adding the prefactor and the lambda*q factor
                    DRF0_hw[l, :, j, n_idx] *= C_RF_hw[:, j] / lambda_q_w[l, j]
                    if DKE_calc:
                        DRF0D_hw[l, :, j, n_idx] *= C_RF_hw[:, j] / lambda_q_w[l, j]
                        DRF0F_hw[l, :, j, n_idx] *= C_RF_hw[:, j] / lambda_q_w[l, j]


            else:
                # Trapped!
                passing = False
                
                # In this case, we have to shift to a different theta grid

                # The theta roots are the boundaries of the region where the particles are trapped
                theta_T_m, theta_T_M = theta_T_w[l, j]
                theta_w_aux= theta_w[(theta_w >= theta_T_m) & (theta_w <= theta_T_M)]
                # Add the boundaries to the grid
                theta_w_aux = np.concatenate(([theta_T_m], theta_w_aux, [theta_T_M]))


                d_theta_grid_j_w = np.diff(theta_w_aux) # The grid is now the full grid, so we can just take the dif

                #Update: Try to follow DKEp134 on the numerical integration, we need the half grid!
                # Otherwise we inevitably run into issues where B_ratio_h*(1-ksi_val**2) > 1

                theta_grid_j_w = theta_w_aux[:-1] + d_theta_grid_j_w/2
                if theta_grid_j_w[-1] > theta_h[-1]:
                    theta_grid_j_w[-1] = theta_h[-1]
                if theta_grid_j_w[0] < theta_h[0]:
                    theta_grid_j_w[0] = theta_h[0]
     

                # From here, most is the same as the passing case, but we have to interpolate the Wfct

                B0_w, _               = minmaxB(ptB_Int_at_psi, theta_h)
                B_at_psi_j_w          = interp1d(theta_h, ptB[l, :])(theta_grid_j_w)
                BR_at_psi_j_w         = interp1d(theta_h, ptBR[l, :])(theta_grid_j_w)
                Bz_at_psi_j_w         = interp1d(theta_h, ptBz[l, :])(theta_grid_j_w)
                R_axis_at_psi_j_w     = interp1d(theta_h, ptR[l, :])(theta_grid_j_w) - Rp
                Z_axis_at_psi_j_w     = interp1d(theta_h, ptZ[l, :])(theta_grid_j_w) - Zp

                # Get configuration quantities on the newly defined theta grid
                # Only the ones that depend on theta
                _, _, _, _, _, _, _, _, _, X_w, R_w, L_w, S_w = config_quantities([psi_l], theta_grid_j_w, omega, Eq)


                # Calculate the internal factor of the bounce integral [t]
                CB_j_w = B_at_psi_j_w * (R_axis_at_psi_j_w**2 + Z_axis_at_psi_j_w**2)\
                  / (Rp * abs(BR_at_psi_j_w*Z_axis_at_psi_j_w - Bz_at_psi_j_w*R_axis_at_psi_j_w))
                
                # And the B/B0 ratio [t]
                B_ratio_w = B_at_psi_j_w/B0_w
                #B_ratio_w = np.where(B_ratio_w < 1., 1. + eps, B_ratio_w)

                # Precalculate ksi over the theta grid [t]
                ksi_vals = np.sign(ksi0_val) * np.sqrt(1 - (B_ratio_w)*(1 - ksi0_val**2))
                ksi0_over_ksi_j_w = ksi0_val/ksi_vals

                # Calculate the lambda*q factor [t]
                lambda_q_w[l, j] = bounce_sum(d_theta_grid_j_w, CB_j_w, ksi0_over_ksi_j_w, passing, False)

                # Define the integrands for the different bounce integrals [t]
                # See the notes for the derivation of these
                DRF0_integrand = ksi0_over_ksi_j_w**2 * B_ratio_w

                # All is set up now, we just need to perform the expensive D_RF_nobounce calculation

                D_rf_lj_hw = np.zeros((len(theta_grid_j_w), len(p_norm_h), len(n)))

                # Interpolate the Wfct to the new theta grid
                # We need to meshgrid the theta and npar, nperp grids
                Theta_grid_j_w, Npar_grid_j_w, Nperp_grid_j_w = np.meshgrid(theta_grid_j_w, npar, nperp, indexing='ij')
                Edens_interp_lj_w = Edens_Int_at_psi((Theta_grid_j_w, Npar_grid_j_w, Nperp_grid_j_w))


                for n_idx, harm in enumerate(n):
                    # Split over the harmonics here because it will be important for the final quantities
                    for t, theta_val in enumerate(theta_grid_j_w):
                        if Edens_interp_lj_w[t].max() > 0: # Only calculate if there is a beam present

                            D_rf_lj_hw[t, :, n_idx] = \
                                D_RF_nobounce(p_norm_h, ksi_vals[t], npar, nperp, \
                                    Edens_interp_lj_w[t, :, :], Te_ref, \
                                        P[l, 0], X_w[0, t], R_w[0, t], L_w[0, t], S_w[0, t], harm, eps).T
                        else:
                            D_rf_lj_hw[t, :, n_idx] = np.zeros_like(p_norm_h)
                        
                    # With the calculated values for all [t] at given psi, ksi_O,
                    # we can now calculate the bounce integrals
                    # As D_rf_lj is [t, i], we need to loop over the momentum values

                    for i, _ in enumerate(p_norm_h):

                        DRF0_hw[l, i, j, n_idx] = bounce_sum(d_theta_grid_j_w, CB_j_w, DRF0_integrand*D_rf_lj_hw[:, i, n_idx] , passing, False)
                        if DKE_calc:
                            DRF0D_hw[l, i, j, n_idx] = np.sign(ksi0_val) * bounce_sum(d_theta_grid_j_w, CB_j_w, DRF0D_integrand*D_rf_lj_hw[:, i, n_idx] , passing, True)
                            DRF0F_hw[l, i, j, n_idx] = np.sign(ksi0_val) * bounce_sum(d_theta_grid_j_w, CB_j_w, DRF0F_integrand*D_rf_lj_hw[:, i, n_idx] , passing, True)
                    
                    # Finish off by adding the prefactor and the lambda*q factor
                    DRF0_hw[l, :, j, n_idx] *= C_RF_hw[:, j] / lambda_q_w[l, j]
                    if DKE_calc:
                        DRF0D_hw[l, :, j, n_idx] *= C_RF_hw[:, j] / lambda_q_w[l, j]
                        DRF0F_hw[l, :, j, n_idx] *= C_RF_hw[:, j] / lambda_q_w[l, j]

        #--------------------------------#
        #---Gaussian filter added as test#
        from scipy.ndimage import gaussian_filter
        sigma = 2
        for n_idx, harm in enumerate(n):
            DRF0_wh[l, :, :, n_idx] = gaussian_filter(DRF0_wh[l, :, :, n_idx], sigma=sigma)
            DRF0_hh[l, :, :, n_idx] = gaussian_filter(DRF0_hh[l, :, :, n_idx], sigma=sigma)
            DRF0_hw[l, :, :, n_idx] = gaussian_filter(DRF0_hw[l, :, :, n_idx], sigma=sigma)

            if DKE_calc:
                DRF0D_wh[l, :, :, n_idx] = gaussian_filter(DRF0D_wh[l, :, :, n_idx], sigma=sigma)
                DRF0D_hh[l, :, :, n_idx] = gaussian_filter(DRF0D_hh[l, :, :, n_idx], sigma=sigma)
                DRF0D_hw[l, :, :, n_idx] = gaussian_filter(DRF0D_hw[l, :, :, n_idx], sigma=sigma)

                DRF0F_wh[l, :, :, n_idx] = gaussian_filter(DRF0F_wh[l, :, :, n_idx], sigma=sigma)
                DRF0F_hw[l, :, :, n_idx] = gaussian_filter(DRF0F_hw[l, :, :, n_idx], sigma=sigma)  
                
    return DRF0_wh, DRF0D_wh, DRF0F_wh, DRF0_hw, DRF0D_hw, DRF0F_hw, DRF0_hh, DRF0D_hh, Trapksi0_h, Trapksi0_w

#-------------------------------#
#---End of function definitions---#
#-------------------------------#