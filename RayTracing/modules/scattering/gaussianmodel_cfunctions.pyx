"""
Some basic functions called from ScatteringPlasmaModel are defined.
"""



##############################################################################
# IMPORT STATEMENTS
##############################################################################

# Load standard modules
import numpy as np
cimport numpy as np
from cpython cimport bool
from libc.stdlib cimport malloc
# Load local modules
import RayTracing.modules.dispersion_matrix_cfunctions as disp

cdef extern from "math.h":
    double sqrt( double )
    double exp( double ) 
    double sin( double )
    double cos( double )
    bint isnan( double )

# constants
cdef double pi = 3.141592654   
cdef double emass = 9.11*1e-31            # electron mass in kg
cdef double echarge = 1.602*1e-19         # electron charge in C
cdef double epsilon0 = 8.85*1e-12         # dielectric constant in V/Cm


# and some global variables
cdef double gk0
cdef double gomega
cdef double gepsilonRegS
cdef int gnmbrinitialisationMetropolisHastingsScattering

## test -
## parallel and perpendicular correlation lengths are no longer global
#cdef double gxiParallel, gxiPerp
## end test -

##############################################################################
# DEFINE INITIALISATION FUNCTION FOR THE GLOBAL VARIABLES
##############################################################################
# shape of the fluctuations
## test -
## gxiParallel and gxiPerp are no longer global
cpdef  scatteringinitialiseGlobalVariables(double k0, double omega, double epsilonRegS, int nmbrinitialisationMetropolisHastingsScattering, bool freeze_seed, int rank): # SIGMA AS THIRD VARIABLE REMOVED
#cpdef  scatteringinitialiseGlobalVariables(double k0, double omega, double epsilonRegS, double xiParallel, double xiPerp, int nmbrinitialisationMetropolisHastingsScattering, bool freeze_seed, int rank): # SIGMA AS THIRD VARIABLE REMOVED
## end test -

    # refer to global variables
    global gk0, gomega, gepsilonRegS 
    global gnmbrinitialisationMetropolisHastingsScattering
    global random_state
## test - xi variables no longer global    
##    global gxiParallel, gxiPerp

    # # copy arguments to the global variables
    # gxiParallel = xiParallel
    # gxiPerp = xiPerp
## end test - 

    gk0 = k0
    gomega = omega
    gepsilonRegS = epsilonRegS
    gnmbrinitialisationMetropolisHastingsScattering = nmbrinitialisationMetropolisHastingsScattering
    
    # random number generator
    if freeze_seed:
        random_state = np.random.RandomState(rank+1)
    else:
        randomseed = np.random.randint(1234)
        random_state = np.random.RandomState((rank+1)*randomseed)
        
    # return from function
    return 






##############################################################################
# THIS FUNCTIONS ARE ALSO REPEATED IN THE DISPERSION MATRIX MODULE.
##############################################################################
cpdef inline double disParamOmega(double B):
    return B*echarge / emass   
        
cpdef inline double disParamomegaP(double Ne):
    return echarge*sqrt(Ne*1e19/epsilon0/emass)

cpdef inline double disParamS(double omega, double B, double Ne):
    return 1. - disParamomegaP(Ne)**2 / (omega**2 - disParamOmega(B)**2)

cpdef inline double disParamP(double omega, double B, double Ne):
    return 1. - disParamomegaP(Ne)**2 / omega**2

cpdef inline double disParamD(double omega, double B, double Ne):
    return disParamOmega(B) / (omega**2 - disParamOmega(B)**2) * disParamomegaP(Ne)**2 / omega

cpdef inline double disParamdS_dne(double omega, double B, double Ne):
    return -1./(omega**2-B**2*echarge**2/emass**2)*echarge**2/epsilon0/emass*1e19

cpdef inline double disParamdP_dne(double omega, double B, double Ne):
    return -1./(omega**2)*echarge**2/epsilon0/emass*1e19

cpdef inline double disParamdD_dne(double omega, double B, double Ne):
    return (1./omega/(omega**2-B**2*echarge**2/emass**2))*B*echarge/emass*echarge**2/epsilon0/emass*1e19


# fct that returns the Nperp-component such that the refractive index vector
# fullfills the dispersion relation.
# arguments:
# beam omega in rad/s
# magnetic field norm in T
# electron density in 1e-19 / m^3
# Nparallel component of the refractive index vector
# mode sigma
# regularisation parameter for the inversion of S
cpdef double disNperp(double Bnorm, double Ne, double Nparallel, double sigma):
                                         
    """Calculate the perpendicular wavevector component for a given Nparallel at (R,z) for mode alpha
    such that the dispersion relation is fullfilled.
    For propagation perpendicular to the magnetic field sigma=+1 corresponds to the
    O-mode, sigma=-1 to the X-mode.
    """
    global gomega, gepsilonRegS

    cdef S = disParamS(gomega, Bnorm, Ne)
    cdef P = disParamP(gomega, Bnorm, Ne)
    cdef D = disParamD(gomega, Bnorm, Ne)
    cdef Omega = disParamOmega(Bnorm)

    B = (S+P)*Nparallel**2 \
        - S*P + D**2 - S**2
    C = (S**2-D**2)*P - 2.*S*P*Nparallel**2 + P*Nparallel**4
    SregInv = S / (S**2 + gepsilonRegS**2)
    
    return sqrt((-B+sigma*D*sqrt(4.*P*Nparallel**2+Omega**2/gomega**2*(Nparallel**2-1.)**2)) / 2. * SregInv)
              



##############################################################################
# DEFINE SOME BASIC FUNCTIONS
##############################################################################




# normalisation constant for eigenvectors
cpdef double NormalisationEigenvectors(double Bnorm, double Ne,             # plasma parameters
                                       double Nparallel, double Nperp):   # refractive index components
                                  
    """Computes the normalisation constant squared for the eigenvectors."""

    global gomega

    cdef double S = disParamS(gomega,Bnorm,Ne)
    cdef double P = disParamP(gomega,Bnorm,Ne)
    cdef double D = disParamD(gomega,Bnorm,Ne)    

    return (Nparallel**2+Nperp**2-S)**2*(Nperp**2-P)**2 + D**2*(Nperp**2-P)**2 \
            + Nparallel**2*Nperp**2*(Nparallel**2+Nperp**2-S)**2






# correction factor f
cpdef double discorrectionFactorf(double omega, 
                                  double Bnorm, double Ne, 
                                  double Nparallel, double sigma):
    

    return 4.*disp.disTrDispersionMatrixDivBySHamiltonian(omega,Bnorm,Ne,Nparallel,sigma)




# probability distribution for scattering
## test - xiPerp and xiParallel no longer global
cpdef ScatteringProbabilityDistributionWithoutShape(double phiNprime, double Nparallelprime,
                                                    double Bnorm, double Ne, 
                                                    double rho, double theta,          # tokamak coordinates
                                                    double Nparallel, double Nperp,
                                                    double phiN, double sigma,
                                                    double xiParallel, double xiPerp):

# cpdef ScatteringProbabilityDistributionWithoutShape(double phiNprime, double Nparallelprime,
#                                                     double Bnorm, double Ne, 
#                                                     double rho, double theta,          # tokamak coordinates
#                                                     double Nparallel, double Nperp,
#                                                     double phiN, double sigma):
## end test -


    """Computes the probability distribution for the new refractive index 
    components.
    See theory sheets 'analytical model for density fluctuations in a tokamak' 
    on page 5 for details on computations."""


## test - xiPerp and xiParallel no longer global
    global gk0
##    global gk0, gxiPerp, gxiParallel
## end test - 
    global gomega

    cdef double S = disParamS(gomega,Bnorm,Ne)
    cdef double P = disParamP(gomega,Bnorm,Ne)
    cdef double D = disParamD(gomega,Bnorm,Ne)    
    cdef double dS_dne = disParamdS_dne(gomega,Bnorm,Ne)
    cdef double dP_dne = disParamdP_dne(gomega,Bnorm,Ne)
    cdef double dD_dne = disParamdD_dne(gomega,Bnorm,Ne)

    # compute the perpendicular refractive index component
    cdef double Nperpprime = disNperp(Bnorm, Ne, Nparallelprime, sigma)



    cdef double costheta = Nparallel / sqrt(Nparallel**2 + Nperp**2) 
    cdef double sintheta = sqrt(1. - costheta**2)
    cdef double costhetaprime = Nparallelprime / sqrt(Nparallelprime**2 + Nperpprime**2) 
    cdef double sinthetaprime = sqrt(1. - costhetaprime**2)
    
    cdef double X = disParamomegaP(Ne)**2 / gomega**2
    cdef double Y = disParamOmega(Bnorm) / gomega

    cdef double Delta = 0.5*sqrt(Y**4*sintheta**4 + (1.-X)**2*costheta**2)
    cdef double Deltaprime = 0.5*sqrt(Y**4*sinthetaprime**4 + (1.-X)**2*costhetaprime**2)
    
    cdef double T
    cdef double Tprime
    cdef double L
    cdef double Lprime   

    # in case Nparallel is very small, sometimes, the computation crashes.
    # then, the limit of purely perpendicular propagation  
    # is appropriate
    try:
        T = Y*(1.-X)*costheta / (0.5*Y**2*sintheta**2 - sigma*Delta)
        L = X*Y*sintheta/(1.-X) * T / (T-Y*costheta)
    except:
        if sigma == +1.:
            T = 10000.
            L = X*Y*sintheta/(1.-X)
        else:
            T = 0.
            L = X*Y*sintheta / (1-X-Y**2+X*Y**2*costheta**2)         

    try:
        Tprime = Y*(1.-X)*costhetaprime / (0.5*Y**2*sinthetaprime**2 - sigma*Deltaprime)
        Lprime = X*Y*sinthetaprime/(1.-X) * Tprime / (Tprime-Y*costhetaprime)
    except:
        if sigma == +1.:
            Tprime = 10000.
            Lprime = X*Y*sinthetaprime/(1.-X)
        else:
            Tprime = 0.
            Lprime = X*Y*sinthetaprime / (1-X-Y**2+X*Y**2*costhetaprime**2)

    cdef double norm = L**2 + T**2 + 1.
    cdef double normprime = Lprime**2 + Tprime**2 + 1.
    #cdef double curlybracketssquared = (L*Lprime*(sintheta*sinthetaprime*dS_dne + costheta*costhetaprime*dP_dne) \
    #                           + L*Tprime*(sintheta*costhetaprime*dS_dne + costheta*sinthetaprime*dP_dne) \
    #                           + T*Lprime*(costheta*sinthetaprime*dS_dne + sintheta*costhetaprime*dP_dne) \
    #                           + T*Tprime*(costheta*costhetaprime*dS_dne + sintheta*sinthetaprime*dP_dne) \
    #                           - L*sintheta*dD_dne + Lprime*sinthetaprime*dD_dne \
    #                           - T*costheta*dD_dne + Tprime*costhetaprime*dD_dne \
    #                           - dS_dne)**2 


    cdef double curlybracketssquared = (L*Lprime*(-sintheta*sinthetaprime*dS_dne - costheta*costhetaprime*dP_dne) \
                               - L*Tprime*(sintheta*costhetaprime*dS_dne - costheta*sinthetaprime*dP_dne) \
                               - T*Lprime*(costheta*sinthetaprime*dS_dne - sintheta*costhetaprime*dP_dne) \
                               - T*Tprime*(costheta*costhetaprime*dS_dne + sintheta*sinthetaprime*dP_dne) \
                               - L*sintheta*dD_dne + Lprime*sinthetaprime*dD_dne \
                               - T*costheta*dD_dne + Tprime*costhetaprime*dD_dne \
                               - dS_dne)**2 


## test - xiPerp and xiParallel no longer global    
    cdef double Gamma = (gk0/sqrt(2.*pi))**3/sqrt(xiParallel*xiPerp**2)  \
                        * exp(-0.5*gk0**2 \
                        *((Nparallel-Nparallelprime)**2/xiParallel \
                        + (Nperp*cos(phiN)-Nperpprime*cos(phiNprime))**2/xiPerp \
                        + (Nperp*sin(phiN)-Nperpprime*sin(phiNprime))**2/xiPerp))

    # cdef double Gamma = (gk0/sqrt(2.*pi))**3/sqrt(gxiParallel*gxiPerp**2)  \
    #                     * exp(-0.5*gk0**2 \
    #                     *((Nparallel-Nparallelprime)**2/gxiParallel \
    #                     + (Nperp*cos(phiN)-Nperpprime*cos(phiNprime))**2/gxiPerp \
    #                     + (Nperp*sin(phiN)-Nperpprime*sin(phiNprime))**2/gxiPerp))
## end test -    


    cdef correctionfactorprime = abs(discorrectionFactorf(gomega, Bnorm, Ne, Nparallelprime, sigma))

    return pi/2.*gk0*Ne**2 * Gamma * curlybracketssquared / norm / normprime * correctionfactorprime




# probability distribution for scattering
## test - xiPerp and xiParallel no longer global    
cpdef ScatteringProbabilityDistributionToDifferentModeWithoutShape(double phiNprime, double Nparallelprime,
                                                                   double Bnorm, double Ne, 
                                                                   double rho, double theta,    # tokamak coordinates
                                                                   double Nparallel, double Nperp,
                                                                   double phiN, double sigma,
                                                                   double xiParallel, double xiPerp):

# cpdef ScatteringProbabilityDistributionToDifferentModeWithoutShape(double phiNprime, double Nparallelprime,
#                                                                    double Bnorm, double Ne, 
#                                                                    double rho, double theta,    # tokamak coordinates
#                                                                    double Nparallel, double Nperp,
#                                                                    double phiN, double sigma):
## end test -


    """Computes the probability distribution for the new refractive index 
    components.
    See theory sheets 'analytical model for density fluctuations in a tokamak' 
    on page 5 for details on computations."""


## test - xiPerp and xiParallel no longer global    
    global gk0
    # global gk0, gxiPerp, gxiParallel
## end test -    
    global gomega

    cdef double S = disParamS(gomega,Bnorm,Ne)
    cdef double P = disParamP(gomega,Bnorm,Ne)
    cdef double D = disParamD(gomega,Bnorm,Ne)    
    cdef double dS_dne = disParamdS_dne(gomega,Bnorm,Ne)
    cdef double dP_dne = disParamdP_dne(gomega,Bnorm,Ne)
    cdef double dD_dne = disParamdD_dne(gomega,Bnorm,Ne)

    # compute the perpendicular refractive index component
    cdef double Nperpprime = disNperp(Bnorm, Ne, Nparallelprime, -sigma)
    # if no solution is found, the other mode is not a solution at the present point,
    # so just return 0 indicating that no mode-to-mode scattering takes place
    if isnan(Nperpprime) == True:
        return 0.

    cdef double costheta = Nparallel / sqrt(Nparallel**2 + Nperp**2) 
    cdef double sintheta = sqrt(1. - costheta**2)
    cdef double costhetaprime = Nparallelprime / sqrt(Nparallelprime**2 + Nperpprime**2) 
    cdef double sinthetaprime = sqrt(1. - costhetaprime**2)
    
    cdef double X = disParamomegaP(Ne)**2 / gomega**2
    cdef double Y = disParamOmega(Bnorm) / gomega
    
    cdef double Delta = 0.5*sqrt(Y**4*sintheta**4 + (1.-X)**2*costheta**2)
    cdef double Deltaprime = 0.5*sqrt(Y**4*sinthetaprime**4 + (1.-X)**2*costhetaprime**2)
    
    cdef double T
    cdef double Tprime
    cdef double L
    cdef double Lprime     

    # in case Nparallel is very small, sometimes, the computation crashes.
    # then, the limit of purely perpendicular propagation  
    # is appropriate
    try:
        T = Y*(1.-X)*costheta / (0.5*Y**2*sintheta**2 - sigma*Delta)
        L = X*Y*sintheta/(1.-X) * T / (T-Y*costheta)
    except:
        if sigma == +1.:
            T = 10000.
            L = X*Y*sintheta/(1.-X)
        else:
            T = 0.
            L = X*Y*sintheta / (1-X-Y**2+X*Y**2*costheta**2)
        
    try:
        Tprime = Y*(1.-X)*costhetaprime / (0.5*Y**2*sinthetaprime**2 + sigma*Deltaprime)
        Lprime = X*Y*sinthetaprime/(1.-X) * Tprime / (Tprime-Y*costhetaprime)
    except:
        if sigma == -1.:
            Tprime = 10000.
            Lprime = X*Y*sinthetaprime/(1.-X)
        else:
            Tprime = 0.
            Lprime = X*Y*sinthetaprime / (1-X-Y**2+X*Y**2*costhetaprime**2)

    cdef double norm = L**2 + T**2 + 1.
    cdef double normprime = Lprime**2 + Tprime**2 + 1.
    #cdef double curlybracketssquared = (L*Lprime*(sintheta*sinthetaprime*dS_dne + costheta*costhetaprime*dP_dne) \
    #                           + L*Tprime*(sintheta*costhetaprime*dS_dne + costheta*sinthetaprime*dP_dne) \
    #                           + T*Lprime*(costheta*sinthetaprime*dS_dne + sintheta*costhetaprime*dP_dne) \
    #                           + T*Tprime*(costheta*costhetaprime*dS_dne + sintheta*sinthetaprime*dP_dne) \
    #                           - L*sintheta*dD_dne + Lprime*sinthetaprime*dD_dne \
    #                           - T*costheta*dD_dne + Tprime*costhetaprime*dD_dne \
    #                           - dS_dne)**2 
    
    cdef double curlybracketssquared = (L*Lprime*(-sintheta*sinthetaprime*dS_dne - costheta*costhetaprime*dP_dne) \
                               - L*Tprime*(sintheta*costhetaprime*dS_dne - costheta*sinthetaprime*dP_dne) \
                               - T*Lprime*(costheta*sinthetaprime*dS_dne - sintheta*costhetaprime*dP_dne) \
                               - T*Tprime*(costheta*costhetaprime*dS_dne + sintheta*sinthetaprime*dP_dne) \
                               - L*sintheta*dD_dne + Lprime*sinthetaprime*dD_dne \
                               - T*costheta*dD_dne + Tprime*costhetaprime*dD_dne \
                               - dS_dne)**2 

## test - xiPerp and xiParallel no longer global    
    cdef double Gamma = (gk0/sqrt(2.*pi))**3/sqrt(xiParallel*xiPerp**2)  \
                        * exp(-0.5*gk0**2 \
                        * ((Nparallel-Nparallelprime)**2/xiParallel \
                        + (Nperp*cos(phiN)-Nperpprime*cos(phiNprime))**2/xiPerp \
                        + (Nperp*sin(phiN)-Nperpprime*sin(phiNprime))**2/xiPerp))

    # cdef double Gamma = (gk0/sqrt(2.*pi))**3/sqrt(gxiParallel*gxiPerp**2)  \
    #                     * exp(-0.5*gk0**2 \
    #                     * ((Nparallel-Nparallelprime)**2/gxiParallel \
    #                     + (Nperp*cos(phiN)-Nperpprime*cos(phiNprime))**2/gxiPerp \
    #                     + (Nperp*sin(phiN)-Nperpprime*sin(phiNprime))**2/gxiPerp))
## end test -

    cdef correctionfactorprime = abs(discorrectionFactorf(gomega, Bnorm, Ne, Nparallelprime, -sigma))


    return pi/2.*gk0*Ne**2 * Gamma * curlybracketssquared / norm / normprime * correctionfactorprime











# uses Metropolis-Hastings algorithm in order to pick a new set of refractive index 
# vector components
# returns: Nparallel, Nperp, phiN

## test - xiPerp and xiParallel no longer global    
cpdef np.ndarray [double, ndim=1] ScatteringChooseRefractiveIndex(double Nparallel, double Nperp, double phiN, 
                                                                  double Bnorm, double Ne, 
                                                                  double rho, double theta, double sigma,
                                                                  double target_sigma,
                                                                  double xiParallel, double xiPerp):

# cpdef np.ndarray [double, ndim=1] ScatteringChooseRefractiveIndex(double Nparallel, double Nperp, double phiN, 
#                                                                   double Bnorm, double Ne, 
#                                                                   double rho, double theta, double sigma,
#                                                                   double target_sigma):
## end test -

    # refer to global variables
    
## test - xiPerp and xiParallel no longer global    
    global gk0
    # global gk0, gxiPerp, gxiParallel
## end test -    
    global gnmbrinitialisationMetropolisHastingsScattering


    # define some variables
    cdef double oldNparallel, oldphiN, oldNperp
    cdef double newNparallel, newphiN

    cdef double probaccept, randomvariable

    # initialise old variables with recent values (most likely to occur)
    oldNparallel = Nparallel
    oldphiN = phiN
    oldNperp = disNperp(Bnorm, Ne, oldNparallel, sigma)

    # generate some random numbers using the Metropolis-Hastings algorithm such that it co
    # converges to the correct probability distribution
    for i in range(0,gnmbrinitialisationMetropolisHastingsScattering):
  
        # generate new values using gaussian probability distribution
## test -        
        newNparallel = random_state.normal(Nparallel,sqrt(xiParallel)/gk0 )
        newphiN = phiN + random_state.normal(0.,sqrt(xiPerp)/gk0/Nperp)

        # newNparallel = random_state.normal(Nparallel,sqrt(gxiParallel)/gk0 )
        # newphiN = phiN + random_state.normal(0.,sqrt(gxiPerp)/gk0/Nperp)
## end test -        
        try:
            newNperp = disNperp(Bnorm, Ne, newNparallel, sigma)
        except:
            continue    # if no refractive index vector component Nperp found for the randomly generated Nparallel, 
                        # try again

        # and see if they can be accepted or not
        if target_sigma == sigma:
        
            probaccept = ScatteringProbabilityDistributionWithoutShape(newphiN, newNparallel,
                                                                       Bnorm, Ne,
                                                                       rho, theta,                       
                                                                       Nparallel, Nperp, phiN, sigma,
                                                                       xiParallel, xiPerp)
            probaccept /= ScatteringProbabilityDistributionWithoutShape(oldphiN, oldNparallel,
                                                                        Bnorm, Ne,
                                                                        rho, theta,                     
                                                                        Nparallel, Nperp, phiN, sigma,
                                                                        xiParallel, xiPerp)
        else:
            probaccept = ScatteringProbabilityDistributionToDifferentModeWithoutShape(newphiN, newNparallel,
                                                                       Bnorm, Ne,
                                                                       rho, theta,                       
                                                                       Nparallel, Nperp, phiN, sigma,
                                                                       xiParallel, xiPerp)
            probaccept /= ScatteringProbabilityDistributionToDifferentModeWithoutShape(oldphiN, oldNparallel,
                                                                        Bnorm, Ne,
                                                                        rho, theta,                     
                                                                        Nparallel, Nperp, phiN, sigma,
                                                                        xiParallel, xiPerp)

## test -            
        probaccept *= exp(-0.5*gk0**2/xiParallel*((oldNparallel-Nparallel)**2-(newNparallel-Nparallel)**2))
        probaccept *= exp(-0.5*gk0**2/xiPerp*Nperp**2*((oldphiN-phiN)**2-(newphiN-phiN)**2))

        # probaccept *= exp(-0.5*gk0**2/gxiParallel*((oldNparallel-Nparallel)**2-(newNparallel-Nparallel)**2))
        # probaccept *= exp(-0.5*gk0**2/gxiPerp*Nperp**2*((oldphiN-phiN)**2-(newphiN-phiN)**2))
## end test -        
      
        randomvariable = random_state.uniform(0.,1.)
       
        # see if new set of variables is accepted or not
        if probaccept > randomvariable:
            oldNparallel = newNparallel
            oldphiN = newphiN
            oldNperp = newNperp
        else:
            pass

    
    # return result
    return np.array([oldNparallel, oldNperp, oldphiN])



