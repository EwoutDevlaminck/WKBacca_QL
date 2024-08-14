"""Some basic functions called from ScatteringPlasmaModel are defined.
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
    double tan( double )
    double cosh( double )
    bint isnan( double )

# constants
cdef double pi = 3.141592654   
cdef double emass = 9.11*1e-31            # electron mass in kg
cdef double echarge = 1.602*1e-19         # electron charge in C
cdef double epsilon0 = 8.85*1e-12         # dielectric constant in V/Cm


# and some global variables
cdef double gLambda
cdef double gscatteringrhocentral
cdef double gDelta
## test - xiParallel and xiPerp no longer global contants
# cdef double gxiParallel, gxiPerp
## end test -
cdef double gk0
cdef double gomega
cdef double gepsilonRegS
cdef int gnmbrinitialisationMetropolisHastingsScattering
cdef double gscatteringDeltaneOverne
cdef double gscatteringLcz
cdef double gscatteringLcr
cdef double gscatteringkz


##############################################################################
# DEFINE INITIALISATION FUNCTION FOR THE GLOBAL VARIABLES
##############################################################################
# shape of the fluctuations
## test - xiPErp and xiParallel no longer global constants
cpdef  scatteringinitialiseGlobalVariables(double k0, double omega, double epsilonRegS, double Lambda, double scatteringrhocentral, double Delta, int nmbrinitialisationMetropolisHastingsScattering, double scatteringDeltaneOverne, double scatteringLcz, double scatteringLcr, double scatteringkz, bool freeze_seed, int rank):

# old version
# cpdef  scatteringinitialiseGlobalVariables(double k0, double omega, double epsilonRegS, double Lambda, double scatteringrhocentral, double Delta, double xiParallel, double xiPerp, int nmbrinitialisationMetropolisHastingsScattering, double scatteringDeltaneOverne, double scatteringLcz, double scatteringLcr, double scatteringkz, bool freeze_seed, int rank):
## end test - 
    
    # refer to global variables
## test - xiPerp and xiParallel no longer global constants    
    global gLambda, gscatteringrhocentral, gDelta, gk0, gomega, gepsilonRegS, gscatteringDeltaneOverne

    # old version
    # global gLambda, gscatteringrhocentral, gDelta, gxiParallel, gxiPerp, gk0, gomega, gepsilonRegS, gscatteringDeltaneOverne
## end test -        
    global gnmbrinitialisationMetropolisHastingsScattering
    global gscatteringLcz, gscatteringLcr, gscatteringkz
    global random_state

    # copy arguments to the global variables
    gLambda = Lambda 
    gscatteringrhocentral = scatteringrhocentral
    gDelta = Delta

## test - these are no longer global constants    
    # gxiParallel = xiParallel
    # gxiPerp = xiPerp
## end test -    

    gk0 = k0
    gomega = omega
    gepsilonRegS = epsilonRegS
    gnmbrinitialisationMetropolisHastingsScattering = nmbrinitialisationMetropolisHastingsScattering
    gscatteringDeltaneOverne = scatteringDeltaneOverne

    gscatteringLcz = scatteringLcz
    gscatteringLcr = scatteringLcr
    gscatteringkz = scatteringkz

    # random number generator
    if freeze_seed:
        random_state = np.random.RandomState(rank+1)
    else:
        randomseed = np.random.randint(1234)
        random_state = np.random.RandomState((rank+1)*randomseed)

    # return from function
    return None






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


# function that returns the Nperp-component such that the refractive index vector
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
# shape of the fluctuations
cpdef double ShapeModel(double rho, double theta):

    """ Return the shape-prefactor for the scattering model.
    """

# test - only essential constants are loaded    
    global gLambda, gDelta, gscatteringrhocentral,gscatteringDeltaneOverne

    # old version
    # global gLambda, gDelta, gscatteringrhocentral,gscatteringDeltaneOverne,gxiPerp,gxiParallel
## end test -    

# omaj - this can be negative for gLambda = -1.
    # return 0.5*(1.+gLambda+(1.-gLambda)*cos(theta)) \
    #         *exp(-(rho-gscatteringrhocentral)**2/gDelta**2) \
    #         *gscatteringDeltaneOverne
# changed to
    return 0.5*(1.+gLambda+(1.-gLambda)*0.5*(1.+cos(theta))) \
            *exp(-(rho-gscatteringrhocentral)**2/gDelta**2) \
            *gscatteringDeltaneOverne
# omaj - end



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
## test - xiPErp and xiParallel passed explicitly
cpdef ScatteringProbabilityDistributionWithoutShape(double phiNprime, double Nparallelprime,
                                                    double Bnorm, double Ne, 
                                                    double rho, double theta,          # tokamak coordinates
                                                    double Nparallel, double Nperp, double phiN, double sigma,
                                                    double xiParallel, double xiPerp):

# old version
# cpdef ScatteringProbabilityDistributionWithoutShape(double phiNprime, double Nparallelprime,
#                                                     double Bnorm, double Ne, 
#                                                     double rho, double theta,          # tokamak coordinates
#                                                     double Nparallel, double Nperp, double phiN, double sigma):
## end test -


    """Computes the probability distribution for the new refractive index 
    components.
    See theory sheets 'analytical model for density fluctuations in a tokamak' 
    on page 5 for details on computations.

    """

    # ATTENTION:
    # THERE ARE TWO ANGLES THETA AROUND IN THIS FUNCTION.
    # NOTE, THAT costheta and sintheta REFER TO THE ANGLE EXPRESSING PLASMA PARAMETERS,
    # WHEREAS theta (cf. PARAMETER) IS THE GEOMETRICAL ANGLE OF COORDINATES IN THE
    # POLOIDAL PLANE.


## test - xiPerp and xiParallel no longer global constants
    global gk0

    # old version
    # global gk0, gxiPerp, gxiParallel
## end test -

    global gomega
    global gscatteringLcz, gscatteringLcr, gscatteringkz

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
    cdef double curlybracketssquared = (L*Lprime*(sintheta*sinthetaprime*dS_dne + costheta*costhetaprime*dP_dne) \
                               + L*Tprime*(sintheta*costhetaprime*dS_dne + costheta*sinthetaprime*dP_dne) \
                               + T*Lprime*(costheta*sinthetaprime*dS_dne + sintheta*costhetaprime*dP_dne) \
                               + T*Tprime*(costheta*costhetaprime*dS_dne + sintheta*sinthetaprime*dP_dne) \
                               - L*sintheta*dD_dne + Lprime*sinthetaprime*dD_dne \
                               - T*costheta*dD_dne + Tprime*costhetaprime*dD_dne \
                               - dS_dne)**2 


    # compute spectrum of the electron density fluctuations
    # (for conventions see theory sheet "Shafer model for fluctuations" in collection
    # for publication)
    cdef double Nr1 = +Nperp*sin(phiN)*cos(theta) - sin(theta)*Nperp*cos(phiN)
    cdef double Nz1 = -Nperp*sin(phiN)*sin(theta) - cos(theta)*Nperp*cos(phiN)
    cdef double Nt1 = Nparallel
    cdef double Nr2 = +Nperpprime*sin(phiNprime)*cos(theta) - sin(theta)*Nperpprime*cos(phiNprime) 
    cdef double Nz2 = -Nperpprime*sin(phiNprime)*sin(theta) - cos(theta)*Nperpprime*cos(phiNprime)
    cdef double Nt2 = Nparallelprime

    cdef double Gamma = (gk0/sqrt(2.*pi))**3/sqrt(xiParallel)*gscatteringLcr*gscatteringLcz  \
        * exp(-0.5*gk0**2 \
                        *((Nt1-Nt2)**2/xiParallel \
                        + (Nr1-Nr2)**2*gscatteringLcr**2 \
                        + (Nz1-Nz2)**2*gscatteringLcz**2)) \
        * exp(-0.5*gscatteringLcz**2*gscatteringkz**2) * cosh(gk0*gscatteringLcz*gscatteringkz*(Nz1-Nz2))



    cdef correctionfactorprime = abs(discorrectionFactorf(gomega, Bnorm, Ne, Nparallelprime, sigma))

    return pi/2.*gk0*Ne**2 * Gamma * curlybracketssquared / norm / normprime * correctionfactorprime






# probability distribution for scattering
## test - xiPerp and xiParallel passsed explicitly
cpdef ScatteringProbabilityDistributionToDifferentModeWithoutShape(double phiNprime, double Nparallelprime,
                                                                   double Bnorm, double Ne, 
                                                                   double rho, double theta,    # tokamak coordinates
                                                                   double Nparallel, double Nperp, double phiN, double sigma,
                                                                   double xiParallel, double xiPerp):

# old version
# cpdef ScatteringProbabilityDistributionToDifferentModeWithoutShape(double phiNprime, double Nparallelprime,
#                                                                    double Bnorm, double Ne, 
#                                                                    double rho, double theta,    # tokamak coordinates
#                                                                    double Nparallel, double Nperp, double phiN, double sigma):
# end test -


    """Computes the probability distribution for the new refractive index 
    components.
    See theory sheets 'analytical model for density fluctuations in a tokamak' 
    on page 5 for details on computations."""


## test - xiParallel and xiPErp no longer global constants    
    global gk0

    # old version
    # global gk0, gxiPerp, gxiParallel
## end test -        
    global gomega
    global gscatteringLcz, gscatteringLcr, gscatteringkz

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
    cdef double curlybracketssquared = (L*Lprime*(sintheta*sinthetaprime*dS_dne + costheta*costhetaprime*dP_dne) \
                               + L*Tprime*(sintheta*costhetaprime*dS_dne + costheta*sinthetaprime*dP_dne) \
                               + T*Lprime*(costheta*sinthetaprime*dS_dne + sintheta*costhetaprime*dP_dne) \
                               + T*Tprime*(costheta*costhetaprime*dS_dne + sintheta*sinthetaprime*dP_dne) \
                               - L*sintheta*dD_dne + Lprime*sinthetaprime*dD_dne \
                               - T*costheta*dD_dne + Tprime*costhetaprime*dD_dne \
                               - dS_dne)**2 
    

    # compute spectrum of the electron density fluctuations
    # (for conventions see theory sheet "Shafer model for fluctuations" in collection
    # for publication)
    cdef double Nr1 = +Nperp*sin(phiN)*cos(theta) - sin(theta)*Nperp*cos(phiN)
    cdef double Nz1 = -Nperp*sin(phiN)*sin(theta) - cos(theta)*Nperp*cos(phiN)
    cdef double Nt1 = Nparallel
    cdef double Nr2 = +Nperpprime*sin(phiNprime)*cos(theta) - sin(theta)*Nperpprime*cos(phiNprime) 
    cdef double Nz2 = -Nperpprime*sin(phiNprime)*sin(theta) - cos(theta)*Nperpprime*cos(phiNprime)
    cdef double Nt2 = Nparallelprime

    cdef double Gamma = (gk0/sqrt(2.*pi))**3/sqrt(xiParallel)*gscatteringLcr*gscatteringLcz  \
        * exp(-0.5*gk0**2 \
                        *((Nt1-Nt2)**2/xiParallel \
                        + (Nr1-Nr2)**2*gscatteringLcr**2 \
                        + (Nz1-Nz2)**2*gscatteringLcz**2)) \
        * exp(-0.5*gscatteringLcz**2*gscatteringkz**2) * cosh(gk0*gscatteringLcz*gscatteringkz*(Nz1-Nz2))

    # in case Nz1 and Nz2 are (unphysically) far from each other, no scattering occurs and 
    # Gamma may be set to zero by hand; otherwise, in the equation too large terms occur for the 
    # computer standard double variables
    if abs(Nz1 - Nz2) > 10:
        Gamma = 0.

    cdef correctionfactorprime = abs(discorrectionFactorf(gomega, Bnorm, Ne, Nparallelprime, sigma))

    return pi/2.*gk0*Ne**2 * Gamma * curlybracketssquared / norm / normprime * correctionfactorprime






# uses Metropolis-Hastings algorithm in order to pick a new set of refractive index 
# vector components
# returns: Nparallel, Nperp, phiN

## test - xiPerp and xiParallel no longer global constants
cpdef np.ndarray [double, ndim=1] ScatteringChooseRefractiveIndex(double Nparallel, double Nperp, double phiN, 
                                                                  double Bnorm, double Ne, 
                                                                  double rho, double theta,
                                                                  double sigma, double target_mode,
                                                                  double xiParallel, double xiPerp):

# old version
# cpdef np.ndarray [double, ndim=1] ScatteringChooseRefractiveIndex(double Nparallel, double Nperp, double phiN, 
#                                                                   double Bnorm, double Ne, 
#                                                                   double rho, double theta,
#                                                                   double sigma, double target_mode):
## end test -

    # refer to global variables
    global gnmbrinitialisationMetropolisHastingsScattering

## test - xiPerp and xiParallel no longer global constants    
    global gk0

    # old version
    # global gxiParallel, gxiPerp, gk0
## end test -    

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
        newNparallel = random_state.normal(Nparallel,sqrt(xiParallel)/gk0 )
        newphiN = phiN + random_state.normal(0.,sqrt(xiPerp)/gk0/Nperp)
        try:
            newNperp = disNperp(Bnorm, Ne, newNparallel, sigma)
        except:
            continue    # if no refractive index vector component Nperp found for the randomly generated Nparallel, 
                        # try again

        # and see if they can be accepted or not
        if target_mode == sigma:
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

        probaccept *= exp(-0.5*gk0**2/xiParallel*((oldNparallel-Nparallel)**2-(newNparallel-Nparallel)**2))
        probaccept *= exp(-0.5*gk0**2/xiPerp*Nperp**2*((oldphiN-phiN)**2-(newphiN-phiN)**2))
      
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



