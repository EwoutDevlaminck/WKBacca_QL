"""THIS FILE CONTAINS A FUNCTION WHICH IS ABLE TO COMPUTE THE NORMALISATION
FACTOR FOR THE INPUT POWER.
"""

# Load standard modules
import numpy as np
import math
from scipy.integrate import dblquad

# Load local modules
import CommonModules.physics_constants as PhysConst

############################################################################
# BELOW: A FUNCTION WHICH COMPUTES THE NORMALISATION
# FACTOR 
# in fact this is the energy flow through the antenna assuming an
# electric field normalised to one at the center
# In order to obtain correct results for any chosen input power,
# multiply with this power and divide by the normalisation factor
# produced using this function.
############################################################################
def compute_norm_factor(freq,
                        beamwidth1, beamwidth2,
                        curvatureradius1, curvatureradius2,
                        centraleta1, centraleta2):


    # compute some quantities
    c = PhysConst.SpeedOfLight               # speed of light in cm / s
    omega = PhysConst.AngularFrequency(freq) # 2.*math.pi*freq*1e9
    k0 = PhysConst.WaveNumber(omega)         # omega / c  wave vector in free space

    eigenvalueD1 = PhysConst.EigenvalueD(k0, beamwidth1)   # eigenvalues for 
    eigenvalueD2 = PhysConst.EigenvalueD(k0, beamwidth2)   # curvature and beamwidth
    eigenvalueS1 = PhysConst.EigenvalueS(curvatureradius1) # matrices
    eigenvalueS2 = PhysConst.EigenvalueS(curvatureradius2)

    # compute factor to achieve microwave beam power
    # therefore: define 2.*Nx * Wfct, already integrated analytically over y, z
    normfacttemp = lambda eta2, eta1: 2.*math.sqrt(1.-eta1**2-eta2**2) \
        / math.sqrt(eigenvalueD1**2+eigenvalueS1**2) \
        / math.sqrt(eigenvalueD2**2+eigenvalueS2**2) \
        * math.exp(-k0*eigenvalueD1 \
                        /(eigenvalueD1**2+eigenvalueS1**2) \
                        *(eta1-centraleta1)**2) \
                        * math.exp(-k0*eigenvalueD2 \
                                        /(eigenvalueD2**2+eigenvalueS2**2) \
                                        *(eta2-centraleta2)**2)
                
    # and integrate over eta1, eta2
    normfact = dblquad(
        normfacttemp,                      #fct to integrate
        -1.,                               # boundaries for eta1
        +1.,      
        lambda x: -math.sqrt(1.-x**2),     # boundaries for eta2 
        lambda x: +math.sqrt(1.-x**2),             
        args=())[0]

    return normfact
