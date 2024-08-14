"""
Model of perpendicular correlation length deduced from the 
sound Larmor radius rho_s parameter defined by

  rho_s = Cs/Omega = (Cs/Ci) rho_i,

where Cs and Ci are the ion sound and ion thermal speed, respectively,
whereas rho_i is the Larmor radius of a thermal ion.

From the NRL Plasma Formulary (pp. 28, 29) one has

  Cs = 9.79e5 (Z*Te/mu)^1/2 cm/sec,
  Ci = 9.79e5 (Ti/mu)^1/2  cm/sec,
  rho_i = 1.02e2 (mu*Ti)^1/2 Z^-1 B^-1 cm

where mu = m_i/m_p, Z is the charge status, Ti and Te the ion and 
electron temperature, B the magnetic field strength.

It follows that

  Cs/Ci = (Z*Te/Ti)^1/2,

and thus

  rho_s = (Z*Te/Ti)^1/2 * 1.02e2 (mu*Ti)^1/2 Z^-1 B^-1 cm
        = 1.02e2 * (mu*Te/Z)^1/2 B^-1 cm.

At last, we set the perpendicular correlation length to

  Lperp = factor * rho_s = factor * 1.02e2 * (mu*Te/Z)^1/2 B^-1 cm,

according to the usual "rule of thunb".

"""

from math import sqrt


# Ion parameter (deuterium plasma) 
mu = 2.0
Z = 1.0

# parameters of the model
epsilon = 0.3
factor = 15


def scatteringLengthPerp(rho, theta, Ne, Te, Bnorm):


    """
    Compute the scattering perpendicular correlation length Lperp,
    according to the rule 
     
          Lperp = max(epsilon, factor * rho_s), 

    where rho_s is the sound Larmor radius and epsilon is an offset 
    in order to prevent it to become too small where Te=0. 
    """

    # Convert Te from keV to eV and Bnorm from Tesla to Gauss.
    # Small negative Te are possible but irrelevant.
    Te = abs(Te) * 1.e3 
    Bnorm = Bnorm * 1.e4
    
    return max(epsilon, factor * 1.02e2 * sqrt(mu*Te/Z) / Bnorm)


# End of file

