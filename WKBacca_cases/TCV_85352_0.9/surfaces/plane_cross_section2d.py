"""
Definition of the section of a parametrized plane in AUG domain.
"""

import numpy as np

# Center of the plane in Cartesian coordinates (X, Z)
x0 = np.array([245.0, 0.0])

# Tangent and normal of the plane
#et = np.array([0.5, 0.5]) 
#en = np.array([-0.5, 0.5]) 
et = np.array([0.0, 1.0]) 
en = np.array([-1.0, 0.0]) 

# Parametrization
phi = lambda u: x0 + et * u
phi_u = lambda u: et
phi_u_perp = lambda u: en
 
# Range of the parameter and number of grid points
umin = -10.
umax = +10.
nptu = 200

# end of file
