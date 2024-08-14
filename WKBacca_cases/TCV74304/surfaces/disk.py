"""
Definition of a parametrized disk in AUG domain
"""

import numpy as np

# Center of the plane in Cartesian coordinates (X, Y, Z)
x0 = np.array([170.0, 0.0,  30.0])
eY = np.array([0.0, 1.0, 0.0])
eZ = np.array([0.0, 0.0, 1.0])

phi = lambda u, v: x0 + eY * u * np.cos(v) + eZ * u * np.sin(v)
phi_u = lambda u, v: eY * np.cos(v) + eZ * np.sin(v)
phi_v = lambda u, v: -eY * u * np.sin(v) + eZ * u * np.cos(v)

umin = 0.01
umax = 5.0 # cm
nptu = 100

vmin = 0.0
vmax = 2.0*np.pi
nptv = 100

# end of file
