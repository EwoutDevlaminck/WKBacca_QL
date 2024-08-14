"""
Definition of a parametrized plane in TCV
"""

import numpy as np

# Center of the plane in Cartesian coordinates (X, Y, Z)
x0 = np.array([95.0, 0.0,  0.0])
eX = np.array([1.0, 0.0, 0.0])
eY = np.array([0.0, 1.0, 0.0])
eZ = np.array([0.0, 0.0, 1.0])

phi = lambda u, v: x0 + eZ * u + eY * v
phi_u = lambda u, v: eZ
phi_v = lambda u, v: eY

umin = -20.0
umax = 20.0
nptu = 200

vmin = -30.0
vmax = 30.0
nptv = 200

# end of file
