"""
Definition of a parametrized sphere in AUG domain.
"""

import numpy as np

# Parameter of the sphere, centered in the origin
R = 5.0             # radius
center = (170.0, 0.0,  30.0)

# Define a surface parametrically in 3d
# (and provide the derivative of the mapping as well)
pi = np.pi
def phi(u, v):
    x = center[0] + R * np.sin(pi*u) * np.cos(2.0*pi*v)
    y = center[1] + R * np.sin(pi*u) * np.sin(2.0*pi*v)
    z = center[2] + R * np.cos(pi*u)
    return (x, y, z)
def phi_u(u, v): 
    e_ux = R * pi * np.cos(pi*u) * np.cos(2.0*pi*v)
    e_uy = R * pi * np.cos(pi*u) * np.sin(2.0*pi*v)
    e_uz = -R * pi * np.sin(pi*u)
    return (e_ux, e_uy, e_uz)
def phi_v(u, v):
    e_vx = -R * 2.0 * pi * np.sin(pi*u) * np.sin(2.0*pi*v)
    e_vy = +R * 2.0 * pi * np.sin(pi*u) * np.cos(2.0*pi*v)
    e_vz = 0.0
    return (e_vx, e_vy, e_vz)


umin = 1.e-8
umax = 1.0
nptu = 100

vmin = 0.0
vmax = 1.0
nptv = 100

# end of file
