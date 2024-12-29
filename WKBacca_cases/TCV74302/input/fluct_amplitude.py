"""
Model for the fluctuation envelope amplitude."""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

#Get the points from Chellaï's paper, taken in an ugly way.

ampl_points = np.genfromtxt("/home/devlamin/WKBacca_QL/WKBacca_cases/TCV74302/input/amplitude_points.csv", delimiter=";")

#Add some points before and after the ones from the paper, in order to have a smooth interpolation.
points_before = np.array([np.linspace(0., ampl_points[0, 0], 10), [ampl_points[0, 1]]*10]).T
points_after = np.array([np.linspace(ampl_points[-1, 0], 2.0, 10), [ampl_points[-1, 1]]*10]).T
ampl_points = np.vstack([points_before[:-1, :], ampl_points, points_after[1:, :]])


ampl_rho_spline = interp1d(ampl_points[:,0], ampl_points[:,1], kind="cubic")

ampl_rho = lambda rho: ampl_rho_spline(rho)

ampl_theta = lambda theta, l: .5 * (1 + l + (1 - l) * np.cos(theta))

l = 0.5 #Strength of ballooning, 1 is no ballooning, 0 is full ballooning

def scatteringDeltaneOverne(ne, rho, theta):
    return 2*ampl_rho(rho) * ampl_theta(theta, l) #The factor 2 was artificially added by Ewout here.

if __name__ == "__main__":
    
    """
    rho = np.linspace(0., 1.5, 100)
    theta = np.linspace(0., 2*np.pi, 360)

    RHO, THETA = np.meshgrid(rho, theta)
    R, Z = RHO * np.cos(THETA), RHO * np.sin(THETA)

    delt = scatteringDeltaneOverne(1., RHO, THETA)

    plt.scatter(R, Z, c=delt, s=1)
    plt.colorbar()
    plt.show()
    """
    Ne_file = np.loadtxt('WKBacca_cases/TCV72644/t_1.05/input/ne.dat', skiprows=1)
    rho, Ne = Ne_file[:, 0], Ne_file[:, 1]

    theta = 0
    delt = scatteringDeltaneOverne(Ne, rho, theta)
    plt.plot(rho, delt*Ne)
    plt.xlabel(r'$\rho$')
    plt.ylabel(r'$\delta n_e (1e19 /m³)$')
    plt.show()
