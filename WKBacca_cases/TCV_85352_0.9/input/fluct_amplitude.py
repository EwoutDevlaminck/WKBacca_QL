"""
Model for the fluctuation envelope amplitude."""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


Ne_file = np.loadtxt('/home/devlamin/WKBacca_QL/WKBacca_cases/TCV_85352_0.9/input/ne.dat', skiprows=1)
rho, Ne = Ne_file[:, 0], Ne_file[:, 1]
# Start by imposing a given exponential profile and add the experimental values at the end

steepness = 20
baseline = 0.01
l = 0.5 #Strength of ballooning, 1 is no ballooning, 0 is full ballooning

# Fluct values from THB
fluct_values = [0.755356871721400, 0.571983405806562, 0.499026358203502, 0.431578231767353, 0.388040903642541, 0.316949611593894, 0.251359293177575]

def fluct_profile(rho, fluct_values, steepness=10, baseline=0.02):
    rho_first_exp = rho[-len(fluct_values)]
    # We want a profile that goes like 0.02 + a*exp(rho-1) and that matches the first experimental value
    a = (fluct_values[-1] - baseline) / np.exp(steepness*(rho_first_exp-1))
    delta_ne = baseline + a * np.exp(steepness*(rho-1))
    delta_ne[-len(fluct_values):] = fluct_values[::-1]
    return delta_ne

delta_ne = fluct_profile(rho, fluct_values, steepness, baseline)
# extend the profile to the end of the domain
rho_ext = np.concatenate([rho, np.linspace(rho[-1]+0.01, 2.0, 30)])
delta_ne_ext = np.concatenate([delta_ne, fluct_values[0]*np.ones(30)])


ampl_theta = lambda theta, l: .5 * (1 + l + (1 - l) * np.cos(theta))

ampl_rho_spline = interp1d(rho_ext, delta_ne_ext, kind="cubic")

ampl_rho = lambda rho: ampl_rho_spline(rho)


def scatteringDeltaneOverne(ne, rho, theta):
    return ampl_rho(rho) * ampl_theta(theta, l)

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
    Ne_file = np.loadtxt('/home/devlamin/WKBacca_QL/WKBacca_cases/TCV_85352_0.9/input/ne.dat', skiprows=1)
    rho, Ne = Ne_file[:, 0], Ne_file[:, 1]
    

    theta = 0
    delt = scatteringDeltaneOverne(fluct_values, np.concatenate([rho, np.linspace(rho[-1]+0.01, 2.0, 30)]), theta)
    plt.plot(np.concatenate([rho, np.linspace(rho[-1]+0.01, 2.0, 30)]), delt, label='Fitted profile')
    plt.scatter(rho[-len(fluct_values):], fluct_values[::-1], color='r', label='THB')
    plt.grid()
    plt.xlim(0, 1.1)
    plt.ylim(0, 1)
    plt.xlabel(r'$\rho_{\psi}$', fontsize=14)
    plt.ylabel(r'$RMS \delta n_e / n_e$', fontsize=14)
    plt.title('Fluctuation envelope amplitude')
    plt.legend(loc='upper left', fontsize=12)
    plt.tight_layout()
    plt.show()
