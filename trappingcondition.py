import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d
from scipy.optimize import minimize, fsolve
from scipy.io import loadmat

import CommonModules.BiSplineDer as bispl


# Function for getting the data on the psi, theta grid

def Eq_pt(filename):
    data = loadmat(filename)['equil'][0, 0] 

    psi = np.array(data['psi_apRp'][0])
    psi_norm = psi/np.max(psi)
    theta = np.array(data['theta'][0])

    plt.plot(psi)
    plt.show()
    plt.plot(theta)
    plt.show()
    print(psi_norm.shape, theta.shape)

    Rp = np.array(data['Rp'][0][0])
    Zp = np.array(data['Zp'][0][0])

    pt_R = np.array(data['ptx'] + Rp)
    pt_Z = np.array(data['pty'] + Zp)

    ptBP = np.array(data['ptBP'])
    ptBPHI = np.array(data['ptBPHI'])

    ptB = np.sqrt(ptBP**2 + ptBPHI**2)

    # Change the shape of the theta arrays and pt_B, to get minima well resolved
    npt_theta = len(theta)
    theta_shift = np.concatenate((theta[npt_theta//2:-1]-2*np.pi, theta[:npt_theta//2]))
    ptB_shift = np.concatenate((ptB[:, npt_theta//2:-1], ptB[:, :npt_theta//2]), axis=1)

    ptB_shiftInt = bispl.BiSpline(psi_norm, theta_shift, ptB_shift)

    return psi_norm, theta_shift, ptB_shiftInt, pt_R, pt_Z

# Functions for the minimal and maximal values of the magnetic field on a given flux surface
# Usually is just very close to the values at theta 0 and pi,
# but nonetheless some difference (see __name__ == '__main__' part)

def minmaxB(psi, BInt, theta):

    # Find B(theta) at the requested psi
    B_at_psi = np.array([BInt.eval(psi, theta) for theta in theta])
    B_at_psiInt = interp1d(theta, B_at_psi, kind='cubic')

    minusB_at_psi = np.array([-BInt.eval(psi, theta) for theta in theta])
    minusB_at_psiInt = interp1d(theta, minusB_at_psi, kind='cubic')

    minimum = minimize(B_at_psiInt, 0.)
    maximum = minimize(minusB_at_psiInt, -np.pi)

    # Return the interpolated function, the minimum and maximum

    return B_at_psiInt, B_at_psiInt(minimum.x), B_at_psiInt(maximum.x)


# Function to calculate the quantities related to trapping.
#   - B_bounce(psi, ksi0): at what field a particle will bounce
#   - Ksi_trapping(psi): The boundary value for given psi. Particles with smaller ksi will be trapped
#   - theta_T,m and theta_T,M(psi,ksi0): minimal and maximal angle reached by particles (where they meet B_bounce)


def Trapping_boundary(psi, ksi, BInt, theta_grid=[]):
    TrapB = np.zeros((len(psi), len(ksi)))
    Trapksi = np.zeros_like(psi)
    theta_roots = np.zeros((len(psi), len(ksi), 2))


    for i, psi_val in enumerate(psi):
        B_at_psiInt, B0, Bmax = minmaxB(psi_val, BInt, theta_grid)
        TrapB[i] = B0/(1-ksi**2)
        Trapksi[i] = np.sqrt(1-B0/Bmax)

        for j, ksi_val in enumerate(ksi):
            if abs(ksi_val) <= Trapksi[i]:
                def deltaB(x):
                    return B_at_psiInt(x) - TrapB[i, j]

                theta_roots[i, j] = fsolve(deltaB, [-np.pi/2, np.pi/2])
            else:
                theta_roots[i, j, 0] = np.nan
                theta_roots[i, j, 1] = np.nan

    return TrapB, Trapksi, theta_roots



if __name__== '__main__':

    filename = '/home/devlamin/Documents/WKBeam_related/WKBacca_dev_v1/TCV_preprocess/EQUIL_TCV_72644_1.05s.mat'

    psi_norm, theta_shift, ptB_shiftInt, pt_R, pt_Z = Eq_pt(filename)


    # Comparing to just B value at theta 0 and pi

    def minmaxBalt(psi_val, BInt):
        # Just return the value at theta=0 and pi
        return BInt.eval(psi_val, 0.), BInt.eval(psi_val, np.pi)
    

    fig = plt.figure()

    for i in psi_norm:
        _, m, M = minmaxB(i, ptB_shiftInt, theta_shift)
        m2, M2 = minmaxBalt(i, ptB_shiftInt)
        plt.scatter(i, 100*(m-m2)/m2, c='r')
        plt.scatter(i, 100*(M-M2)/M2, c='k')
        plt.title('Difference between two methods for finding the max and min Bfield')
        plt.ylabel(r'$\Delta B [\%]$')

    #plt.show()

    # Get some resulting plots

    ksi = np.linspace(-0.7, 0.7, 150)
 


    Bbound, ksibound, thetasbound = Trapping_boundary(psi_norm, ksi, ptB_shiftInt, theta_shift)
    fig = plt.figure()

    a = plt.contourf(psi_norm, ksi, Bbound.T, levels=100)
    plt.plot(psi_norm, ksibound, c='r', label=r'$\xi_{trapped}\ boundary$')
    plt.plot(psi_norm, -ksibound, c='r')


    cb = plt.colorbar(a)
    cb.set_label(label=r'$B_{bounce}[T]$', size=14)
    plt.xlabel(r'$\psi$', fontsize=14)
    plt.ylabel(r'$\xi$', fontsize=14)
    plt.legend(fontsize=12)

    #plt.show()


    from matplotlib.colors import LinearSegmentedColormap
    coolwarm = plt.cm.get_cmap('coolwarm', 256)

    coolmap = coolwarm(np.linspace(0, 0.5, 128))
    warmmap = coolwarm(np.linspace(0.5, 1, 128))


    fig = plt.figure(figsize=(10, 6))

    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)

    figone = ax1.contourf(psi_norm, ksi, thetasbound[:,:,0].T, levels=100, cmap=LinearSegmentedColormap.from_list('coolwarm', coolmap))
    ax1.set_xlabel(r'$\psi$', fontsize=14)
    ax1.set_ylabel(r'$\xi$', fontsize=14)
    ax1.set_title(r'$\theta_{trapped, m}$')

    figtwo = ax2.contourf(psi_norm, ksi, thetasbound[:,:,1].T, levels=100, cmap=LinearSegmentedColormap.from_list('coolwarm', warmmap))
    ax2.set_xlabel(r'$\psi$', fontsize=14)
    ax2.set_ylabel(r'$\xi$', fontsize=14)
    ax2.set_title(r'$\theta_{trapped, M}$')


    cb1 = plt.colorbar(figone, ax=ax1)
    cb1.set_label(label=r'$\theta_T$', size=14)

    cb2 = plt.colorbar(figtwo, ax=ax2)
    cb2.set_label(label=r'$\theta_T$', size=14)

    plt.tight_layout()
    plt.show()
