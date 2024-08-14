"""Reads the ne profile and optimizes it such that
it can be used for spline interpolation later on.
In particular, points are added outside the separatrix.
The decay-length and the number of points may be specified.

call this file with the command
python opt_ne.py
"""


# Import Statements
import sys
from loadInputFile import load_torbeam_profile
import numpy as np
import math
from scipy.optimize import curve_fit


# specify parameters
##############################################################################
# filename from where the electron density profile is read
input_file = './../../../../InputFiles/AUGexpdecay/ne.dat'
# filename where the modified profile is written
output_file = './ne.dat'

# number of points to add (number of points added outside the last point in 
# the electron density profile)
nmbrpoints = 100

# step size in psi for the added points (so the last psi-value will be 
# the last value of the original file + nmbrpoints*Deltapsi
Deltapsi = 0.003

# decay constant: outside last value, a decay exp(-alpha*(rho-1)) is assumed
# (Taylort approximated with exp(-alpha/2 (psi-1))
alpha = 110.
#############################################################################



# load profile and add points 
##############################################################################

psi_profile_orig, ne_profile_orig = load_torbeam_profile(input_file)

psi_profile = np.empty([psi_profile_orig.size + nmbrpoints])
ne_profile = np.empty([ne_profile_orig.size + nmbrpoints])


for i in range(psi_profile.size):
    if i < psi_profile_orig.size:
        # just copy the original value as long as available
        psi_profile[i] = psi_profile_orig[i]
        ne_profile[i] = ne_profile_orig[i]
    else:
        # and extend with additional data points 
        psi_profile[i] = psi_profile[i-1] + Deltapsi
        ne_profile[i] = ne_profile[i-1] * math.exp(-alpha/2. * Deltapsi)

##############################################################################



# show the result 
##############################################################################
import pylab 
import matplotlib.pyplot as plt

plt.figure(1)

plt.plot(np.sqrt(psi_profile_orig),ne_profile_orig,'+')
plt.hold(True)
plt.plot(np.sqrt(psi_profile),ne_profile)
plt.xlabel('rho')
plt.ylabel('ne')
plt.show()
##############################################################################


# write the results to a file
##############################################################################
Data = np.zeros([2,psi_profile.size])
Data[0,:] = np.sqrt(psi_profile)
Data[1,:] = ne_profile
np.savetxt(output_file,Data.T, delimiter=' ', newline='\n',)
##############################################################################
