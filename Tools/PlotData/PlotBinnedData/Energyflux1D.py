"""
Added by Ewout Devlaminck

Compute the flux through a surface defined by the user, summed over 
all other dimensions as to obtain a 1D flux profile.

The surface is defined by the user in exactly the same way as for the 2D and
3D flux profiles.

This function is not fully general, but I assume the correct data is stored
and we just sum over X,Y or Z.
"""

# Import statements
import sys
import h5py
import importlib
import numpy as np
import matplotlib.pyplot as plt

# All the files we want to compare, multiple can be plot
binned_files = ['/home/devlamin/Documents/WKBeam_related/WKBEAM_ED/StandardCases/TCV60612_1/output/flux_data_chellai.hdf5',
                '/home/devlamin/Documents/WKBeam_related/WKBEAM_ED/StandardCases/TCV60612_1/output/flux_data_0fluct.hdf5']


resolved_dimension = 'X' # 'X', 'Y' or 'Z'
axis_to_sum = 1 # 0 or 1, do this manually.

fluxes = {}

for filename in binned_files:
    # load and check the hdf5 dataset
    hdf5data = h5py.File(filename, 'r')

    binned_data = hdf5data.get(f'surface {resolved_dimension}')[()]
    minValue, maxValue = np.min(binned_data), np.max(binned_data)
    
    surface_flux = hdf5data.get('surface Fn')[()]

    # Sum over one dimension
    flux = np.sum(surface_flux, axis=axis_to_sum)

    local_name = filename.split('/')[-1][:-5]
    fluxes[local_name] = np.array([np.linspace(minValue, maxValue, len(flux)), flux])


# Plot the fluxes
plt.figure()
for filename, flux in fluxes.items():
    plt.plot(flux[0], -flux[1], label=filename)
plt.legend()
plt.show()
