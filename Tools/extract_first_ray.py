"""
Run this script after a full ray tracing calculation in order to
extract the first ray computed by a group of CPUs. If the
option takecentralrayfirst = True  is activated and the first group
of CPUs is considered, then that would be the central ray.

Usage:

  $ python extract_fisrt_ray.py raytracingdata fname

where raytracingdata is an hdf5 dataset produced by the ray tracing
module of WKBeam.

The variable fname is optional. If present, that is the name of the
txt file where the ray is written in the ASCII format

 X0 Y0 Z0
 X1 Y1 Z1
 X2 Y2 Z2
 ...

If not prefent the default 'first_ray.txt' is used with the same format.
"""

import sys
import h5py
import numpy as np

# Load the dataset
try:
    dataset = sys.argv[1]
except:
    print(__doc__)
    raise RuntimeError

# Define the filename
try:
    fname = sys.argv[2]
except:
    fname = 'first_ray.txt'
    

# Open the dataset
fid = h5py.File(dataset, 'r')

# Get the data
TracesXYZ = fid.get('TracesXYZ')[()]
XYZ = TracesXYZ[0]

# Extract the fisrt ray (which usually is the reference ray)
rr = np.rollaxis(XYZ, 1)

# Save to txt format
np.savetxt(fname, rr)

# End of file

