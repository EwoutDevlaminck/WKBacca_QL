"""
Auxiliary function used to extract grid in a given direction
from binned files.
"""

# Standard modules
import math
import numpy as np
from RayTracing.modules.rotation_matrix import rotMatrix


# Get the contents of the dataset
def get_contents(hdf5data):
    grid_parameters = hdf5data.get('WhatToResolve')[()]
    # the try loop is needed for compatibility with python 2.x
    try:
        wtr = grid_parameters.decode()
    except:
        wtr = grid_parameters 
    wtr_list = wtr.split(',')
    return wtr_list[:-1]

# Get grid parameters for the x axis
def get_grid(data, datalabel):

    # Reading grid data depending on the data label
    minimum = data.get(datalabel+'min')[()]
    maximum = data.get(datalabel+'max')[()]
    npt = data.get('nmbr'+datalabel)[()]
    # points
    grid = np.linspace(minimum, maximum, npt)
    # step size
    step = (maximum - minimum) / npt   
    # return results
    return grid, step

# This function re-arrange the data on the energy flux vector field 
# in the format required by mlab.flow.
def build_grid_and_vector_field(field, skip):
    
    """
    Prepare the energy field for te stream-plot.
    """
    
    # Build the three-dimensional grid
    x1 = field.x1[::skip]
    x2 = field.x2[::skip] 
    x3 = field.x3[::skip] 
    Xgrid = np.array(np.meshgrid(x1, x2, x3, indexing='ij'))

    # Reference to the field
    Ffield = field.F[:, ::skip, ::skip, ::skip]

    return Xgrid, Ffield


# This function construct a plane surface which is oriented as the antenna
# plane where the beam launching conditions are geiven
def build_antenna_plane(x0, polangle, torangle, w1, w2):
    
    """
    Contruct a parametric representation of a plane oriented as the antenna
    in the launching system.
    """

    # use the function given above to compute the rotational matrix
    T = rotMatrix(polangle, torangle)

    # define the two-dimensional mesh
    antw1 = 1.5 * w1
    antw2 = 1.5 * w2
    dy1 = dy2 = 0.1
    Y = np.mgrid[-antw1:antw1:dy1, -antw2:antw2:dy2]
                   
    # and the starting point in laboratory system
    Xant = x0[0] + T[0,0]*Y[0] + T[0,1]*Y[1]
    Yant = x0[1] + T[1,0]*Y[0] + T[1,1]*Y[1]
    Zant = x0[2] + T[2,0]*Y[0] + T[2,1]*Y[1]

    return Xant, Yant, Zant
