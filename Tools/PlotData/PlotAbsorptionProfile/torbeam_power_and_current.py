""" This module provides functions for the visualization
of power deposition and current density profiles from TORBEAM.

The relevant data file is t2_new_LIB.dat.
"""

__author__ = 'Omar Maj (omaj@ipp.mpg.de)'
__version__ = '$Revision: $'
__date__ = '$Date: $'
__copyright__ = ' '
__license__ = 'Max-Planck-Institut fuer Plasmaphysik'

# Import statements
import numpy as np

# Reading the data file for extended rays
def read_power_and_current(data_directory):

    """Read the power deposition and driven current profiles 
    computed by TORBEAM and stored in the output file t2_new_LIB.dat.

    USAGE:
    
        rho, dP_dV, Jlle = read_power_and_current(data_directory)

    INPUT:
        > data_directory = string with the path to the directory
          where the desired file "to_relax.dat" is stored.

    OUTPUT:
        > rho = ndarray shape = (nptrho), grid in the variabla rho 
          defined as the square-root of the normalized poloidal flux;
        > dP_dV = ndarray shape = (nptrho), power deposition profile
          in Mw/m^3;
        > Jlle = ndarray shape = (nptrho), parallel electron current
          density in MA/m^2.
    """

    # Open file and load data
    filename = data_directory + 't2_new_LIB.dat'
    datafile = open(filename, 'r')
    lines = datafile.readlines()

    # Number of points
    nptrho = np.size(lines)

    # Initialize arrays
    rho = np.empty([nptrho])
    dP_dV = np.empty([nptrho])
    Jlle = np.empty([nptrho])
    
    # Load the arrays
    for irho in range(0, nptrho):
        line = lines[irho]
        data = line.split()
        rho[irho] = float(data[0])
        dP_dV[irho] = float(data[1])
        Jlle[irho] = float(data[2])

    # Return the arrays
    return rho, dP_dV, Jlle
