"""
"""



# Import Statements
import sys
import numpy as np



# This function loads TORBEAM profile files
def load_torbeam_profile(fname):

    """Open the data file specified in the argument string fname in
    the TORBEAM data directory and read the profile therein.
    """

    # Open file and read all lines
    f = open(fname, 'r')
    lines = f.readlines()

    # Account for differences in data formatting
    if fname == 'volumes.dat':
        first_data_line = 0
        number_data_lines = np.size(lines)
    else:
        first_data_line = 1
        number_data_lines = np.size(lines) - 1            

    # Load data
    psi_prf = np.empty([number_data_lines]) 
    y_prf = np.empty([number_data_lines])
    for ix in range(0, number_data_lines): 
        index = first_data_line + ix
        line = lines[index]
        split_line = line.split()
        # (note the square: profiles are functions of rho = sqrt(psi))
        psi_prf[ix] = float(split_line[0])**2
        y_prf[ix] = float(split_line[1])
            
    # Close file and return
    f.close()
    return psi_prf, y_prf

