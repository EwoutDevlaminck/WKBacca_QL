"""Read magnetic configuration of a tokamaks from standard equilibrium
formats.
"""

__author__ = 'Omar Maj (omaj@ipp.mpg.de)'
__version__ = 'Revision: '
__date__ = 'Date: '
__copyright__ = ' '
__license__ = 'Max-Planck-Institut fuer Plasmaphysik'

# Import statement
import numpy as np

#
# --- functions ---
#

# Read a topfile (torbeam, torray, lhbeam, ...)
def read(data_directory):
    """Read a topfile (an equilibrium configuration format typically
    used by codes TORBEAM, LHBEAM, TORRAY, ...), and store data in numpy
    arrays.

    USAGE: R, z, Bfield, psi, psi_sep = read_topfile(data_directory)
    
    Input variables:
       > data_directory, string with the path to the directory
         where the topfile is stored.

    Returns: the list (R, z, Bfield, psi, psi_sep) 
    where
       > R[iR, iz], 2d numpy array, with R on the 2d poloidal grid.
       > z[iR, iz], 2d numpy array, with z on the 2d poloidal grid.
       > Bfield[ib, iR, iz], 3d numpy array, with the components 
         of the magnetic field on grid points (R, z) in the poloidal plane;
         the indices iR and iz run over grid points in R and z, respectively, 
         while the index ib runs over the components of the magnetic field, 
           - ib = 0 for the radial (R) component, 
           - ib = 1 for the vertical (z) component, and 
           - ib = 2 for the toroidal component.
       > psi[iR, iz], 2d numpy with the poloidal flux at grid
         points in the poloidal plane.
       > psi_sep, the value of the flux function psi on the separatrix.

    WARNING: The grid in R and z is returned in cm, not in m!
    """

    # Open and read the topfile
    filename1 = data_directory + "/topfile"
    filename2 = data_directory + "/TOPFILE"
    try:
        datafile = open(filename1)
    except IOError:
        datafile = open(filename2)
    lines = datafile.readlines()
    
    # Close the data file
    datafile.close()

    # Read the number of grid points
    datastring = lines[1].split()
    nptR = int(datastring[0])
    nptz = int(datastring[1])
    try:
        datastring = lines[3].split()
        psi_sep = float(datastring[2])
    except IndexError:
        print('\n WARNING topfile: assuming psi = 1. at the separatrix. \n')
        psi_sep = 1.

    # Define lists for variables
    R_val = []         # major radius grid
    z_val = []         # z coordinate grid
    br_val = []        # values of B, R component
    bz_val = []        # values of B, z component
    bt_val = []        # values of B, toroidal component
    psi_val = []       # values of poloidal flux

    # Loop over remaining lines 
    nlines = len(lines)
    for i in range(0, nlines):
        data = lines[i].split()
        # Read grid in major radius
        try:

            test = (data[0] == 'R') or (data[0] == 'Radial')
        except:
            # ... when the line is blank cycle to the next line ...
            continue
        try:
            # ... in some format the information is on the second
            # element of the line ...
            test = test or (data[1] == 'X-coordinates')
        except:
            pass
        if test:
            start = i + 1
            for line in lines[start:]:
                values = map(float, line.split())
                R_val.extend(values)
                if len(R_val) == nptR: break
        # Read grid in the vertical coordinate z
        try:
            # ... try to read the first element of the line ...
            test = (data[0] == 'Z') or (data[0] == 'Vertical')
        except:
            # ... when the line is blank cycle to the next line ...
            continue
        try:
            # ... in some format the information is on the second
            # element of the line ...
            test = test or (data[1] == 'Z-coordinates')
        except:
            pass
        if test:
            start = i + 1
            for line in lines[start:]:
                values = map(float, line.split())
                z_val.extend(values)
                if len(z_val) == nptz: break

        # Read the RADIAL component of the magnetic field
        try:
            # ... try to read the first element of the line ...
            test = (data[0] == 'Br') or (data[0] == 'B_r')
        except:
            # ... when the line is blank cycle to the next line ...
            continue            
        try:
            # ... in some format the information is on the second
            # element of the line ...
            test = test or (data[2] == 'B_R')
        except:
            pass
        if test:
            start = i + 1
            for line in lines[start:]:
                values = map(float, line.split())
                br_val.extend(values)
                if len(br_val) == nptR * nptz: break

        # Read the TOROIDAL component of the magnetic field
        try:
            # ... try to read the first element of the line ...
            test = (data[0] == 'Bt') or (data[0] == 'B_t') or \
                (data[0] == 'B_phi')
        except:
            # ... when the line is blank cycle to the next line ...
            continue
        try:
            # ... in some format the information is on the second
            # element of the line ...
            test = test or (data[2] == 'B_t')
        except:
            pass
        if test:
            start = i + 1
            for line in lines[start:]:
                values = map(float, line.split())
                bt_val.extend(values)
                if len(bt_val) == nptR * nptz: break

        # Read the z component of the magnetic field
        try:
            # ... try to read the first element of the line ...
            test = (data[0] == 'Bz') or (data[0] == 'B_z')
        except:
            # ... when the line is blank cycle to the next line ...
            continue
        try:
            # ... in some format the information is on the second
            # element of the line ...
            test = test or (data[2] == 'B_Z')
        except:
            pass
        if test:
            start = i + 1
            for line in lines[start:]:
                values = map(float, line.split())
                bz_val.extend(values)
                if len(bz_val) == nptR * nptz: break

        # Read the poloidal magnetic flux
        try:
            # ... try to read the first element of the line ...
            test = (data[0] == 'psi') or (data[0] == 'psi_pol')
        except:
            # ... when the line is blank cycle to the next line ...
            continue
        try:
            # ... in some format the information is on the first
            # element of the line ...
            test = test or (data[1] == 'psi')
        except:
            pass
        try:
            # ... in some format the information is on the second
            # element of the line ...
            test = test or (data[2] == 'psi')
        except:
            pass
        if test:
            start = i + 1
            for line in lines[start:]:
                values = map(float, line.split())
                psi_val.extend(values)
                if len(psi_val) == nptR * nptz: break

    # Convert lists of grid points into numpy arrays (converted in cm)
    R = 100. * np.array(R_val)
    z = 100. * np.array(z_val)
    z, R = np.meshgrid(z, R)

    # Convert the magnetic field lists into a numpy array
    br_val = np.array(br_val)
    br_val.shape = (nptz, nptR) #(data are written with z first)
    br_val = br_val.T #(transpose to have R first)
    #
    bz_val = np.array(bz_val)
    bz_val.shape = (nptz, nptR) #(data are written with z first)
    bz_val = bz_val.T #(transpose to have R first)
    #
    bt_val = np.array(bt_val)
    bt_val.shape = (nptz, nptR) #(data are written with z first)
    bt_val = bt_val.T #(transpose to have R first)
    #
    Bfield = np.array([br_val, bz_val, bt_val])

    # Convert the poloidal magnetic flux list into numpy array
    psi = np.array(psi_val)
    psi.shape = (nptz, nptR) #(data are written with z first)
    psi = psi.T #(transpose to have R first)

    # Return
    return (R, z, Bfield, psi, psi_sep)

#
# end of file
