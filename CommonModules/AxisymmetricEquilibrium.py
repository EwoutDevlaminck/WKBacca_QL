"""
Read plasma equilibrium for axisymmetric devices.
The data format stores the coordinates (R,Z) of the grid
in the poloidal plane of the device along with the value of the
radial, vertical, and toroidal components of the equilibrium magnetic
field as well as the density in the form

n m  # optional line
R1 R2 ... Rn
Z1 Z2 ... Zm
X(R1,Z1) X(R2,Z1) ... X(Rn, Z1)
X(R1,Z2) X(R2,Z2) ... X(Rn, Z2)
...
X(R1,Zm) X(R2,Zm) ... X(Rn, Zm)

where X stands for the components BR, BZ, and Btor of the magnetic field
in Tasla and the electron density density Ne in m^-3. 
Those quantities are stored in the files:

- B_V_x.txt for the radial component BR of the magnetic field;
- B_V_z.txt for the vertical component BZ of the magnetic field;
- B_Tbc_y.txt for the toroidal component Btor of the magnetic field;
- ne.txt for the electron density.

This module and the corresponding data format have been developed 
specifically for the TORPEX machine in the framework of the RFSCAT
EUROfusion Enabling Reasearch Project.
"""


# Import statements
import numpy as np


# Read a line of floting point numbers and convert to ndarray
def read_line(line):
    """
    Given a string obtained from the readline method of a file
    object, convert it to an ndarray. This assumes that only
    floating point number are included in the line, with at most
    the possibility of NaN entries.
    
    USAGE:
           a = read_line(line)
    INPUT:
           - line, string, line of data in string format.
    OUTPUT:
           - a, ndarray ndim=1, conveted array (possibly including
             NaNs and Infs, if present i the data set).
    """

    splt = line.split()
    n = len(splt)
    a = np.empty([n])
    index = 0
    for element in splt:
        a[index] = float(element)
        index += 1
    
    return a

# General function for loading the specific file format
def load_2D_profile(dir, filename):
    """
    Load data from files formatted as described in the 
    parent module doc-string.
    
    USAGE:
          R, Z, Data = load_2d_profile(dir, filename)
          
    INPUT ARGUMENTS:
          - dir, string, path to the directory of the data file.
          - filename, string, name of the target data file.

    OUTPUT:
          - R, ndarray ndim=1, radial grid points R[i].
          - Z, ndarray ndim=1, vertical grid points Z[j].
          - Data, ndarray, ndim=2, grid values Data[i,j].
    """
    
    # Open data file
    datafile = open(dir+"/"+filename, 'r')
    raw_data = datafile.readlines()
    datafile.close()

    # Extract the grid size
    # (the first line may or may not store n, m)
    iline = 0    
    dataoffset = 1
    line = raw_data[iline]
    sl = line.split()
    try:
        n = int(sl[0]);  m = int(sl[1])
    except ValueError:
        dataoffset = 0
    
    # Read the radial grid 
    iline = dataoffset
    line = raw_data[iline]
    R = read_line(line)

    # Read the vertical grid
    iline = dataoffset + 1
    line = raw_data[iline]
    Z = read_line(line)

    # Read the values
    nrows = dataoffset + 2 # (number of rows that must be skipped)
    Data = np.loadtxt(dir+"/"+filename, skiprows=nrows, ndmin=2)

    # Traspose to have the R-index first
    Data = Data.T

    # If n and m are given, check dimensions
#    ..... to be added in case (ask for a clear format definition?)

    # Return data
    return R, Z, Data

# Main function for reading the data
def read_axisymmetric_equilibrium(dir):
    """
    Read data for the plasma equilibrium of axisymmetric devices
    form the files:
      - B_V_x.txt for the radial component BR of the magnetic field;
      - B_V_z.txt for the vertical component BZ of the magnetic field;
      - B_Tbc_y.txt for the toroidal component Btor of the magnetic field;
      - ne.txt for the electron density.
    Data are stored in the format:

      n m
      R1 R2 ... Rn
      Z1 Z2 ... Zm
      X(R1,Z1) X(R2,Z1) ... X(Rn, Z1)
      X(R1,Z2) X(R2,Z2) ... X(Rn, Z2)
      ...
      X(R1,Zm) X(R2,Zm) ... X(Rn, Zm)
    
    where X stends for the components BR, BZ, and Btor of the magnetic field
    in Tasla and the electron density density Ne in m^-3. Here, n is
    the number of grid points in the radial coordinates R and m is the number
    of grid points in the vertical coordinate Z.
    
    USAGE:
       radial_field, vertical_field, toroidal_field, \\ 
                  density = read_axisymmetric_equilibrium(dir)

    INPUT:
       dir (string) = path to the directory holding the files.
    
    OUTPUT:
       radial_field = (R_BR, Z_BR, BR)
       vertical_field = (R_BZ, Z_BZ, BZ)
       toroidal_field = (R_Bt, Z_Bt, Bt)
       density = (R_Ne, Z_Ne, Ne)       

    where 
       BR (ndarray, shape=(n,m)) = radial magnetic field in Tesla,
       BZ (ndarray, shape=(n,m)) = vertical magnetic field in Tesla,
       Bt (ndarray, shape=(n,m)) = toroidal magnetic field in Tesla,
       Ne (ndarray, shape=(n,m)) = electron density in m^-3,
    with R_?? being the corresponding radial grid (ndraay, ndim=1, size=n)
    and Z_?? the corresponding vertical grid (ndarray, ndim=1, size=m).
    Length are read in m and then converted in cm.
    """

    # Read the data from file
    R_BR, Z_BR, BR = load_2D_profile(dir, 'B_V_x.txt')   # radial field
    R_BZ, Z_BZ, BZ = load_2D_profile(dir, 'B_V_z.txt')   # vertical field
    R_Bt, Z_Bt, Bt = load_2D_profile(dir, 'B_Tbc_y.txt') # toroidal field
    R_Ne, Z_Ne, Ne = load_2D_profile(dir, 'ne.txt')      # density

    # Convert lengths from m to cm
    R_BR *= 100.; Z_BR *= 100.
    R_BZ *= 100.; Z_BZ *= 100.
    R_Bt *= 100.; Z_Bt *= 100.
    R_Ne *= 100.; Z_Ne *= 100.

    # Return raw arrays (no interpolation)
    radial_field = (R_BR, Z_BR, BR)
    vertical_field = (R_BZ, Z_BZ, BZ)
    toroidal_field = (R_Bt, Z_Bt, Bt)
    density = (R_Ne, Z_Ne, Ne)
    return radial_field, vertical_field, toroidal_field, density
