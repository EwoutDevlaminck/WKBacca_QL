"""
Definition of physics constants and parameters used throughout the code.
This might include function defining physical parameters in terms of basic 
physical constants. 
"""

# import statements
import math



############################################################################
# BASIC PHYSICAL CONSTANTS
############################################################################

# Speed of light in free space in cm/s
SpeedOfLight = 2.9979e10 


############################################################################
# BASIC MATHEMATICAL CONSTANTS
############################################################################

# pi
pi =  math.pi



############################################################################
# DERIVED QUANTITIES
############################################################################

# Angular frequency (radian/s) given the frequency in GHz
AngularFrequency = lambda f: 2. * pi * f * 1.e9

# Wave number (cm^-1) in free space given the angular frequency (radian/s) 
WaveNumber = lambda omega: omega / SpeedOfLight

# Eigenvalue of the matrix D, given the beam width w in cm and the wave
# number k0 in free space in cm^-1
EigenvalueD = lambda k0, w: 2. / k0 / w**2

# Eigenvalue of the matrix S, given the radius of curvature R in cm
EigenvalueS = lambda R: 1. / R
