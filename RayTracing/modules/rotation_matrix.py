"""
Matrix of the rotation operator which maps the anteanna coordinates into 
the laboratory coordinates.
"""

import math
import numpy as np

###########################################################################
# DEFINE A ROUTINE WHICH RETURNS THE ROTATION MATRIX TO TURN
# THE INITIAL CONDITIONS IN A SYSTEM ALINED TO THE ANTENNA 
# TO THE CARTESIAN COORDINATE SYSTEM X,Y,Z
###########################################################################
def rotMatrix(theta,phi):
    
    T = np.empty([3,3])
    T[0,0] = -math.cos(phi)*math.sin(theta)
    T[1,0] = -math.sin(phi)*math.sin(theta)
    T[2,0] =               -math.cos(theta)
    T[0,1] =  math.sin(phi)
    T[1,1] = -math.cos(phi)
    T[2,1] =  0.
    T[0,2] = -math.cos(phi)*math.cos(theta)
    T[1,2] = -math.sin(phi)*math.cos(theta)
    T[2,2] =                math.sin(theta)

    return T

    
# END OF FILE
