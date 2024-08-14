"""This function provides a somewhat more advanced atan function.
It calculates the angle out of the given OL and AL and therefore chooses
the right branch of atan."""


cdef extern from "math.h":
   double atan ( double )
  
cdef double pi = 3.141592653589


# Here, a somewhat more advanced atan function is defined.
# It automatically choses the right branch:
# calculate an angle out of the opposite leg and the adjacent leg
# making sure that the right branch of atan is used
# 
#             OL
#             ^      *
#             |    *
#             |  *
#             |* angle
#  --------------------------> AL
#             |
#             |
#             |
#             |
#
# returns angle in (-pi, +pi)
cpdef double atanRightBranch(double OL,double AL):
     
    """Takes as arguments the OL and AL of a triangle
    and returns the angle out of the interval (-pi, pi).
    Therefore, the right branch of atan is chosen."""
    cdef double __angle
    

    if (AL != 0):
        #atan gives values in (-pi/2,+pi/2)
        __angle = atan(OL / AL)    
    else:   
        # in case AL  = 0, define the angle by hand
        if (OL > 0):
            __angle = pi / 2.
        else:
            __angle = -pi / 2.

    if (OL >= 0) and (AL < 0):
        __angle = pi + __angle
    elif (OL < 0) and (AL < 0):
        __angle = -pi + __angle      
    
    return __angle
