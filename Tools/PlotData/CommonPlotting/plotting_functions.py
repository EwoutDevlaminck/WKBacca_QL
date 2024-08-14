
"""
Collection of commonly used plotting functions.
"""

import warnings
import numpy as np


# We want to plot the contours corresponding to the
# the cyclotrob harmonics omega = n omega_ce for n=1,2,3, the cut-offs
# and other wave-wave resonances. Depending on the frequency the some or
# all contours might be empty returning in a warning.
# We want to catch the warning and issue better information to the user ...

# ... All warnings triggered ...
warnings.simplefilter("always")

# ... The warning message to catch is...
wmsg = 'No contour levels were found within the data range.'


# Plotting the cyclotron resonances
def add_cyclotron_resonances(R, Z, StixY, axes):

    """
    Plot the locus of point that satisfy the resonance condition StixY = 1/n,
    for n=1,2,3, where StixY = omega_ce/omega.

    Usage: 
           h1, h2, h3 = add_cyclotron_resonances(R, Z, StixY, axes)
           
    where R, Z are coordinates in the poloidal plane and StixY is the
    Stix parameter X = omega_ce/omega as a function of (R,Z). The last argument
    is the axes to be polulated.  

    The returned objects h1, h2, h3 refer to the first, second and third harmonics
    respectively and can be used, for instance, to build a colorbar or a legend.
    """

    with warnings.catch_warnings(record=True) as warn:
        
        # This counter is used to check if new warnings are issued
        len_warn = 0

        # Try to plot the third-harmonic resonance ...
        h3 = axes.contour(R, Z, StixY, [0.33], colors='lime', linestyles='dashed')
        # ... check the last (-1) warning ...
        if len(warn) > len_warn and str(warn[0].message) == wmsg:
            
            print('The third-haronic resonance is not in the domain.')
            len_warn += 1

        # ... try to plot the second-harmonic resonance ...
        h2 = axes.contour(R, Z, StixY, [0.50], colors='lime', linestyles='dashdot')
        # ... check the last (-1) warning ...
        if len(warn) > len_warn and str(warn[0].message) == wmsg:

            print('The second-haronic resonance is not in the domain.')
            len_warn += 1

        # ... try to plot the first-harmonic resonance ...
        h1 = axes.contour(R, Z, StixY, [1.00], colors='lime', linestyles='dotted')
        # ... check the last (-1) warning ...
        if len(warn) > len_warn and str(warn[0].message) == wmsg:

            print('The first-haronic resonance is not in the domain.')
    
    return h1, h2, h3


# Plotting the O-mode cutoff    
def add_Omode_cutoff(R, Z, StixX, axes):

    """
    Add to a plot in the R,Z plane, a curve for the level set StixX = 1,
    which corresponds to the O-mode cutoff.

    Usage: 
            contour = add_Omode_cutoff(R, Z, StixX, axes)
           
    where R, Z are coordinates in the poloidal plane and StixX is the
    Stix parameter Y = omega_pe^2/omega^2 as a function of (R,Z). The last argument
    is the axes to be polulated.  

    The returned object can be used, for instance, to build a colorbar or a legend.
    """

    # If present in the domain add line for the O-mode cut-off which is 
    # the level set StixX == 1.

    # If the O-mode is not in the domain, the level set is empty and
    # a warning is issued. We want to catch the warning
    with warnings.catch_warnings(record=True) as warn:

        # ... O-mode cut-off
        O_cutoff = axes.contour(R, Z, StixX, [1.], colors='g', linestyles='dashed')

        # ... check the last (-1) warning ...
        if len(warn)>0 and str(warn[-1].message) == wmsg:
            print('The O-Mode cut-off surface not found in the domain.')
            
    return O_cutoff


def add_Xmode_cutoff(R, Z, StixX, StixY, axes):

    """
    Add a plot of the X-mode cutoff to the give axes. The X-mode cutoff is given
    by the condition
    
       (Y/2) + sqrt(X + (Y/2)^2) = 1,
    
    where X and Y are the standard Stix parameters.

    Usage: 
           contour = add_Xmode_cutoff(R, Z, StixX, StixY, axes)
           
    where R, Z are coordinates in the poloidal plane, whereas StixX and StixY 
    are the Stix parameters X = omega_pe^2/omega^2 and Y = omega_ce/omega 
    as functions of (R,Z). The last argument is the axes to be polulated. 
    
    The returned object can be used, for instance, to build a colorbar or a legend.
    """

    Xcutoff = 0.5*StixY + np.sqrt(StixX + (0.5*StixY)**2)

    # If the level set is empty, the UH resonance is not present in the domain
    # and a warning is issued. We want to catch this warning
    with warnings.catch_warnings(record=True) as warn:

        # ... Upper-hybrid resonance for perpendicular propagation
        X_cutoff = axes.contour(R, Z, Xcutoff, [1.0], colors='g')

        # ... check the last (-1) warning ...
        if len(warn) > 0 and str(warn[-1].message) == wmsg:
            print('The O-Mode cut-off surface not found in the domain.')
            
    return X_cutoff


def add_UHresonance(R, Z, StixX, StixY, axes):

    """
    Plot the upper-hybrid resonance for the case of exactly perpendicular
    propagation as an indication on the position of the upper-hybrid resonance.

    Usage: 
            contour = add_UHresonance(R, Z, StixX, StixY, axes)
           
    where R, Z are coordinates in the poloidal plane, whereas StixX and StixY 
    are the Stix parameters X = omega_pe^2/omega^2 and Y = omega_ce/omega 
    as functions of (R,Z). The last argument is the axes to be polulated.  

    The returned object can be used, for instance, to build a colorbar or a legend.
    """

    # For perpendicular propagation, the upper-hybrid (UH) resonance is
    # the level set StixX + StixY**2 == 1. If present in the domain, add
    # this contour line to the axes. 

    # If the level set is empty, the UH resonance is not present in the domain
    # and a warning is issued. We want to catch this warning
    with warnings.catch_warnings(record=True) as warn:

        # ... Upper-hybrid resonance for perpendicular propagation
        UH_res = axes.contour(R, Z, StixX + StixY**2, [1.0], colors='m')

        # ... check the last (-1) warning ...
        if len(warn)>0 and str(warn[-1].message) == wmsg:
            print('The upper-hybrid resonance X+Y^2=1 is not in the domain.')
    
    return UH_res
