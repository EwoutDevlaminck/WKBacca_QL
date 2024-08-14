"""
Check WKBeam dispersion relation versus explicit formulas
given in the review paper by Bornatici et al. [Nuclear Fucion 23 (1983) 1153],
equations (3.1.55a) and (3.1.55b).
"""

# Flags for the propagation modes
modes = {+1: 'O-mode', -1: 'X-mode'} # sigma index: mode label


# Load coefficients S, D, and P
def __loadSDP__(X, Y):

    denom = 1. - Y**2
    S = 1. - X / denom
    D = -X * Y / denom
    P = 1. - X
    
    return S, D, P


# WKBeam dispersion relation (the squared perpendicular refractive index)
def Nperp2_WKBeam(X, Y, sigma, Nparal2, epsilon=1.e-9):
    
    """
    Evaluate the dispersion relation in equation (7.34)
    with the regularization (7.38) of Weber IPP report, IPP_5_134.
    The dispersion relation is solved for Nperp^2 as a function of Nparal
    and N^2 = N_paral^2 + Nperp^2 is returned. 
    
    Input arguments:

       X = omega_p^2 / omega^2  
       Y = omega_c / omega
       sigma = mode index (+1 for the O-mode, -1 for the X-mode)
       Nparal2 = squared parallel refractive index
       epsilon = (opional) small regularization parameter

    With the only exception of epsilon and sigma, all input arguments 
    can be either scalar or numpy array of the same shape. The parameters 
    epsilon must always be a scalar and sigma = -1, +1.
    """
    
    import numpy as np

    # Stix parameters
    P = 1. - X

    # Coefficient of the bi-quadatic form
    # (corrected for compensation of round-off errors)
    Y2 = Y**2
    P2 = P**2
    A = P - Y2
    B = (1. + P) * Y2 - 2. * P2 + (2. * (P - Y2) + X * Y2) * Nparal2
    C = P * (P2 - Y2) - 2. * P * (P - Y2) * Nparal2 + P * (1. - Y2) * Nparal2**2

    # Discriminant function
    F2 = B**2 - 4. * A * C
    F2[F2 < 0.] = 0.                   # (remove small negative values)
    F = np.sqrt(F2)
    
    # Solution of the bi-quadratic equation for the squared perpendicular
    # refractive index with regularized inverse
    invA = A / (A**2 + epsilon**2)
    Nperp2 = -0.5 * invA *  (B - sigma * F)

    Nperp2[Nparal2 < 0.] = np.nan      # (Nparal2<0 --> meaningless input)

    return Nperp2


# WKBeam dispersion relation (the total squared refractive index)
def N2_WKBeam(X, Y, sigma, Nparal2, epsilon=1.e-9):
    
    """
    Call Nperp2_WKBeam and build N^2 = Nperp^2 + Nparallel^2.
    Cf. Nperp2_WKBeam doc string.
    """
    
    Nperp2 = Nperp2_WKBeam(X, Y, sigma, Nparal2, epsilon=1.e-9)
    
    return  Nparal2 + Nperp2

 
# WKBeam Hamiltonian
def H_WKBeam(X, Y, sigma, Nperp2, Nparal2, epsilon=1.e-9):
    
    """
    Call Nperp2_WKBeam and build H = 2. Nperp^2 + f(Nparal2), where

       f(Nparal2) = (B - sigma * F) / A

    Cf. Nperp2_WKBeam doc string.
    """
    
    Nperp2WKBeam = Nperp2_WKBeam(X, Y, sigma, Nparal2, epsilon=1.e-9)

    return Nperp2 - Nperp2WKBeam 


# Expression given in Maj, Ph.D. Thesis 2003
def N2_Maj(X, Y, sigma, cos_theta, epsilon=1.e-9):

    """
    Evaluate the squared absolute value of the refractive index as in 
    equation (C.18) of Maj Ph.D. Thesis (2003).
    
    Input arguments:

       X = omega_p^2 / omega^2  
       Y = omega_c / omega
       sigma = mode index (+1 for the O-mode, -1 for the X-mode)
       cos_theta = cosine of the angle between N and B

    With the only exception of sigma, all input arguments can be either 
    scalar or numpy array of the same shape.
    """
    
    import numpy as np

    cos_theta2 = cos_theta**2
    sin_theta2 = 1. - cos_theta2

    # Stix parameters
    P = 1. - X

    # Coefficients of the bi-quedratic form
    # ('tilde coefficients' are modifided to avoid round-off errors!!)
    A = P * (1. - Y**2) * cos_theta2 + (P - Y**2) * sin_theta2
    B = P * (P - Y**2) * (1. + cos_theta2) + (P**2 - Y**2) * sin_theta2
    F = X * Y * np.sqrt(Y**2 * sin_theta2**2 + 4. * P**2 * cos_theta2 )

    # Solution of the bi-quadratic equation with regularized inverse
    invA = A / (A**2 + epsilon**2)
    N2 = 0.5 * invA * (B + sigma * F) 

    return N2


# Explicit refractive index by Bornatici at al.
def N2_Bornatici(X, Y, sigma, cos_theta, epsilon=1.e-9):

    """
    Squared absolute value of the refractive index vector according
    to equation (3.1.55) of Bornatici at al. Nuclear Fusion 23 (1983), 1153.

    Slightly modified from the version kindly offered by Severin Denk.
    """

    import numpy as np

    # Angle theta between N and B
    cos_theta2 = cos_theta**2
    sin_theta2 = 1. - cos_theta2
    
    # The analytical formula 
    # ... original formula from the paper
    # rho =  sin_theta2**2 + 4. * ((1 - X) / Y)**2 * cos_theta2
    # rho = np.sqrt(rho)
    # f = 2. * (1. - X) / (2. * (1. - X) - 
    #                      Y**2 * (sin_theta2 - sigma * rho) ) 
    # ... modinfied in order to avoid a division by zero for Y==0 ...
    rho2 =  Y**4 * sin_theta2**2 + 4. * Y**2 * (1 - X)**2 * cos_theta2
    rho = np.sqrt(rho2)
    D = 1. - X - 0.5 * (Y**2 * sin_theta2 - sigma * rho)
    invD = D / (D**2 + epsilon**2)
    N2 = 1. - X * (1. - X) * invD

    return N2


# Compare the dispersion curves
def compare_dispersions(X, Y, sigma, npttheta, epsilon=1.e-9, fig=1):

    """
    Plot the absolute value of the refractive index obtaind from the
    WKBeam Hamiltonian and from the exact dispersion formula by 
    Bornatic et al. Nuclear Fusion 23 (1983), 1153.
    """

    import numpy as np
    import matplotlib.pyplot as plt

    if sigma==1. or sigma==-1:
        mode = modes[sigma]
    else:
        raise RuntimeError('sigma must be either 1 (O-mode) or -1 (X-mode)')

    # Analytical expression from Bronatici et al.
    theta = np.linspace(0., 2.*np.pi, npttheta)
    cos_theta = np.cos(theta)
    N2Bornatici = N2_Bornatici(X, Y, sigma, cos_theta, epsilon=epsilon)
    N2Maj = N2_Maj(X, Y, sigma, cos_theta, epsilon=epsilon)

    # Sampling the WKBeam Hamiltonian on a uniform mesh
    Nparal2 = N2Bornatici * cos_theta**2
    N2WKBeam = N2_WKBeam(X, Y, sigma, Nparal2, epsilon=epsilon) 
    Nperp2WKBeam = Nperp2_WKBeam(X, Y, sigma, Nparal2, epsilon=epsilon)
    HWKBeam = H_WKBeam(X, Y, sigma, Nperp2WKBeam, Nparal2, epsilon=epsilon)

    # Sample the WKBeam Hamiltonian with the refractive index obtained
    # from the analytical formula of Bornatici et al.
    Nperp2Bornatici = N2Bornatici - Nparal2
    Nperp2Bornatici[Nparal2<0.] = np.nan
    HWKBeam = H_WKBeam(X, Y, sigma, Nperp2Bornatici, Nparal2, 
                       epsilon=epsilon)

    # Plotting
    plt.figure(fig)
    plt.subplot(111, polar=True)
    plt.plot(theta, N2Bornatici, 'g', linewidth=2, label='Barnatici NF 1983')
    plt.plot(theta, N2Maj, 'xr', label='Maj Ph.D. 2003')
    plt.plot(theta, N2WKBeam, '+b', label='WKBeam')
    plt.legend(bbox_to_anchor=(0.76, 0.1), loc=2)
    plt.title('X = {}, Y = {}, {}'.format(X, Y, mode) )
    plt.figure(fig+1)
    plt.plot(theta, HWKBeam)
    plt.xlabel(r'$\theta$')
    plt.show()


# Scan in magnetic field at constant density
def scan_Bfield(X, angles, Ymax, nptY, epsilon=1.e-9, fig=3):
    
    """
    For a fixed X and for a given set of angles angles = [theta1, theta2, ...]
    plot the dispersion function in Y over the intervale Y in [0, Ymax].
    """

    import numpy as np
    import matplotlib.pyplot as plt

    if sigma==1.:
        mode = 'O-mode'
    elif sigma==-1.:
        mode = 'X-mode'
    else:
        raise RuntimeError('sigma must be either 1 (O-mode) or -1 (X-mode)')

    # Define the scan parameter
    Y = np.linspace(0., Ymax, nptY)

    # Plotting
    fig = fig
    for theta in angles:
        cos_theta = np.cos(theta)
        plt.figure(fig)

        # Bornatici et al.
        N2Bornatici = N2_Bornatici(X, Y, sigma, cos_theta, epsilon=epsilon)
        plt.plot(Y, N2Bornatici, '.-g',
                 label='Bornatici, theta = {}'.format(round(theta,2)))

        # Maj 2003
        N2Maj = N2_Maj(X, Y, sigma, cos_theta, epsilon=epsilon)
        plt.plot(Y, N2Maj, 'xr', 
                 label='Maj, theta = {}'.format(round(theta,2)))

        # WKBeam
        Nparal2 = N2Bornatici * cos_theta**2
        N2WKBeam = N2_WKBeam(X, Y, sigma, Nparal2, epsilon=epsilon)
        plt.plot(Y, N2WKBeam, '+b', 
                 label='WKBeam, theta = {}'.format(round(theta,2)))

        plt.xlabel('Y')
        plt.ylabel('N^2')
        plt.title(r'X = {}, $\vartheta = ${}, {}'.format(X, theta, mode))
        plt.legend()
        plt.grid('on')
        
        fig += 1
 
    plt.show()


# Squared refractive indices for parallel propagation
def parallel_prop(X, Y, sigma):
    
    """
    N^2 for parallel propagation, i.e., theta = 0, pi.
    """
    
    try:
        N2 = 1. - X / (1. + sigma * Y)
    except ZeroDivisionError:
        from numpy import inf
        N2 = inf

    return N2

       
# Squared refractive indices for perpendicular propagation
def perpendicular_prop(X, Y, sigma):
    
    """
    N^2 for perpendicular propagation, i.e., theta = pi/2, 3 pi /2 .
    """
    
    if sigma == 1.:
        # O-mode
        N2 =  1. - X 
    elif sigma == -1.:
        # X mode
        try:
            N2 =  1. - X * (1. - X) / (1. - X - Y**2)
        except ZeroDivisionError:
            from numpy import inf
            N2 = inf

    return N2


# Testing
if __name__=='__main__':
    
    from math import sqrt, pi
    from sys import argv

    # Plasma parameter
    try:
        X = float(argv[1])
        Y = float(argv[2])
        sigma = float(argv[3])
    except:
        X = 1.2 ##1.9 ###0.8
        Y = 1.2 ###0.5
        sigma = +1.
    # Numerical parameters
    npttheta = 201 # (better to use an odd number of points)
    angles = [0., 0.5*pi, 0.33*pi]
    Ymax = 2.5
    nptY = 101
    epsilon = 1.e-9

    # N at theta=0deg and theta=90deg
    N2_0 = parallel_prop(X, Y, sigma)
    N2_90 = perpendicular_prop(X, Y, sigma)
    print('\n')
    print( 'X = {}, Y = {}, mode = {}'.format(X, Y, sigma) )
    print('\n')
    print( 'Parallel propagation: N^2 = {}'.format(N2_0) )
    print( 'Perpendicular propagation: N^2 = {}'.format(N2_90) )

    # Testing
    compare_dispersions(X, Y, sigma, npttheta, epsilon=epsilon)
    scan_Bfield(X, angles, Ymax, nptY, epsilon=epsilon) 

# End of file
