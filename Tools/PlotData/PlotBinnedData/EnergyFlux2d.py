"""
Set of tool to compute and visualize the wave energy flux
through a generic surface given parametrically.

This is for the two-dimensional calculation. Therefore the surface is actually
a path in the two-dimensional space. We assume consistency with the WKBeam
calculation with the flag twodim = True activated in raytracing file, that is,
the two-dimensional domain is the x-z plane (assuming y=0, Ny=0).

These procedures should only be used for two-dimensional test runs and not on
projected three-dimensional beam. In the latter case the results would have no
physical menaing.
"""


# Import statements
import collections
import numpy as np
import scipy.interpolate as Interp
import scipy.integrate as Integrate
import matplotlib.pyplot as plt

# -------------------------------------------------------------------------------
# Data structures

# Definition of a namedtuple spacifying parametric surfaces and fluxes
Line2D = collections.namedtuple('Line', ['phi', 'phi_u', 'phi_u_perp',
                                         'umin', 'umax', 'nptu'])
FluxVector2D = collections.namedtuple('FluxVector2D', ['x1', 'x2', 'F1', 'F2'])



# -------------------------------------------------------------------------------
# Main procedures

# This is the main procedure called by WKBeam
def flux_through_line_in_2d(idata, hdf5data, surface):

    """
    Takes data from the WKBeam run (input data idata, hdf5 dataset hdf5data,
    and the surface model surface_model) and construct an instance of the 
    named tuple FluxVector2D for the energy flux and Line2D for the surface in 
    two dimentions. The named tuples are then passed to the relevant procedure
    to compute the energy flux through the surface and visualize the normal
    component of the energy flux vector on the surface.
    
    Call signature:
    
       res = flux_through_line_in_2d(idata, hdf5data, surface)
    
    where idata is an instance of InputData, hdf5data is obtained from a 
    WKBeam binning hdf5 file, ans surface is surface-model object.
    
    The result is the tuple res = (computed_flux, data, field) where
    computed_flux (float) is the value of the flux in MW, data is another tuple
    containing the information on the flux on the surface, and field is a named 
    tuple which has information of the Poynting flux reconstracted on the domain.
    
    Specifically: 
    
      data = (u, x1, x2, FnJ, Fn)
               u, 1d array of the parameter of the line
        (x1, x2), position of sample points of the line in the 2D domain
             FnJ, normal flux times Jacobain J on the line
              Fn, normal flux.
    
      field = FluxVector2D(x1, x2, F1, F2)
    """
    
    # Check the velocity field stored (cut the last element which is empty)
    Vfield = hdf5data.get('VelocityFieldStored')[()].split(',')[0:-1]
    try:
        assert len(Vfield) == 2
    except AssertionError:
        msg = """Warning: More than two velocity componets found.
        Using the first two components."""
    
    # Grid 
    from Tools.PlotData.PlotBinnedData.grids import get_grid
    Coordinates = hdf5data.get('WhatToResolve')[()].split(',')[0:-1]
    x1, dx1 = get_grid(hdf5data, Coordinates[0])
    x2, dx2 = get_grid(hdf5data, Coordinates[1])

    # Energy flux Field
    F1 = hdf5data.get('VelocityField')[...,0,0] / dx1 / dx2 
    F2 = hdf5data.get('VelocityField')[...,1,0] / dx1 / dx2 

    # Define the flux vector    
    field = FluxVector2D(x1, x2, F1, F2)

    # Check the default for the surface model
    if hasattr(surface, 'umin'):
        umin = surface.umin
    else:
        umin=0.0
    if hasattr(surface, 'umax'):
        umax = surface.umax
    else:
        umax=1.0
    if hasattr(surface, 'nptu'):
        nptu = surface.nptu
        # Simpson quadrature rule requires an odd number of points 
        # (even number of intervals) 
        if nptu % 2 == 0: nptu += 1
    else:
        nptu=101      

    # Define the section in two-dimensions (namely a line)
    line = Line2D(surface.phi, surface.phi_u, surface.phi_u_perp, 
                  umin, umax, nptu)
    
    # At last compute the flux
    computed_flux, data = compute_flux(field, line)
    u, x1, x2, FnJ, Fn = data 
    plot_Fn(field, x1, x2, Fn, density=idata.density_of_flux_lines)
    
    return computed_flux, data, field


# Auxiliary function which perform most of the field and line
# reconstruction needed to compute the energy flux
def build_FnJ(field, line, interpolate_field=True, s=0.0):

    """
    Compute the scalar product of the vector field with the unit normal times
    the Jacobian of the line element, namely
    
       F .ndl = F.n J du
    
    where u is the parameter of the line, n is the unit normal.
    
    If interpolate_field=True, then a Bivariate spline with smooth s is applied
    in order to get field value ot the needed points on the line.
    
    If on the other hand, interpolate_field=False, then field is supposed to
    be a callable object and field(x,z) returns the value of the field at (x,z).
 
    This function returns the tuple
    
       (u, x, z, FnJ, Fn)
    
    where u is the one-dimensional grids, x,z the Cartesian coordinates of the 
    points of the line, and FnJ is the value of the scalar product F.nJ on the 
    point corresponding to the parameter u and Fn = F.nJ / J is the normal
    component of the flux without the Jacobian.
    """

    # Sample the points on the line. The surface is given parametrically
    u = np.linspace(line.umin, line.umax, line.nptu)
    xz = np.array([line.phi(u[i]) for i in range(line.nptu)])
    x, z = xz.T

    # Contruction of the interpolant for the vector field
    if interpolate_field:

        Vx = Interp.RectBivariateSpline(field.x1, field.x2, field.F1[:,:], s=s)
        Vz = Interp.RectBivariateSpline(field.x1, field.x2, field.F2[:,:], s=s)
        
        Vx_on_line = Vx(x, z, grid=False)
        Vz_on_line = Vz(x, z, grid=False)
        V_on_line = [(Vx_on_line[i], Vz_on_line[i]) for i in range(line.nptu)]

    else:

        V_on_line = [field(xz[i]) for i in range(line.nptu)]

    # Unit normals and area element:
    # nJ is the normal vector n times the Jacobian J defined so that
    # the element of area is dS = Jdudv
    phi_u = [line.phi_u(u[i]) for i in range(line.nptu)]
    phi_u_perp = [line.phi_u_perp(u[i]) for i in range(line.nptu)]
    J = [np.linalg.norm(phi_u[i]) for i in range(line.nptu)]
    integrand = [np.dot(V_on_line[i], phi_u_perp[i]) for i in range(line.nptu)]

    # Reshaping to get the integrand on the surface
    x, z  = xz.T
    FnJ = np.array(integrand)
    J = np.array(J)
    Fn = FnJ/J

    return u, x, z, FnJ, Fn
    

# Main function which computes fluxes through surfaces
def compute_flux(field, line, interpolate_field=True, s=0.0):

    """
    Compute the flux of a vector field defined on a regular two-dimensional
    grid through a line (surface in two dimensions) given parametrically.

    USAGE:
    
        compute_flux(field, line, interpolate_field=True, s=0.0)
    
    where for the intended use:
      - "field" represents the vector field and is an instance on the 
        nemedtuple FluxVector2D,
      - "line" represents the line through which the flux is computed 
        and it is an instance of the namedtuple Line2D,
    
    Optional arguments:
      - interpolate_field, is a boolean variable. When True it triggers the 
        intended behaviour described above. When False, field is assumed to be
        a callable object which evaluated on the point give the exact vector
        with no need for interpolation. The callable object should be of the 
        form lambda (x,z): flux_vector(*(x,z)). This is used for testing only.
      - s, spline smoothing parameter.
    """

    # Compute the vector field F on the surface and the scalar product F.n J
    # where n is the unit normal and J the Jacobian of the area element, i.e., 
    # dS = J dudv.
    u, x1, x2, FnJ, Fn = build_FnJ(field, line,
                                   interpolate_field=interpolate_field, s=s)
    
    # Simpson quadrature applied to each dimension
    computed_flux = Integrate.simps(FnJ, u)

    print('\n Computed flux = {}\n'.format(computed_flux))

    return computed_flux, [u, x1, x2, FnJ, Fn] 



# Main function which computes fluxes through surfaces
def plot_Fn(field, x1, x2, Fn, density=1.0):

    """
    The same as compute flux, but instead of juyt applying athe quadarture
    it also plot the points of the line color coded with the value of the 
    normal flux Fn.
    """

    # Mappable for the normal component
    # This is used to color point of the line with the value of the normal
    # component of the flux vector
    m = plt.cm.ScalarMappable(cmap='coolwarm')
    m.set_array([])
    fcolors = m.to_rgba(Fn)

    # Plotting
    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(111)
    ax.streamplot(field.x1.T, field.x2.T, field.F1.T, field.F2.T,
                  color=field.F1.T**2 + field.F2.T**2, density=density,
                  cmap='cool')
    ax.scatter(x1, x2, c=fcolors, linewidth=0.0)
    ax.set_aspect('equal', 'datalim')
    ax.set_xlabel('$x_1$', fontsize=20)
    ax.set_ylabel('$x_2$', fontsize=20)
    fig.colorbar(m, shrink=0.5, aspect=5)
    plt.show()

    return None

    

# -------------------------------------------------------------------------------
# Testing procedures


# Generate test data for flux through a cross-section of the beam
def __generate_data_for_test_on_cross_section(test_beam, N=None):

    # It is convenient for the test to define the box first
    # and then the surface which must be included in the box
    z0 = 0.5          # anchor of the surface
    Lx = 6.0          # length in x
    dx = 0.005
    dz = 0.0053       # avoid to have a point at z = 0

    if N == None:
        nptu = 141
    else:
        nptu = N

    xmin = -0.5 * Lx
    xmax = +0.5 * Lx
    zmin = z0 - 1.5
    zmax = z0 + 1.5

    # Define a surface parametrically in 2d
    # (and provide the derivative of the mapping as well)
    x0 = np.array([xmin, z0])
    ex = np.array([Lx, 0.0])
    ex_perp = np.array([0.0, Lx])
    
    phi = lambda u: x0 + ex * u
    phi_u = lambda u: ex
    phi_u_perp = lambda u: ex_perp
    
    discr_line = Line2D(phi=phi, phi_u=phi_u, phi_u_perp=phi_u_perp,
                        umin=0.0, umax=1.0, nptu=nptu)

    # Define the 2d spatial grid 
    # (It must be big enough to contain the surface)
    x, z = np.mgrid[slice(xmin, xmax+dx, dx), slice(zmin, zmax+dz, dz)]

    # Energy flux of the beam
    Fx, Fz = test_beam.flux(x, z)
    flux_vector_field = FluxVector2D(x1=x[:,0], x2=z[0,:], F1=Fx, F2=Fz)

    return discr_line, flux_vector_field
    

# Testing the full flux calculation 
def __test_flux_reconstruction_on_cross_section(test_beam, N=None):

    """
    Compute flux of a Gaussian beam in free space through a plane
    orthogonal to the beam axis. The flux is contant and can be
    computed analytically. This procedure return the error of the
    numerical flux evaluation with and without interpolation.
    """

    # Generate data for the test
    sect, Ffield = __generate_data_for_test_on_cross_section(test_beam, N=N)
    
    # Call the main funtion computing the flux through the surface
    # twice: once with the field collocated on a grid and once passing
    # the analytical beam, thus skipping the interpolation step
    flux, data = compute_flux(Ffield, sect)
    analytic_flux = lambda xz: test_beam.flux(*xz)
    flux_no_interp, data = compute_flux(analytic_flux, sect, 
                                        interpolate_field=False)

    # Expected flux
    k0 = test_beam.k0
    w0 = test_beam.w0
    a0 = test_beam.a0 
    expected_flux = np.sqrt(0.5 * np.pi) * k0 * w0 * a0**2 
    difference = abs(flux - expected_flux)
    difference_no_interp = abs(flux_no_interp - expected_flux)

    print("\n N. of points in u = {}".format(sect.nptu))
    print("\n Expected flux = {}".format(expected_flux))
    print("\n Computed flux with interp. = {}".format(flux))
    print("\n Difference with interp. = {}".format(difference))
    print("\n Computed flux without interp. = {}".format(flux_no_interp))
    print("\n Difference without interp. = {} \n".format(difference_no_interp))

    return difference, difference_no_interp



# Testing the full flux calculation 
def __test_plotting(test_beam, flag, N=None):
    
    # Generate data for the test
    if flag == 'section':
        surf, Ffield = __generate_data_for_test_on_cross_section(test_beam, N=N)
    elif flag == 'circle':
        surf, Ffield = __generate_data_for_test_on_circle(test_beam, N=N)
    else:
        raise NotImplementedError("""flag is either 'section' or 'circle'.""")

    # Plotting 
    computed_flux, data = compute_flux(Ffield, surf)
    u, x1, x2, FnJ, Fn = data 
    plot_Fn(Ffield, x1, x2, Fn)
    
    return



def __generate_data_for_test_on_circle(test_beam, N=None):

    # Parameter of the sphere, centered in the origin
    R = 0.5             # radius
    center = (0.2, 0.0)

    # Box for field interpolation
    dx = dy = dz = 0.0051
    xmin = center[0] - 1.1 * R 
    zmin = center[1] - 1.1 * R 
    xmax = center[0] + 1.1 * R  
    zmax = center[1] + 1.1 * R 

    # Numper of mesh points on polar angle normalized in [0., 1.]
    if N == None:
        nptu = 151
    else:
        nptu = N

    # Define a line parametrically in 2d
    # (and provide the derivative of the mapping as well)
    pi = np.pi
    def phi(u):
        x = center[0] + R * np.cos(2.0*pi*u)
        z = center[1] + R * np.sin(2.0*pi*u)
        return (x, z)
    def phi_u(u): 
        e_ux = -2.0 * pi * R * np.sin(2.0*pi*u)
        e_uz = +2.0 * pi * R * np.cos(2.0*pi*u)
        return (e_ux, e_uz)
    def phi_u_perp(u):
        e_u_perp_x = +2.0 * pi * R * np.cos(2.0*pi*u) 
        e_u_perp_z = +2.0 * pi * R * np.sin(2.0*pi*u)
        return (e_u_perp_x, e_u_perp_z)
    
    discr_line = Line2D(phi=phi, phi_u = phi_u, phi_u_perp = phi_u_perp,
                        umin=0.0, umax=1.0, nptu=nptu)

    # Define the 2d spatial grid 
    # (It must be big enough to contain the surface)
    x, z = np.mgrid[slice(xmin, xmax+dx, dx), slice(zmin, zmax+dz, dz)]

    # Energy flux of the beam
    Fx, Fz = test_beam.flux(x, z)
    flux_vector_field = FluxVector2D(x1=x[:,0], x2=z[0,:], F1=Fx, F2=Fz)
    
    return discr_line, flux_vector_field


# Testing the full flux calculation 
def __test_flux_reconstruction_on_circle(test_beam, N=None):
    
    # Generate data for the test
    circle, Ffield = __generate_data_for_test_on_circle(test_beam, N=N)

    # Call the main funtion computing the flux through the surface
    # twice: once with the field collocated on a grid and once passing
    # the analytical beam, thus skipping the interpolation step
    # (test_beam is a global variable)
    flux, data = compute_flux(Ffield, circle)
    analytic_flux = lambda xz: test_beam.flux(*xz)
    flux_no_interp, data = compute_flux(analytic_flux, circle, 
                                        interpolate_field=False)

    # Expected flux
    expected_flux = 0.0
    difference = abs(flux - expected_flux)
    difference_no_interp = abs(flux_no_interp - expected_flux)

    print("\n N. of points in u = {}".format(circle.nptu))
    print("\n Expected flux = {}".format(expected_flux))
    print("\n Computed flux with interp. = {}".format(flux))
    print("\n Difference with interp. = {}".format(difference))
    print("\n Computed flux without interp. = {}".format(flux_no_interp))
    print("\n Difference without interp. = {} \n".format(difference_no_interp))

    return difference, difference_no_interp


# Just run __test_flux_reconstruction_in_3d with an increasing
# resolution and plot the error versus the expected order
def __test_scan_error(test_beam, test):
    
    resolutions = np.arange(4, 303, 10, dtype=float)
    errors = []
    errors_no_interp = []
    for N in resolutions:
        difference, difference_no_interp = test(test_beam, N=int(N))
        errors.append(difference)
        errors_no_interp.append(difference_no_interp)
        
    # Expected order
    regression = 0.9 * errors[0] * (resolutions[0] / resolutions)**4

    plt.figure()
    plt.loglog(resolutions, errors, label='error')
    plt.loglog(resolutions, errors_no_interp, label='error - no interp.')
    plt.semilogy(resolutions, regression, 'd', label='regression')
    plt.legend()
    plt.grid('on')
    
    plt.show()



# Running as a script will execute tests
if __name__ == '__main__':

    # Import WKBeam moduls
    try:
        import os, sys
        cwd = os.getcwd()
        sys.path.append(cwd) 
        import Tools.DevelopmentTools.ReferenceBeams.standard_Gaussian_beam as B
    except ImportError:
        msg= '\n For testing just run the module from the base WKBEAM directory.'
        raise ImportError(msg)

    # Generate a standard Gaussian beam for tests
    # ... Define the test beam ...
    k0 = 10.  # wave number in free space
    w0 = 0.2  # beam width at the waist
    a0 = 1.0  # beam amplitude at the waist
    # ... Contruct the Gaussian beam ...
    test_beam = B.GaussianBeam2D(k0, w0, a0)
    
    # Testing
    __test_flux_reconstruction_on_cross_section(test_beam)
    __test_flux_reconstruction_on_circle(test_beam)
    __test_plotting(test_beam, 'section', N=61)
    __test_plotting(test_beam, 'circle', N=101)
    __test_scan_error(test_beam, test=__test_flux_reconstruction_on_cross_section)
    __test_scan_error(test_beam, test=__test_flux_reconstruction_on_circle)

# end of file
