"""
Set of tool to compute and visualize the wave energy flux
through a generic surface given parametrically.

This is for the full three-dimensional calculation, and uses matplotlib as a 
graphics backend. See the module EnergyFlux3d_mayavi for the mayavi version
which has slightly different features
"""


# Import statements
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Local modules
import Tools.PlotData.PlotBinnedData.EnergyFlux3d_computations as EFcomp



# -------------------------------------------------------------------------------
# Main procedures


# This is the main procedure called by WKBeam
def flux_through_surface_in_3d(idata, hdf5data, surface_model, backend):

    """
    Takes data from the WKBeam run (input data 'idata', hdf5 dataset 'hdf5data',
    and the surface model 'surface') and construct an instance of the 
    named tuple FluxVector3D for the energy flux and Surface3D for the surface in 
    two dimentions. The named tuples are then passed to the relevant procedure
    to compute the energy flux through the surface and visualize the normal
    component of the energy flux vector on the surface.
    """
    
    # At last compute the flux and plot
    field, flux, data \
        = EFcomp.load_energy_flux_and_surface(hdf5data, surface_model)

    # Unpack data
    u, v, X, Y, Z, FnJ, Fn = data

    # Plotting
    Fn_msk = np.ma.masked_invalid(Fn)
    if backend == 'matplotlib2d': plot_with_matplotlib2d(u, v, Fn_msk)
    if backend == 'matplotlib3d': plot_with_matplotlib3d(X, Y, Z, Fn_msk)
    
    return flux, data, field

# Two-dimensional plotting in logical coordinates of the surface
def plot_with_matplotlib2d(u, v, Fn):
    
    """
    Plot the normal component of the energy flux on the plane
    of the parameters (u,v) of the surface.
    """

    U, V = np.meshgrid(u,v, indexing='ij')
    
    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(111, aspect='equal')
    m = ax.contourf(U, V, Fn, 100, cmap='coolwarm')
    ax.set_xlabel('$u$', fontsize=20)
    ax.set_ylabel('$v$', fontsize=20)
    fig.colorbar(m, ax=ax, shrink=0.5, aspect=5)
    plt.show()
    
    return None


# Three-dimensional visualization in physical coordinates
def plot_with_matplotlib3d(X, Y, Z, Fn):
    
    """
    Plot the surface in 3D using matplotlib and color-code it with
    the normal component of the energy flux.
    """
    m = plt.cm.ScalarMappable(cmap='coolwarm')
    m.set_array([])
    fcolors = m.to_rgba(Fn)

    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                    edgecolor='none', facecolors=fcolors,
                    antialiased=True, shade=False)
    ax.set_xlabel('$x$', fontsize=20)
    ax.set_ylabel('$y$', fontsize=20)
    ax.set_zlabel('$z$', fontsize=20)
    fig.colorbar(m, shrink=0.5, aspect=5)
    plt.show()
    
    return None


# -------------------------------------------------------------------------------
# Testing 


# Generate test data for flux through a cross-section of the beam
def __generate_data_for_test_on_cross_section(test_beam, N=None):

    # It is convenient for the test to define the box first
    # and then the surface which must be included in the box
    z0 = 0.5          # anchor of the surface
    Lx = 6.0          # length in x
    Ly = 6.0          # length in y
    dx = dy = 0.005
    dz = 0.0053       # avoid to have a point at z = 0

    if N == None:
        nptu = 141
        nptv = 141
    else:
        nptu = N
        nptv = N

    xmin = -0.5 * Lx
    xmax = +0.5 * Lx
    ymin = -0.5 * Ly
    ymax = +0.5 * Ly
    zmin = z0 - 0.02
    zmax = z0 + 0.02

    # Define a surface parametrically in 3d
    # (and provide the derivative of the mapping as well)
    x0 = np.array([xmin, ymin,  z0])
    ex = np.array([Lx,  0.0,  0.0])
    ey = np.array([ 0.0, Ly,  0.0])
    
    phi = lambda u, v: x0 + ex * u + ey * v
    phi_u = lambda u, v: ex
    phi_v = lambda u, v: ey
    
    discr_surf = EFcomp.Surface3D(phi=phi, phi_u=phi_u, phi_v=phi_v,
                                  umin=0.0, umax=1.0, nptu=nptu,
                                  vmin=0.0, vmax=1.0, nptv=nptv)

    # Define the 3d spatial grid 
    # (It must be big enough to contain the surface)
    x, y, z = np.mgrid[slice(xmin, xmax+dx, dx), 
                       slice(ymin, ymax+dy, dy),
                       slice(zmin, zmax+dz, dz)]

    # Energy flux of the beam
    F = test_beam.flux(x, y, z)
    flux_vector_field = EFcomp.FluxVector3D(x1=x[:,0,0], 
                                            x2=y[0,:,0], 
                                            x3=z[0,0,:], F=F)

    return discr_surf, flux_vector_field
    



# Testing the full flux calculation in 3D
def __test_flux_reconstruction_on_cross_section(test_beam, N=None):

    """
    Compute flux of a Gaussian beam in free space through a plane
    orthogonal to the beam axis. The flux is contant and can be
    computed analytically. This procedure return the error of the
    numerical flux evaluation with and without interpolation.
    """

    # Generate data for the test
    section, Ffield = __generate_data_for_test_on_cross_section(test_beam, N=N)
    
    # Call the main funtion computing the flux through the surface
    # twice: once with the field collocated on a grid and once passing
    # the analytical beam, thus skipping the interpolation step
    flux, r = EFcomp.compute_flux(Ffield, section)
    beam = lambda xyz: test_beam.flux(*xyz)
    flux_no_interp, r = EFcomp.compute_flux(beam, section, interpolate_field=False)

    # Expected flux
    k0 = test_beam.k0
    w0 = test_beam.w0
    a0 = test_beam.a0
    expected_flux = 0.5 * k0 * w0**2 * a0**2 * np.pi
    difference = abs(flux - expected_flux)
    difference_no_interp = abs(flux_no_interp - expected_flux)

    print("\n N. of points in u, v = {}, {}".format(section.nptu, section.nptv))
    print("\n Expected flux = {}".format(expected_flux))
    print("\n Computed flux with interp. = {}".format(flux))
    print("\n Difference with interp. = {}".format(difference))
    print("\n Computed flux without interp. = {}".format(flux_no_interp))
    print("\n Difference without interp. = {} \n".format(difference_no_interp))

    return difference, difference_no_interp


# Generate data for the flux on a sphere
def __generate_data_for_test_on_sphere(test_beam, N=None):

    # Parameter of the sphere, centered in the origin
    R = 0.5             # radius
    center = (0.0, 0.2, 0.0)

    # Box for field interpolation
    dx = dy = dz = 0.051
    xmin = center[0] - 1.1 * R 
    ymin = center[1] - 1.1 * R  
    zmin = center[2] - 1.1 * R 
    xmax = center[0] + 1.1 * R  
    ymax = center[1] + 1.1 * R  
    zmax = center[2] + 1.1 * R 

    # Numper of mesh points on polar angles normalized in [0., 1.]
    if N == None:
        nptu = 151
        nptv = 151
    else:
        nptu = N
        nptv = N

    # Define a surface parametrically in 3d
    # (and provide the derivative of the mapping as well)
    pi = np.pi
    def phi(u, v):
        x = center[0] + R * np.sin(pi*u) * np.cos(2.0*pi*v)
        y = center[1] + R * np.sin(pi*u) * np.sin(2.0*pi*v)
        z = center[2] + R * np.cos(pi*u)
        return (x, y, z)
    def phi_u(u, v): 
        e_ux = R * pi * np.cos(pi*u) * np.cos(2.0*pi*v)
        e_uy = R * pi * np.cos(pi*u) * np.sin(2.0*pi*v)
        e_uz = -R * pi * np.sin(pi*u)
        return (e_ux, e_uy, e_uz)
    def phi_v(u, v):
        e_vx = -R * 2.0 * pi * np.sin(pi*u) * np.sin(2.0*pi*v)
        e_vy = +R * 2.0 * pi * np.sin(pi*u) * np.cos(2.0*pi*v)
        e_vz = 0.0
        return (e_vx, e_vy, e_vz)
    
    discr_surf = EFcomp.Surface3D(phi=phi, phi_u=phi_u, phi_v=phi_v,
                                  umin=1.0e-8, umax=1.0, nptu=nptu,
                                  vmin=0.0, vmax=1.0, nptv=nptv)

    # Define the 3d spatial grid 
    # (It must be big enough to contain the surface)
    x, y, z = np.mgrid[slice(xmin, xmax+dx, dx), 
                       slice(ymin, ymax+dy, dy),
                       slice(zmin, zmax+dz, dz)]

    # Energy flux of the beam
    F = test_beam.flux(x, y, z)
    flux_vector_field = EFcomp.FluxVector3D(x1=x[:,0,0], 
                                            x2=y[0,:,0], x3=z[0,0,:], F=F)
    
    return discr_surf, flux_vector_field


# Testing the full flux calculation in 3D
def __test_flux_reconstruction_on_sphere(test_beam, N=None):
    
    # Generate data for the test
    sphere, Ffield = __generate_data_for_test_on_sphere(test_beam, N=N)

    # Call the main funtion computing the flux through the surface
    # twice: once with the field collocated on a grid and once passing
    # the analytical beam, thus skipping the interpolation step
    flux, r = EFcomp.compute_flux(Ffield, sphere)

    beam = lambda xyz: test_beam.flux(*xyz)
    flux_no_interp, r = EFcomp.compute_flux(beam, sphere, interpolate_field=False)

    # Expected flux
    expected_flux = 0.0
    difference = abs(flux - expected_flux)
    difference_no_interp = abs(flux_no_interp - expected_flux)

    print("\n N. of points in u, v = {}, {}".format(sphere.nptu, sphere.nptv))
    print("\n Expected flux = {}".format(expected_flux))
    print("\n Computed flux with interp. = {}".format(flux))
    print("\n Difference with interp. = {}".format(difference))
    print("\n Computed flux without interp. = {}".format(flux_no_interp))
    print("\n Difference without interp. = {} \n".format(difference_no_interp))

    return difference, difference_no_interp



# Testing the full flux calculation in 3D
def __test_plotting(test_beam, flag, N=None):
    
    # Generate data for the test
    if flag == 'section':
        surf, Ffield = __generate_data_for_test_on_cross_section(test_beam, N=N)
    elif flag == 'sphere':
        surf, Ffield = __generate_data_for_test_on_sphere(test_beam, N=N)
    else:
        raise NotImplementedError("""flag is either 'section' or 'sphere'.""")

    # Compote flux vector on the surface
    data = EFcomp.build_FnJ(Ffield, surf)

    # Unpack and plot
    u, v, X, Y, Z, FnJ, Fn = data
    plot_with_matplotlib3d(X, Y, Z, Fn)
    
    return None


# Just run __test_flux_reconstruction_in_3d with an increasing
# resolution and plot the error versus the expected order
def __test_scan_error(test_beam, test):
    
    resolutions = np.arange(11, 303, 10, dtype=float)
    errors = []
    errors_no_interp = []
    for N in resolutions:
        difference, difference_no_interp = test(test_beam, N=int(N))
        errors.append(difference)
        errors_no_interp.append(difference_no_interp)
        
    # Expected order
    regression = 0.9 * errors[0] * (resolutions[0] / resolutions)**4

    plt.figure()
    plt.loglog(resolutions, errors)
    plt.loglog(resolutions, errors_no_interp)
    plt.semilogy(resolutions, regression, 'd')
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
    test_beam = B.GaussianBeam3D(k0, w0, a0)
    
    # Testing
    __test_flux_reconstruction_on_cross_section(test_beam)
    __test_flux_reconstruction_on_sphere(test_beam)
    __test_plotting(test_beam, 'section', N=61)
    __test_plotting(test_beam, 'sphere', N=41)
    __test_scan_error(test_beam, test=__test_flux_reconstruction_on_cross_section)
    __test_scan_error(test_beam, test=__test_flux_reconstruction_on_sphere)

# end of file
