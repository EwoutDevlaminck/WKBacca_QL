"""This module collects plotting tools for binned data in 2D, namely,
either the evolution of profiles in X or R coordinates, or projection
of a quantity on the poloidal plane of the device.

The control parameters for the plotting are appended to the binning 
configuration file which is used to create the data set. Therefore,
the user can fisrt call the binning procedure with a given configuration
file, and then call the plotting utilities with the same configuration
file to inspect the results of the binning.
"""

# Load standard modules
import h5py
import matplotlib.pyplot as plt
import numpy as np
import collections as coll

# Load local modules
from CommonModules.input_data import InputData
from CommonModules.PlasmaEquilibrium import IntSample, StixParamSample
from CommonModules.PlasmaEquilibrium import TokamakEquilibrium
from CommonModules.PlasmaEquilibrium import AxisymmetricEquilibrium
from Tools.PlotData.PlotBinnedData.grids import get_contents, get_grid
from Tools.PlotData.CommonPlotting import plotting_functions


###############################################################################
# PRIVATE FUNCTIONS
###############################################################################


# A useful named tuple to pass flags
PlotFlags = coll.namedtuple('PlotFlags', ['plotdensity', 'plotmagneticsurfaces'])


# Given the point xpoint, find the index of the nearest point in xgrid
def _nearest_index(xgrid, xpoint):
    return np.abs(xgrid - xpoint).argmin()


# Final rendering function
def _plot_field(xgrid, ygrid, field, axes, rendering_flag, ncontours=None):

    if rendering_flag == 'contourf':
        assert type(ncontours) is int
        cplot = axes.contourf(xgrid, ygrid, field, ncontours, cmap='coolwarm')
    elif rendering_flag == 'contour':
        assert type(ncontours) is int  
        ncontours //= 10
        cplot = axes.contour(xgrid, ygrid, field, ncontours)
    elif rendering_flag == 'pcolormesh':
        cplot = axes.pcolormesh(xgrid, ygrid, field, shading='auto', cmap='coolwarm')
    
    return cplot


# Plotting generic two-dimensional binned data
def _plot2d_generic(idata, xgrid, ygrid, FreqGHz, beam, error, 
                    label_x, label_y, rendering_flag, plot_flags):

    # Setup the figure
    fig, axs = plt.subplots(2, 1, sharex=True)

    # Plot the full beam and its error
    fplot = _plot_field(xgrid, ygrid, beam.T, axs[0], rendering_flag, ncontours=100)
    fig.colorbar(fplot, ax=axs[0], format="%1.2f")
    axs[0].set_title('electric field energy density')
    axs[0].set_xlim(xgrid[0], xgrid[-1])
    axs[0].set_ylabel(label_y, fontsize=15)
    axs[0].set_ylim(ygrid[0], ygrid[-1])
    # ... estimated variance ...
    eplot = _plot_field(xgrid, ygrid, error.T, axs[1], rendering_flag, ncontours=100)
    fig.colorbar(eplot, ax=axs[1], format="%1.2f")
    axs[1].set_title('standard deviation')
    axs[1].set_xlabel(label_x, fontsize=15)
    axs[1].set_xlim(xgrid[0], xgrid[-1])
    axs[1].set_ylabel(label_y, fontsize=15)
    axs[1].set_ylim(ygrid[0], ygrid[-1])

    # Plot a few sections at a given position, cutting vertically
    # (This runs only if the configuration file has the appropriate
    #  data for this operation, i.e., the list idata.vsections of
    #  vertical sections).
    try:
        vsections = idata.vsections
        fig_vsections = plt.figure(2, figsize=(10,8))
        ax_vsections = fig_vsections.add_subplot(111)
        ax_vsections.set_xlabel(label_y, fontsize=15)
        ax_vsections.set_title('vertical sections')
        for section in vsections:
            index = _nearest_index(xgrid, section)
            ax_vsections.errorbar(ygrid, beam[index,:], yerr=error[index,:])
        fig_vsections.legend(vsections)
    except:
        pass

    # Plot a few sections at a given position, cutting horizontally
    # (Analogous to vertical section above.)
    try:
        hsections = idata.hsections
        fig_hsections = plt.figure(3, figsize=(10,8))
        ax_hsections = fig_hsections.add_subplot(111)
        ax_hsections.set_xlabel(label_x, fontsize=15)
        ax_hsections.set_title('horizontal sections')
        for section in hsections:
            index = _nearest_index(ygrid, section)
            ax_hsections.errorbar(xgrid, beam[:,index], yerr=error[:,index])
        fig_hsections.legend(hsections)
    except:
        pass

    plt.show()

    # return
    pass


# Plotting the poloidal projection with density contours
def _plot2d_tokamak(idata, xgrid, ygrid, FreqGHz, beam, error, 
                    label_x, label_y, rendering_flag, plot_flags):
    
    # Load the plasma equilibrium for a tokamak
    Eq = TokamakEquilibrium(idata)

    # Extract the equilibrium grid
    Req = Eq.Rgrid
    zeq = Eq.zgrid
    psi = Eq.psigrid
    psi_at_separatrix = 1. # psigrid is always normalized

    # Sample plasma parameters on the grid
    nptR, nptz = Req.shape
    R1d = np.linspace(Req[0,0], Req[-1,0], 2*nptR)
    z1d =  np.linspace(zeq[0,0], zeq[0,-1], 2*nptz)
    StixX, StixY, field_and_density = StixParamSample(R1d, z1d, Eq, FreqGHz)
    Ne = field_and_density[-1]

    # Mask the beam where the amplitude is negligible
    maskedbeam = np.ma.masked_less(beam, idata.mask_threshold).T

    # Plotting directives
    fig = plt.figure(1, figsize=(10,10))
    ax = fig.add_subplot(111, aspect='equal')
    ax.set_xlabel('$R$ [cm]', fontsize=15) 
    ax.set_ylabel('$Z$ [cm]', fontsize=15)

    # ... density contours ...
    if plot_flags.plotdensity:
        density_plot = ax.contourf(R1d, z1d, Ne, 200, cmap='copper')
        cDensity = fig.colorbar(density_plot, orientation='vertical')
        cDensity.set_label(r'$n_\mathrm{e}$ [$10^{13} \mathrm{cm}^{-3}$]')

        # ... flux surfaces ...
    if plot_flags.plotmagneticsurfaces:
        if plot_flags.plotdensity:
            clrs_surf = 'w'
            clr_sep = 'r'
        else:
            clrs_surf = 'k'
            clr_sep = 'r'

        ax.contour(Req, zeq, psi, 20, colors=clrs_surf)
        ax.contour(Req, zeq, psi, [psi_at_separatrix], colors=clr_sep)

    # ... plot resonances
    h1, h2, h3 = plotting_functions.add_cyclotron_resonances(R1d, z1d, StixY, ax)
    # try:
    #     h1.collections[0].set_label('first harm.')
    #     h2.collections[0].set_label('second harm.')
    #     h3.collections[0].set_label('third harm.')
    # except IndexError:
    #     pass

    # ... electric field energy density ...
    cplot = _plot_field(xgrid, ygrid, maskedbeam, ax, rendering_flag, 200)
    cField = fig.colorbar(cplot, orientation='vertical')
    cField.set_label(r'field')
    # ... if requested add the corresponding TORBEAM result for the
    # central and peripheral rays ...
    try:
        if idata.compare_to_TORBEAM == 'yes':
            # ... load TORBEAM data ...
            t1 = np.loadtxt(idata.torbeam_dir+'t1_LIB.dat')
            # ... plot TORBEAM central and peripheral rays ...
            ax.plot(t1[:,0], t1[:,1], 'white')
            ax.plot(t1[:,2], t1[:,3], 'white')
            ax.plot(t1[:,4], t1[:,5], 'white')
    except AttributeError:
        pass

    # ... add the legend for resonances...
    h1_handles, h1_labels = h1.legend_elements()
    h2_handles, h2_labels = h2.legend_elements()
    h3_handles, h3_labels = h3.legend_elements()

    handles = h1_handles + h2_handles + h3_handles
    labels = ['first harm.', 'second harm.', 'third harm.']
    
    legend = ax.legend(handles, labels,
                       loc=3, bbox_to_anchor=(0.01, 0.9, 0.98, 0.5),
                       ncol=1, mode='expand', borderaxespad=0.0,
                       fancybox=True, framealpha=1.0)
        
    plt.show()

    # return
    pass


# Plotting the poloidal projection with flux surfaces
def _plot2d_axisymmetric(idata, xgrid, ygrid, FreqGHz, beam, error, 
                         datalabel_x, datalabel_y, rendering_flag, plot_flags):

    # Load the plasma equilibrium for a tokamak
    Eq = AxisymmetricEquilibrium(idata)

    # Grid on the poloidal plane
    R1d = np.linspace(Eq.rmaj-Eq.rmin, Eq.rmaj+Eq.rmin, 201)
    z1d = np.linspace(-Eq.rmin, Eq.rmin, 200)

    # Sample the electron density on the same grid
    Ne = IntSample(R1d, z1d, Eq.NeInt.eval)

    # Plotting directives
    fig = plt.figure(1, figsize=(12,10))
    ax = fig.add_subplot(111, aspect='equal')
    # ... electric field energy density ...
    beam_plot = _plot_field(xgrid, ygrid, beam.T, ax, rendering_flag, ncontours=200)
    cField = fig.colorbar(beam_plot, orientation='vertical')
    # ... density contours ...
    density_plot = ax.contour(R1d, z1d, Ne, 20, cmap='copper')
    cDensity = fig.colorbar(density_plot, orientation='vertical')
    cDensity.set_label(r'$n_\mathrm{e}$ [$10^{13} \mathrm{cm}^{-3}$]')
    cField.set_label(r'field')
    plt.xlabel('$R$ [cm]') 
    plt.ylabel('$Z$ [cm]')

    plt.show()

    # return
    pass



###############################################################################
# DRIVER
###############################################################################


# List of operation modes
modes = {
    'evolution of partial densities' : _plot2d_generic, 
    'poloidal section - tokamaks' : _plot2d_tokamak,
    'poloidal section - TORPEX-like' : _plot2d_axisymmetric,
}
quantities = ['Y', 'Z', 'Nx', 'Ny', 'Nz', 'Nparallel', 'phiN']


# Main function
def plot2d(filename):

    """Plot binned data in the two-dimensional plane.
    Plotting options are controlled by a configuration file,
    cf. Standard cases for examples.
    
    The name of an hdf5 file created by "WKBeam.py bin" can also be
    passed. In this case, it is assumed that
           'plotmode = 'evolution of a partial densities'.
    """

    # Initial message
    print("\n Plotting a two-dimensional projection of the field ... \n")

    # Check the default mode
    ext = filename[-4:]
    if ext == 'hdf5':
        
        # ... if the input file name points to an hdf5 dataset
        # used the default mode ...
        idata = None
        plotmode = 'evolution of partial densities'
        hdf5data = h5py.File(filename, 'r')

    else:

        # ... otherwise assume it's a configuration file
        # and load it (account for the fact that outputfilename
        # is an optional input flag and might not be present)...
        idata = InputData(filename)
        plotmode = idata.plotmode
        datadir = idata.outputdirectory
        if hasattr(idata, 'outputfilename'):
            binnedfile = datadir + idata.outputfilename[0] + '.hdf5'
        else:
            binnedfile = datadir + idata.inputfilename[0] + '_binned.hdf5'
        print('Binned file: {} \n'.format(binnedfile))
        hdf5data = h5py.File(binnedfile, 'r')

    # Check the data set
    wtr = get_contents(hdf5data)
    condition = (len(wtr) != 2 or 
                 wtr[0] not in ['X', 'Z', 'R'] or 
                 wtr[1] not in quantities)
    if condition:
        msg = """INPUT ERROR: The data set does not seem to be appropriate.
        It should contain the electric field amplitude as a function of 
        either (X, Q), (Z, Q) or (R, Q), where 
             Q in ['Y', 'Z', 'Nx', 'Ny', 'Nz', 'Nparallel', 'phiN']. 
        Use the flag WhatToResolve = ['X', Q] in the binning 
        configuration file.
        """
        raise ValueError(msg)

    # Get grid parameters for the x and y axis of the plot
    # (If 'Z' is given put it on the vertical axis for a more
    #  intuitive visualization.)
    if wtr[0] == 'Z':
        datalabel_x = wtr[1]
        datalabel_y = wtr[0]
        xgrid, dx = get_grid(hdf5data, datalabel_x)    
        ygrid, dy = get_grid(hdf5data, datalabel_y)  
    else:
        datalabel_x = wtr[0]
        datalabel_y = wtr[1]
        xgrid, dx = get_grid(hdf5data, datalabel_x)    
        ygrid, dy = get_grid(hdf5data, datalabel_y)

    # Load the beam with its error
    FreqGHz = hdf5data.get('FreqGHz')[()]
    beam = hdf5data.get('BinnedTraces')[:,:,0]
    error = hdf5data.get('BinnedTraces')[:,:,1]
    # ... compute the cell average from the cumulative sum
    beam = beam / dx / dy
    error = error / dx / dy
    if wtr[0] == 'Z':
        # ... trasposition is needed because of the axis reversal ...
        beam = beam.T
        error = error.T

    # Close the data set
    hdf5data.close()

    if len(xgrid) == 1:
        # If one of the dimensions has only one grid value, we want to plot a 1D profile
        # of the beam.
        plt.figure()
        plt.plot(ygrid, beam[0,:])
        plt.title(r'$|E|^2$ at ${} = {}$'.format(datalabel_x, xgrid[0]))
        plt.xlabel(datalabel_y)
        plt.ylabel(r'$|E|^2\ (A.U.)$')
        plt.show()
        return
    elif len(ygrid) == 1:
        # If one of the dimensions has only one grid value, we want to plot a 1D profile
        # of the beam.
        plt.figure()
        plt.plot(xgrid, beam[:, 0])
        plt.title(r'$|E|^2$ at ${} = {}$'.format(datalabel_y, ygrid[0]))
        plt.xlabel(datalabel_x)
        plt.ylabel(r'$|E|^2\ (A.U.)$')
        plt.show()
        return
    

    # Rendering function
    if hasattr(idata, 'rendering'):
        rendering_flag = idata.rendering
    else:
        rendering_flag = 'contourf'

    # Plot content flags
    if hasattr(idata, 'plotdensity'):
        plotdensity = idata.plotdensity
    else:
        plotdensity = True

    if hasattr(idata, 'plotmagneticsurfaces'):
        plotmagneticsurfaces = idata.plotmagneticsurfaces
    else:
        plotmagneticsurfaces = True
        
    plot_flags = PlotFlags(plotdensity, plotmagneticsurfaces)

    # Check the mode flag and call the appropriate function
    if plotmode in modes.keys():
        function = modes[plotmode]
        function(idata, xgrid, ygrid, FreqGHz, beam, error, 
                 datalabel_x, datalabel_y, rendering_flag, plot_flags)
    else:
        raise ValueError('Flag plotmode not understood.')

    # return
    pass
#
# END OF FILE
