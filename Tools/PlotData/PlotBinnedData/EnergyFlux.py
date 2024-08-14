"""
Compute the wave energy flux through a surface for either a fully
three-dimensional beam or a two-dimensional test-beam.

It is physically not correct to use the two-dimensional calculation
on a realistic three dimensional beam projected onto some plane. This would
not give the correct flux. The two-dimensional version of this calculation 
is meant for two-dimensional tests only (flag twodim = True in raytracing file.) 

The calculation can either be run from WKBeam directly

 $ python WkBeam.py flux <binning_file>

where the binning file must correspond to a binning of the wave energy
flux, including the group velocity in the binning. This has to include
a variable surface_model_path with the path to the directory with the python
module with the definition of the parametric surface, as well as the variable 
surface_model with the name of that module without the extension .py. 
Examples of such modules can be found in the standard cases, e.g., 
in StandardCases/Focus/output/. 
"""


# Import statements
import sys
import h5py
import importlib
import numpy as np
import matplotlib.pyplot as plt
from CommonModules.input_data import InputData
from Tools.PlotData.PlotBinnedData.grids import get_contents, get_grid


# Main function called by WKBeam.py
def flux_through_surface(filename):
    
    """
    Compute and visualize the normal component of the energy flux vector on a 
    line (in two dimentions) or a surface (in three dimensions).
    The total power through the line or surface, i.e., the integral over the 
    line or surface of the normal energy flux is computed.
    """
    
    # Initial message
    print("\n Computing the wave energy flux through the given surface ...\n")

    # load and check the hdf5 dataset
    idata = InputData(filename)
    hdf5data, wtr, EnergyFluxAvailable, EnergyFluxValid = get_binned_data(idata)
    
    # Check the components of the velocity field
    if not EnergyFluxAvailable:
        msg = """Dataset does not contain the group-velocity field."""
        raise RuntimeError(msg)

    if not EnergyFluxValid:
        Vfield = hdf5data.get('VelocityFieldStored').asstr()[...]
        msg = """The dataset contains the velocity field {}.
        Computing energy flux requires the group velocity V.
        Bin the three relevant component of V instead.""".format(Vfield) 
        raise RuntimeError(msg)

    # Select plotting backend
    if hasattr(idata, 'plotting_backend'):
        backend = idata.plotting_backend
        assert backend in ['mayavi3d', 'matplotlib3d', 'matplotlib2d',
                           'data_only']
    else:
        backend = 'matplotlib3d'
        print("\nWARNIG: backend not found or invalid. Using 'matplotlib3d'.\n")

    # Load the parametric surface and create a surface object
    try:
        sys.path.append(idata.surface_model_path)
        surf = importlib.import_module(idata.surface_model)
    except AttributeError:
        m = """
        The binning configuration file does not have a specification
        of the surface model. Please add the variables surface_model_path and
        surface_model pointing to the path and the module where the surface
        is specified."""
        raise RuntimeError(m)

    # Call the main function
    if len(wtr) == 2:

        if backend != 'matplotlib2d':
            backend = 'matplotlib2d'
            print("\nWARNIG: beckend ignored. Using 'matplotlib2d'.\n")
        
        import Tools.PlotData.PlotBinnedData.EnergyFlux2d as ef
        data2d = ef.flux_through_line_in_2d(idata, hdf5data, surf)

    elif len(wtr) == 3:

        if backend == 'mayavi3d':
            
            # The import of plotting library has to go here in order to 
            # avoid conflicts between mayavi and matplotlib
            import Tools.PlotData.PlotBinnedData.EnergyFlux3d_mayavi as ef
            data3d_mayavi = ef.flux_and_beam_in_3d(idata, hdf5data, surf)

        elif backend == 'data_only':

            # The import of plotting library has to go here in order to 
            # avoid conflicts between mayavi and matplotlib
            import Tools.PlotData.PlotBinnedData.EnergyFlux3d_mayavi as ef
            data3d_mayavi = ef.Data_for_flux_and_beam_in_3d(idata, hdf5data, surf)

        else:
            
            # The import of plotting library has to go here in order to 
            # avoid conflicts between mayavi and matplotlib
            import Tools.PlotData.PlotBinnedData.EnergyFlux3d_mpl as ef
            data3d_mpl = ef.flux_through_surface_in_3d(idata, hdf5data, surf, backend)

    else:

        msg = """
        The binning input file should correspond to bin in either
        two or three spatial dimensions.
        """
        raise RuntimeError(msg)


    # Write data is requested. This has to mimic the if structure above to
    # aacount for the different data available in the various cases
    if hasattr(idata, 'write_energy_flux_data_to'):
        filename = idata.outputdirectory + idata.write_energy_flux_data_to
        fid = h5py.File(filename, 'w')
        if len(wtr) == 2:

            computed_flux, flux_data, flux_field = data2d
            fid.create_dataset('power flux', data=computed_flux)
            fid.create_dataset('surface x1', data=flux_data[1])
            fid.create_dataset('surface x2', data=flux_data[2])
            fid.create_dataset('surface FnJ', data=flux_data[3])
            fid.create_dataset('surface Fn', data=flux_data[4])
            fid.create_dataset('flux field x1', data=flux_field.x1)
            fid.create_dataset('flux field x2', data=flux_field.x2)
            fid.create_dataset('flux field F1', data=flux_field.F1)
            fid.create_dataset('flux field F2', data=flux_field.F2)

        elif len(wtr) == 3:
            
            if backend in ['mayavi3d', 'data_only']:

                computed_flux, flux_data, flux_field, antenna = data3d_mayavi
                fid.create_dataset('power flux', data=computed_flux)
                fid.create_dataset('surface X', data=flux_data[2])
                fid.create_dataset('surface Y', data=flux_data[3])
                fid.create_dataset('surface Z', data=flux_data[4])
                fid.create_dataset('surface FnJ', data=flux_data[5])
                fid.create_dataset('surface Fn', data=flux_data[6])
                fid.create_dataset('field X', data=flux_field[0])
                fid.create_dataset('field Y', data=flux_field[1])
                fid.create_dataset('field Z', data=flux_field[2])
                fid.create_dataset('flux field Fx', data=flux_field[3])
                fid.create_dataset('flux field Fy', data=flux_field[4])
                fid.create_dataset('flux field Fz', data=flux_field[5])
                fid.create_dataset('antenna X', data=antenna[0])
                fid.create_dataset('antenna Y', data=antenna[1])
                fid.create_dataset('antenna Z', data=antenna[2])

            else:
                
                computed_flux, flux_data, flux_field = data3d_mpl
                fid.create_dataset('power flux', data=computed_flux)
                fid.create_dataset('surface u', data=flux_data[0])
                fid.create_dataset('surface v', data=flux_data[1])
                fid.create_dataset('surface Fn', data=flux_data[6])

        else:
            pass
            
        fid.close()
        print('\n Flux data written in {}\n'.format(filename))
        
    return None
    

# Function which opens the hdf5 files and check that the energy flux vector
# has been properly computed and stored in the data set
def get_binned_data(idata):

    """
    Load the binned traces and check for the energy flux.
    """

    # Load the input data in the usual way with the idata object
    datadir = idata.outputdirectory

    # Open hdf5 binned file
    if hasattr(idata, 'outputfilename'):
        binnedfile = datadir + idata.outputfilename[0] + '.hdf5'
    else:
        binnedfile = datadir + idata.inputfilename[0] + '_binned.hdf5'

    print('Binned file: {}'.format(binnedfile))
    hdf5data = h5py.File(binnedfile, 'r')  

    # Distinguish between two- and three-dimensional data
    # In the dataset, WhatToResolve is a strign of the form 'X,Y,".
    # split(',') return an list splitting the string at the commas.
    # Therefore the last element of the list is empty and it is removed.
    wtr = get_contents(hdf5data)
    Ndim = len(wtr) 
    Coordinates =[]
    try:
        for coordinate in wtr:
            Coordinates.append(coordinate)
            assert coordinate in ['R', 'X', 'Y', 'Z']
    except AssertionError:
        msg = """Coordinates of the domain are {},
        when only physical domain coordinates should be used, namely,
        'R', 'X', 'Y', and 'Z'.""".format(Coordinates)
        raise RuntimeError(msg)

    # The velocity field must really refer to the group velocity
    # (WKBeam treat in the same way the refractive index)
    if 'VelocityField' in hdf5data.keys():
        EnergyFluxAvailable = True
        # For backward compatibility add a try loop
        try:
            Vfield = hdf5data.get('VelocityFieldStored').asstr()[()].split(',')
        except AttributeError:
            Vfield = hdf5data.get('VelocityFieldStored')[()].split(',')
    else:
        EnergyFluxAvailable = False
        Vfield = []
        
    try:
        for Vi in Vfield[0:3]: assert Vi in ['Vx', 'Vy', 'Vz']
        EnergyFluxValid = True
    except:
        EnergyFluxValid = False
        
    return hdf5data, wtr, EnergyFluxAvailable, EnergyFluxValid

# end of file
