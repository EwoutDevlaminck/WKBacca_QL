"""Main driver of the WKBeam code.

Run $ python WKBeam.py for help.

Examples:

  $ mpiexec -np <n> python WKBeam trace <ray_tracing_configuration_file>
  $ python WKBeam.py bin <binning_configuration_file>
  $ python WKBeam.py plotbin <binned_hdf5_file>
  $ python WKBeam.py plot2d <binning configuration_file>
  $ python WKBeam.py plotabs <binning_configuration_file> 
  $ python WKBeam.py ploteq <ray_tracing_configuration_file> 
  $ python WKBeam.py plotfluct <ray_tracing_configuration_file> 
  $ python WKBeam.py flux <binning configuration_file>

Note that the first argument is a control flag and the second is ether
and input file or a txt configuration file, cf. examples.

The ray tracing procedure must be called through mpiexec. Interactive use
is recommended for testing purposes only with a very small number of rays.
"""

# Load standard modules
import sys

# Define the dictionary of operation modes
# (If you update this dictionary, DO NOT forget to update the 
#  help message below and the module doc string.)
WKBeam_modes = {
    'trace': {
        'procedure': 'call_ray_tracer',
        'module': 'RayTracing.tracerays',
    },
    'bin': {
        'procedure': 'call_binning',
        'module': 'Binning.binrays',
    },
    'plotbin': {
        'procedure': 'plot_binned',
        'module': 'Tools.PlotData.PlotBinnedData.plotbinneddata', 
    },
    'plot2d': {
        'procedure': 'plot2d',
        'module': 'Tools.PlotData.PlotBinnedData.plot2d',
    },
    'plotabs': {
        'procedure': 'plot_abs',
        'module': 'Tools.PlotData.PlotAbsorptionProfile.plotabsprofile',
    },
    'ploteq': {
        'procedure': 'plot_eq',
        'module': 'Tools.PlotData.PlotEquilibrium.plotequilibrium',
    },
    'plotfluct': {
        'procedure': 'plot_fluct',
        'module': 'Tools.PlotData.PlotFluctuations.plotfluctuations',
    },
    'flux': {
        'procedure': 'flux_through_surface',
        'module': 'Tools.PlotData.PlotBinnedData.EnergyFlux',
    },
    'beam3d': {
        'procedure': 'plot_beam_with_mayavi',
        'module': 'Tools.PlotData.PlotBinnedData.Beam3D',
    },
    'beamFluct': {
    	'procedure': 'plot_beam_fluct',
    	'module': 'Tools.PlotData.PlotFluctuations.plotBeamFluctuations',
    },
}

# Help messange (list operation modes, etc...)
msg = """ 
 USAGE: 

   $ python WKBEam.py <mode_flag> <input_or_configuration_file>

 If mode_flag = trace, launch with mpiexec:

   $ mpiexec -np <n> python WKBEam.py trace <configuration_file>

 where n is the number of processors. 
    
 LIST OF VALID MODE FLAGS:
  1. trace     - requires <ray_tracing_configuration_file> and mpiexec
  2. bin       - requires <binning_configuration_file>
  3. plotbin   - requires <binned_hdf5_file>
  4. plot2d    - requires <binning_configuration_file>
  5. plotabs   - requires <list_of_binning_configuration_files> 
  6. ploteq    - requires <ray_tracing_configuration_file>  
  7. plotfluct - requires <ray_tracing_configuration_file>  
  8. flux      - requires <binning_configuration_file>
  9. beam3d    - requires <binning_configuration_file>
  10.beamFluct - requires <binned_hdf5_file> and <ray_tracing_configuration_file>
"""

# Check the console input
if len(sys.argv) < 3:
    print(msg)
    sys.exit()

# Mode flag
flag = sys.argv[1]

# Check the mode flag and execute accordingly
if flag not in WKBeam_modes.keys():
    print(msg)
    raise ValueError("mode_flag not understood.")
else:
    # Define the input
    if flag == 'plotabs':
        inputfile = sys.argv[2:]
    elif flag == 'plotbin' or flag == 'beamFluct':
    #Added by Ewout. To also plot the computed equilibrium flux surfaces on top of
    # the beam propagation
    	inputfile = sys.argv[2:]
    else:
        inputfile = sys.argv[2]
        
    # Extract the name of the relevant procedure and module
    procedurename = WKBeam_modes[flag]['procedure']
    modulename = WKBeam_modes[flag]['module']

    # Import the relevant module
    exec('from '+modulename+' import '+procedurename)

    # Launch the appropriate application
    exec(procedurename+'(inputfile)')
#
# END OF FILE
