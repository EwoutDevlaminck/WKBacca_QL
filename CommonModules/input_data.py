"""Definition of the class InputData.

The class input data reads a set of control parameters from a
.txt file (specified by console input).
"""


# Import Statements
import numpy as np
import math
import sys
import importlib



#
# --- classes ---
#

# Class InputData: read and prepare input data for further processing
class InputData:


    # Initialization
    def __init__(self, input_file_name):

        """Initialization procedure. This starts up a class object,
        depending on the data provided by the input list argv and
        reads the input data.
        """

        # Load the control parameters of the interface
        f = open(input_file_name, 'r')
        input_data = f.readlines()
        for line in input_data:
            # ... skip commented lines ...
            l = line.strip()
            if (l == '') or (l[0] == '#'):
                continue
            # ... non-commented line may be blank ...
            else:
                # ... use the try construct to skip blank lines ...
                try:
                    exec('self.' + line)
                except:
                    print("WARNING input_data.py ... skipping line:")
                    print(line)
                    continue
        
        # Close file and return
        f.close()

        # Check the model for the fluctuation envelope:
        # when a string is passed instead of a function, assume the string
        # is the name of a plain python module which contains the relevant
        # model of fluctuations
        if hasattr(self, 'scatteringDeltaneOverne'):
            # Advanced input: scattering model defined in a python module
            # Only for the Gaussian model for the moment
            if type(self.scatteringDeltaneOverne) == str and \
               self.scatteringGaussian == True:
                rms_module = self.scatteringDeltaneOverne
                rms_model = self.__load_external_module__(rms_module)
                self.scatteringDeltaneOverne = rms_model.scatteringDeltaneOverne
            # Standard input
            elif hasattr(self.scatteringDeltaneOverne, '__call__') or \
                 type(self.scatteringDeltaneOverne) == float:
                pass
            else:
                msg = """
                Input variable scatteringDeltaneOverne should be 
                (1) either the name of a python module or a lambda function for
                    Gaussian fluctuations; 
                (2) a number for the fluctuation generated with the Shafer model.
                """
                raise RuntimeError(msg)
        else:
            pass

        # Check the perpendicular correlation length of fluctuations:
        # similarly to the fluctuation amplitude, the perpendicular correlation
        # length can either be loaded from an external function or fixed to a
        # constant number from input file.
        if hasattr(self, 'scatteringLengthPerp'):
            # Variable Lperp with model defined in a python module
            if type(self.scatteringLengthPerp) == str:
                Lperp_module = self.scatteringLengthPerp
                Lperp_model = self.__load_external_module__(Lperp_module)
                self.scatteringLengthPerp = Lperp_model.scatteringLengthPerp
            # Standard input
            elif type(self.scatteringLengthPerp) == float:
                assert self.scatteringLengthPerp > 0.0
                Lperp_value = self.scatteringLengthPerp
                self.scatteringLengthPerp = lambda rho, theta, Ne, Te, Bnorm: Lperp_value
            else:
                msg = """
                Input variable scatteringLengthPerp should be 
                (1) either the name of a python module located in the input directory
                (2) or a constant positive real number. 
                """
                raise RuntimeError(msg)
        else:
            pass

        # The same for the parallel correlation length of fluctuations
        if hasattr(self, 'scatteringLengthParallel'):
            # Variable Lperp with model defined in a python module
            if type(self.scatteringLengthParallel) == str:
                Lparallel_module = self.scatteringLengthParallel
                Lparallel_model = self.__load_external_module__(Lparallel_module)
                self.scatteringLengthParallel = Lparallel_model.scatteringLengthParallel
            # Standard input
            elif type(self.scatteringLengthParallel) == float:
                assert self.scatteringLengthParallel > 0.0
                Lparallel_value = self.scatteringLengthParallel
                self.scatteringLengthParallel = lambda rho, theta, Ne, Te, Bnorm: Lparallel_value
            else:
                msg = """
                Input variable scatteringLengthParallel should be 
                (1) either the name of a python module located in the input directory
                (2) or a constant positive real number. 
                """
                raise RuntimeError(msg)
        else:
            pass        


        # Handle the default for optional flags
        if hasattr(self, 'freeze_random_numbers'):
            assert type(self.freeze_random_numbers) == bool
        else:
            self.freeze_random_numbers = False

        # Handle the optional parameters for the extension of the grid
        if hasattr(self, 'extend_grid_by'):
            assert type(self.extend_grid_by) == tuple
            assert len(self.extend_grid_by) == 4
            for el in self.extend_grid_by:
                assert type(el) == int
        else:
            self.extend_grid_by = None

        
        # Return
        return


    # Metho used to load external profiles
    # Used for the scattering amplitude and the scattering Lperp
    def __load_external_module__(self, ext_module):

        """
        Load an external module. It is assumed that the target module ext_module
        is located in the equilibrium directory.
        """
        
        sys.path.append(self.equilibriumdirectory)
        return importlib.import_module(ext_module)
        

#
# End of class InputData
#

