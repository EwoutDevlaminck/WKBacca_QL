
#    WKBeam - A Monte-Carlo solver for the wave kinetic equation    
#    setup script for the compilation of Cython extensions          
#
#    Always run the setup script from the root folder of the project.
#--------------------------------------------------------------------

# Import statements
import numpy
from distutils.core import setup
from distutils.extension import Extension 
from Cython.Distutils import build_ext

# All module name should be fully qualified
# All path should be fully qualified starting at the root folder
# The list include_dirs must contain the '.' for setup to search all the subfolder of the root folder
# ... cython modules ...
cython_modules = [
    Extension(
        name="RayTracing.modules.dispersion_matrix_cfunctions", 
        sources=["RayTracing/modules/dispersion_matrix_cfunctions.pyx"], 
        include_dirs=['.', numpy.get_include()]
    ),
    Extension(
        name="RayTracing.modules.atanrightbranch", 
        sources=["RayTracing/modules/atanrightbranch.pyx"], 
        include_dirs=['.', numpy.get_include()]
    ),
    Extension(
        name="RayTracing.modules.scattering.gaussianmodel_cfunctions", 
        sources=["RayTracing/modules/scattering/gaussianmodel_cfunctions.pyx"], 
        include_dirs=['.', numpy.get_include()]
    ),
    Extension(
        name="RayTracing.modules.scattering.shafermodel_cfunctions", 
        sources=["RayTracing/modules/scattering/shafermodel_cfunctions.pyx"], 
        include_dirs=['.', numpy.get_include()]
    ),
    Extension(
        name="Binning.modules.binning", 
        sources=["Binning/modules/binning.pyx"],  
        include_dirs=['.', numpy.get_include()]
    )
]

# Setting up the cython extensions    
setup(
    name='WKBEAM',
    cmdclass={'build_ext': build_ext},
    ext_modules=cython_modules,
    script_args=['build_ext'],
    options={'build_ext':{'inplace':True, 'force':True}}
)

