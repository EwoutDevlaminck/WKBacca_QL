
# Makefile for WKBeam
# -------------------
# This Makefile allows the user to build all Cython and Fotran extensions.
# Fortran modules can be compiled manually and tested by using the Makefile in the
# corresponding subdirectory of RayTracing/lib.

# Include configuration file
include config.mk

# Some variables
RAYTRACING = $(CURDIR)/RayTracing
BINNING = $(CURDIR)/Binning
COMMON = $(CURDIR)/CommonModules
TOOLS = $(CURDIR)/Tools
ECDISP = $(RAYTRACING)/lib/ecdisp
WESTERINO = $(RAYTRACING)/lib/westerino
NAG = $(WESTERINO)/nag

# Target: help
help:
	@echo " "
	@echo "Makefile for WKBeam:"
	@echo "This Makefile allows the user to build all pre-compiled"
	@echo "modules at once. The single Fortran modules can also be"
	@echo "compiled manually by using the makefiles in their subdirectory"
	@echo "in RayTracing/code; those makefiles also have the possibility"
	@echo "to launch a few simple stand-alone tests for their modules. "
	@echo " "
	@echo "The executable for  the python interpreter may vary from system"
	@echo "to system, (e.g., python3 is often used for python version 3"
	@echo "and above). Thereferore the python executable should be"
	@echo "set in the configuration file config.mk."	
	@echo " "
	@echo "USAGE: "
	@echo " $ gmake help     --> this help message. "
	@echo " $ gmake clean    --> clean up."
	@echo " $ gmake code     --> build all cython and Fortran extensions." 
	@echo " $ gmake build_c  --> build cython extensions."
	@echo " $ gmake build_f  --> build fortran extensions."
	@echo " "

# Absorption routine ecdisp (D. Farina routine)
ecdisp:
	@echo "Building the module ecdisp ..."
	cd $(ECDISP) ; gfortran -c ecdisp.f90 -O3 -fdefault-real-8 -fPIC
	cd $(ECDISP) ; ar vq libecdisp.a *.o
	cd $(ECDISP) ; rm -f *.o
	cd $(ECDISP) ; $(FTOPY) -m farinaECabsorption -c warmdamp.f90 -L. -lecdisp -I. --f90flags=-fdefault-real-8

# Absorption routine westerino (E. Westerhof routine)
westerino: nag_repl
	@echo "Building westerino ..."
	cd $(WESTERINO) ; $(FTOPY) -m westerinoECabsorption -c westerino.pyf westerino.f90 -L. -lnag --f90flags=-fdefault-real-8

# Replacements for NAG required by westerino
nag_repl:
	@echo "Building NAG replacements ..."
	cd $(WESTERINO) ; rm -f libnag.a
	cd $(NAG)/src/ ; gfortran -c *.f -O3 -fdefault-real-8 -fPIC
	cd $(NAG)/src/ ; ar vq libnag.a *.o
	mv $(NAG)/src/libnag.a $(WESTERINO)/. 
	rm -f $(NAG)/src/*.o

# Fortran extensions
build_f: ecdisp westerino

# Cython extensions
build_c:
	$(PYINT) setup.py build_ext --inplace

# Main target
code: build_c build_f

# Utility targets 
PYCACHE = __pycache__ *.pyc
clean:
	@echo "Cleaning root directory of the project ..."
	rm -rf build
	rm -rf $(PYCACHE)
	@echo "Cleaning the RayTracing tree ..."
	cd $(RAYTRACING) ; rm -rf $(PYCACHE)
	cd $(RAYTRACING)/lib ; rm -rf $(PYCACHE)
	cd $(RAYTRACING)/lib/ecdisp ; rm -rf $(PYCACHE) *.so *.a *.mod
	cd $(RAYTRACING)/lib/westerino ; rm -rf $(PYCACHE) *.so *.a
	cd $(RAYTRACING)/modules ; rm -rf $(PYCACHE) *.so *.c
	cd $(RAYTRACING)/modules/scattering ; rm -rf $(PYCACHE) *.so *.c
	@echo "Cleaning the Binning tree ..."
	cd $(BINNING) ; rm -rf $(PYCACHE)
	cd $(BINNING)/modules ; rm -rf $(PYCACHE) *.so *.c
	@echo "Cleaning the CommonModules tree ..."
	cd $(COMMON) ;  rm -rf $(PYCACHE)
	@echo "Cleaning the Tools tree ..."
	cd $(TOOLS) ; rm -rf $(PYCACHE)
	cd $(TOOLS)/PlotData ; rm -rf $(PYCACHE) 
	cd $(TOOLS)/PlotData/PlotBinnedData ; rm -rf $(PYCACHE) 
	cd $(TOOLS)/PlotData/PlotAbsorptionProfile ; rm -rf $(PYCACHE)
	cd $(TOOLS)/PlotData/PlotEquilibrium ; rm -rf $(PYCACHE)
	cd $(TOOLS)/PlotData/PlotFluctuations ; rm -rf $(PYCACHE)
