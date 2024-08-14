
# Python wrapper for westerino.f90
# Compile the NAG replacements with -fPIC first,
# and then generate the wrapper with "$ gmake wrap"

# include configuration file
include $(CURDIR)/../../../config.mk

# build
build: ecdisp wrap example

# Replacement for nag
ecdisp:
	@echo 'Building the module ecdisp ...'
	gfortran -c ecdisp.f90 -O3 -fdefault-real-8 -fPIC
	ar vq libecdisp.a *.o
	rm -f *.o
	@echo 'Module ecdisp successfully built! Continuing ...'

# Wrapper
wrap:
	@echo 'Running f2py ...'
	$(FTOPY) -m farinaECabsorption -c warmdamp.f90 -L. -lecdisp -I. --f90flags=-fdefault-real-8
	@echo 'f2py completed! '

# Run example
example:
	@echo ' '
	@echo 'Running example ...'
	$(PYINT) test_abs.py

# Clean
clean:
	rm -f ecdisp.mod
	rm -f libecdisp.a
	rm -f *.so	
