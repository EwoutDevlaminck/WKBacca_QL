
# Python wrapper for westerino.f90
# Compile the NAG replacements with -fPIC first,
# and then generate the wrapper with "$ gmake wrap"

# include the configuration file
include $(CURDIR)/../../../config.mk

# build
build: nag_repl wrap example

# Replacement for nag
nag_repl:
	@echo 'Building NAG replacements ...'
	cd ./nag/; rm -f libnag.a
	cd ./nag/src/; gfortran -c *.f -O3 -fdefault-real-8 -fPIC
	cd ./nag/src/; ar vq libnag.a *.o
	mv ./nag/src/libnag.a ./nag/. 
	rm -f ./nag/src/*.o
	@echo 'NAG replacements successfully built! Continuing ...'

# Wrapper
wrap:
	@echo 'Running f2py ...'
	$(FTOPY) -m westerinoECabsorption -c westerino.pyf westerino.f90 -L./nag/ -lnag --f90flags=-fdefault-real-8
	@echo 'f2py completed! '

# Run example
example:
	@echo ' '
	@echo 'Running example ...'
	$(PYINT) test_abs.py

# Clean
clean:
	rm -f *.so
	cd nag ; rm -f libnag.a
