
IPY = ipython3  # set to the desired ipython executable
NOTEBOOK = ../../../Tools/GenerateInputFiles/ProfileExtrapolation/profiles_extrapolation.ipynb

help:
	@echo " "
	@echo "Make the documentation of the density extrapolator by running"
	@echo " "
	@echo "    $ ipython3 nbconvert --to latex --post PDF                "
	@echo " "
	@echo "on the ipython notebook which defines the extrapolation procedures."
	@echo "By default, this REQUIRES ipython3 and pdflatex installed on the system."
	@echo "Previous versions of ipython might also work: The user can edit the"
	@echo "variable $(IPY) in the makefile." 
	@echo " "
	@echo "USAGE: $ gmake document "

document:
	@echo "Building the documentation for profile_extrapolarion.py"
	mkdir tmp
	cd tmp; $(IPY) nbconvert --to latex --SphinxTransformer.author='Hannes Weber and Omar Maj' --post PDF $(NOTEBOOK)
	/bin/mv tmp/profiles_extrapolation.pdf .
	/bin/rm -r tmp
