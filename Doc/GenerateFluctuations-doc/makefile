
IPY = ipython3  # set to the desired ipython executable
NOTEBOOK = ../../../Tools/GenerateFluctuations/DensityFluctuations.ipynb

help:
	@echo " "
	@echo "Make the documentation of DensityFluctuations.py by running"
	@echo " "
	@echo "    $ ipython3 nbconvert --to latex --post PDF                "
	@echo " "
	@echo "By default, this REQUIRES ipython3 and pdflatex installed on the system."
	@echo "Previous versions of ipython might also work: The user can edit the"
	@echo "variable $(IPY) in the makefile." 
	@echo " "
	@echo "USAGE: $ gmake document "

document:
	@echo "Building the documentation for DensityFluctuations.py"
	mkdir tmp
	cd tmp; $(IPY) nbconvert --to latex --SphinxTransformer.author='Hannes Weber and Omar Maj' --post PDF $(NOTEBOOK)
	/bin/mv tmp/DensityFluctuations.pdf .
	/bin/rm -r tmp
