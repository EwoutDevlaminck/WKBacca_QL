#!/bin/bash

# Script to plot all results from the specified case

#Run both cases first
command1="WKBacca_cases/TCV72644_1.25/L4_raytracing.txt"
#command2="WKBacca_cases/TCV72644_1.25/L4_raytracing.txt"

python3 WKBeam.py plotbin WKBacca_cases/TCV72644_1.25/output/L4_binned_XZ.hdf5 $command1 &
python3 WKBeam.py plot2d WKBacca_cases/TCV72644_1.25/L4_angular.txt &
python3 WKBeam.py plotabs WKBacca_cases/TCV72644_1.25/L4_abs.txt &
#python3 WKBeam.py flux WKBacca_cases/TCV72644_1.25/L4_flux3d.txt &
python3 WKBeam.py beamFluct WKBacca_cases/TCV72644_1.25/output/L4_binned_XZ.hdf5 $command1 &

wait

#python3 WKBeam.py plotbin WKBacca_cases/TCV74302/output/L1_binned_XZ.hdf5 $command2 &
#python3 WKBeam.py plot2d WKBacca_cases/TCV72644_1.25/L4_angular.txt &
#python3 WKBeam.py plotabs WKBacca_cases/TCV72644_1.25/L4_abs.txt &
#python3 WKBeam.py flux WKBacca_cases/TCV72644_1.25/L4_Flux3d.txt &

wait 

echo "Both cases have been run."
