#!/bin/bash

# Script to plot all results from the specified case

#Run both cases first
command1="WKBacca_cases/TCV_85352_0.9/L4_raytracing.txt"
#command2="WKBacca_cases/TCV_85352_0.9/L4_raytracing.txt"

python3 WKBeam.py plotbin /home/devlamin/WKBacca_LUKE_cases/TCV_85352_0.9/WKBeam_results/fluct/L4_binned_XZ_uniform.hdf5 $command1 &
python3 WKBeam.py plot2d WKBacca_cases/TCV_85352_0.9/L4_angular.txt &
python3 WKBeam.py plotabs WKBacca_cases/TCV_85352_0.9/L4_abs.txt &
python3 WKBeam.py plotabs WKBacca_cases/TCV_85352_0.9/L4_abs_nonuni.txt &


python3 WKBeam.py beamFluct /home/devlamin/WKBacca_LUKE_cases/TCV_85352_0.9/WKBeam_results/fluct/L4_binned_XZ_uniform.hdf5 $command1 &

wait

echo "Both cases have been run."
