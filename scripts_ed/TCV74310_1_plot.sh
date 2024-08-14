#!/bin/bash

# Script to plot all results from the specified case

#Run both cases first
command1="StandardCases/TCV74310_1/L1_raytracing.txt"
command2="StandardCases/TCV74310_1/L4_raytracing.txt"

python3 WKBeam.py plotbin StandardCases/TCV74310_1/output/L1_binned_XZ.hdf5 $command1 &
python3 WKBeam.py plot2d StandardCases/TCV74310_1/L1_angular.txt &
python3 WKBeam.py plotabs StandardCases/TCV74310_1/L1_abs.txt &
python3 WKBeam.py beamFluct StandardCases/TCV74310_1/output/L1_binned_XZ.hdf5 $command1 

wait

python3 WKBeam.py plotbin StandardCases/TCV74310_1/output/L4_binned_XZ.hdf5 $command2 &
python3 WKBeam.py plot2d StandardCases/TCV74310_1/L4_angular.txt &
python3 WKBeam.py plotabs StandardCases/TCV74310_1/L4_abs.txt &
python3 WKBeam.py beamFluct StandardCases/TCV74310_1/output/L4_binned_XZ.hdf5 $command2 

wait 

python3 WKBeam.py beamFluct StandardCases/TCV74310_1/output/L4_binned_XZ.hdf5 StandardCases/TCV74310_1/output/L1_binned_XZ.hdf5 $command1 &
python3 WKBeam.py plotabs StandardCases/TCV74310_1/L1_abs.txt StandardCases/TCV74310_1/L4_abs.txt 

echo "Both cases have been run."
