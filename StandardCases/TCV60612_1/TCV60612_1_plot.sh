#!/bin/bash

# Script to plot all results from the specified case

#Run both cases first
command1="StandardCases/TCV60612_1/TCV0fluct_raytracing.txt"
command2="StandardCases/TCV60612_1/TCVChellai_raytracing.txt"

python3 WKBeam.py plotbin StandardCases/TCV60612_1/output/TCVChellai_binned_XZ.hdf5 $command2 &
python3 WKBeam.py plot2d StandardCases/TCV60612_1/TCVChellai_angular.txt &
python3 WKBeam.py plotabs StandardCases/TCV60612_1/TCVChellai_abs.txt &
python3 WKBeam.py flux StandardCases/TCV60612_1/TCVChellai_Flux3d.txt &
python3 WKBeam.py beam3d StandardCases/TCV60612_1/TCVChellai_XYZ.txt &
python3 WKBeam.py beamFluct StandardCases/TCV60612_1/output/TCVChellai_binned_XZ.hdf5 $command2 &

wait

python3 WKBeam.py plotbin StandardCases/TCV60612_1/output/TCV0fluct_binned_XZ.hdf5 $command1 &
python3 WKBeam.py plot2d StandardCases/TCV60612_1/TCV0fluct_angular.txt &
python3 WKBeam.py plotabs StandardCases/TCV60612_1/TCV0fluct_abs.txt &
python3 WKBeam.py flux StandardCases/TCV60612_1/TCV0fluct_Flux3d.txt &

wait 

echo "Both cases have been run."
