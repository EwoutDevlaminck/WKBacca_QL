#!/bin/bash

# Script to run a case and bin it in all the necessary ways you see fit.

#Run both cases first
command1="StandardCases/TCV60612_1/TCV0fluct_raytracing.txt"
command2="StandardCases/TCV60612_1/TCVChellai_raytracing.txt"

mpiexec -np 8 python3 WKBeam.py trace $command1 &
mpiexec -np 8 python3 WKBeam.py trace $command2 &

wait 

python3 WKBeam.py bin StandardCases/TCV60612_1/TCVChellai_angular.txt &
python3 WKBeam.py bin StandardCases/TCV60612_1/TCVChellai_abs.txt &
python3 WKBeam.py bin StandardCases/TCV60612_1/TCVChellai_Flux3d.txt &
python3 WKBeam.py bin StandardCases/TCV60612_1/TCVChellai_XZ.txt &
python3 WKBeam.py bin StandardCases/TCV60612_1/TCVChellai_XYZ.txt &

python3 WKBeam.py bin StandardCases/TCV60612_1/TCV0fluct_angular.txt &
python3 WKBeam.py bin StandardCases/TCV60612_1/TCV0fluct_abs.txt &
python3 WKBeam.py bin StandardCases/TCV60612_1/TCV0fluct_Flux3d.txt &
python3 WKBeam.py bin StandardCases/TCV60612_1/TCV0fluct_XZ.txt 

wait 

echo "Both cases have been run."
