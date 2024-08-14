#!/bin/bash

# Script to run a case and bin it in all the necessary ways you see fit.

#Run both cases first
command1="StandardCases/TCV74310_1/L1_raytracing.txt"
command2="StandardCases/TCV74310_1/L4_raytracing.txt"

mpiexec -np 8 python3 WKBeam.py trace $command1 &
mpiexec -np 8 python3 WKBeam.py trace $command2 &

wait 

python3 WKBeam.py bin StandardCases/TCV74310_1/L1_angular.txt &
python3 WKBeam.py bin StandardCases/TCV74310_1/L1_abs.txt &
python3 WKBeam.py bin StandardCases/TCV74310_1/L1_XZ.txt &

python3 WKBeam.py bin StandardCases/TCV74310_1/L4_angular.txt &
python3 WKBeam.py bin StandardCases/TCV74310_1/L4_abs.txt &
python3 WKBeam.py bin StandardCases/TCV74310_1/L4_XZ.txt 

wait 

echo "Both cases have been run."
