#!/bin/bash

# Script to run a case and bin it in all the necessary ways you see fit.

command1="WKBacca_cases/TCV74301/L1_raytracing.txt"


mpiexec -np 32 python3 WKBeam.py trace $command1 &


wait 

python3 WKBeam.py bin WKBacca_cases/TCV74301/L1_angular.txt &
python3 WKBeam.py bin WKBacca_cases/TCV74301/L1_abs.txt &
#python3 WKBeam.py bin WKBacca_cases/TCV74301/L1_flux3d.txt &
python3 WKBeam.py bin WKBacca_cases/TCV74301/L1_XZ.txt &
python3 WKBeam.py bin WKBacca_cases/TCV74301/L1_PsiThetaN.txt &


wait 

echo "Both cases have been run."
