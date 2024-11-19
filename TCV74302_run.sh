#!/bin/bash

# Script to run a case and bin it in all the necessary ways you see fit.

#Run both cases first
command1="WKBacca_cases/TCV74302/L1_raytracing.txt"
#command2="WKBacca_cases/TCV74302/L1_raytracing.txt"

mpiexec -np 16 python3 WKBeam.py trace $command1 &
#mpiexec -np 8 python3 WKBeam.py trace $command2 &

wait 

python3 WKBeam.py bin WKBacca_cases/TCV74302/L1_angular.txt &
python3 WKBeam.py bin WKBacca_cases/TCV74302/L1_abs.txt &
python3 WKBeam.py bin WKBacca_cases/TCV74302/L1_flux3d.txt &
python3 WKBeam.py bin WKBacca_cases/TCV74302/L1_XZ.txt &
python3 WKBeam.py bin WKBacca_cases/TCV74302/L1_PsiThetaN.txt &

#python3 WKBeam.py bin WKBacca_cases/TCV74302/L1_angular.txt &
#python3 WKBeam.py bin WKBacca_cases/TCV74302/L1_abs.txt &
#python3 WKBeam.py bin WKBacca_cases/TCV74302/L1_Flux3d.txt &
#python3 WKBeam.py bin WKBacca_cases/TCV74302/L1_XZ.txt 

wait 

echo "Both cases have been run."
