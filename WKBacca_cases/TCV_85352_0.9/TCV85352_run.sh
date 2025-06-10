#!/bin/bash

# Script to run a case and bin it in all the necessary ways you see fit.

#Run both cases first
command1="WKBacca_cases/TCV_85352_0.9/L4_raytracing.txt"
#command2="WKBacca_cases/TCV_85352_0.9/L4_raytracing.txt"

mpiexec -np 32 python3 WKBeam.py trace $command1 &
#mpiexec -np 8 python3 WKBeam.py trace $command2 &

wait 

python3 WKBeam.py bin WKBacca_cases/TCV_85352_0.9/L4_angular.txt &
python3 WKBeam.py bin WKBacca_cases/TCV_85352_0.9/L4_abs.txt &
python3 WKBeam.py bin WKBacca_cases/TCV_85352_0.9/L4_XZ.txt &
python3 WKBeam.py bin WKBacca_cases/TCV_85352_0.9/L4_PsiThetaN.txt &

#python3 WKBeam.py bin WKBacca_cases/TCV_85352_0.9/L4_angular.txt &
#python3 WKBeam.py bin WKBacca_cases/TCV_85352_0.9/L4_abs.txt &
#python3 WKBeam.py bin WKBacca_cases/TCV_85352_0.9/L4_Flux3d.txt &
#python3 WKBeam.py bin WKBacca_cases/TCV_85352_0.9/L4_XZ.txt 

wait 

echo "Both cases have been run."
