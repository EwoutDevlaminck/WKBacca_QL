#!/bin/bash
command1=/home/devlamin/WKBacca_LUKE_cases/TCV_Test/RayTracing.txt
mpiexec -np 32 python3 WKBeam.py trace $command1
wait
python3 WKBeam.py bin /home/devlamin/WKBacca_LUKE_cases/TCV_Test/Angular.txt
python3 WKBeam.py bin /home/devlamin/WKBacca_LUKE_cases/TCV_Test/Absorption.txt
python3 WKBeam.py bin /home/devlamin/WKBacca_LUKE_cases/TCV_Test/XZ.txt
python3 WKBeam.py bin /home/devlamin/WKBacca_LUKE_cases/TCV_Test/RhoThetaN.txt
wait
echo "All done!"