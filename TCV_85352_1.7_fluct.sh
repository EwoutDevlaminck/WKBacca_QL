#!/bin/bash
command1=/home/devlamin/WKbeam_simulations/TCV_85352_1.7_fluct/RayTracing.txt
#mpiexec -np 32 python3 WKBeam.py trace $command1
wait
python3 WKBeam.py bin /home/devlamin/WKbeam_simulations/TCV_85352_1.7_fluct/Angular.txt
python3 WKBeam.py bin /home/devlamin/WKbeam_simulations/TCV_85352_1.7_fluct/Absorption.txt
python3 WKBeam.py bin /home/devlamin/WKbeam_simulations/TCV_85352_1.7_fluct/AbsorptionUni.txt
python3 WKBeam.py bin /home/devlamin/WKbeam_simulations/TCV_85352_1.7_fluct/XZ.txt
python3 WKBeam.py bin /home/devlamin/WKbeam_simulations/TCV_85352_1.7_fluct/RhoThetaN.txt
wait
echo "All done!"
