#!/bin/bash
command1=/home/devlamin/WKbeam_simulations/TCV_85352_0.9_fluct/RayTracing.txt
mpiexec -np 24 python3 WKBeam.py trace $command1
wait
python3 WKBeam.py bin /home/devlamin/WKbeam_simulations/TCV_85352_0.9_fluct/Angular.txt
python3 WKBeam.py bin /home/devlamin/WKbeam_simulations/TCV_85352_0.9_fluct/Absorption.txt
python3 WKBeam.py bin /home/devlamin/WKbeam_simulations/TCV_85352_0.9_fluct/XZ.txt
python3 WKBeam.py bin /home/devlamin/WKbeam_simulations/TCV_85352_0.9_fluct/RhoThetaN.txt
wait
echo "All done!"
