#!/bin/bash
command1=/home/devlamin/WKbeam_simulations/TCV_86148_1.3_fluct/RayTracing.txt
mpiexec -np 32 python3 WKBeam.py trace $command1
wait
python3 WKBeam.py bin /home/devlamin/WKbeam_simulations/TCV_86148_1.3_fluct/Angular.txt
python3 WKBeam.py bin /home/devlamin/WKbeam_simulations/TCV_86148_1.3_fluct/Absorption.txt
python3 WKBeam.py bin /home/devlamin/WKbeam_simulations/TCV_86148_1.3_fluct/AbsorptionUni.txt
python3 WKBeam.py bin /home/devlamin/WKbeam_simulations/TCV_86148_1.3_fluct/XZ.txt
python3 WKBeam.py bin /home/devlamin/WKbeam_simulations/TCV_86148_1.3_fluct/RhoThetaN.txt
wait

echo "Ray tracing and binning done!"
mpiexec -np 22 python3 WKBeam.py QLdiff /home/devlamin/WKbeam_simulations/TCV_86148_1.3_fluct/QLdiff.txt

echo " Quasilinear diffusion calculation done!"
