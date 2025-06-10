#!/bin/bash
command1=/home/devlamin/WKbeam_simulations/TCV_86149_0.6_fluct/RayTracing.txt
mpiexec -np 32 python3 WKBeam.py trace $command1
wait
python3 WKBeam.py bin /home/devlamin/WKbeam_simulations/TCV_86149_0.6_fluct/Angular.txt
python3 WKBeam.py bin /home/devlamin/WKbeam_simulations/TCV_86149_0.6_fluct/Absorption.txt
python3 WKBeam.py bin /home/devlamin/WKbeam_simulations/TCV_86149_0.6_fluct/AbsorptionUni.txt
python3 WKBeam.py bin /home/devlamin/WKbeam_simulations/TCV_86149_0.6_fluct/XZ.txt
python3 WKBeam.py bin /home/devlamin/WKbeam_simulations/TCV_86149_0.6_fluct/RhoThetaN.txt
wait

echo "Ray tracing and binning done!"
mpiexec -np 22 python3 WKBeam.py QLdiff /home/devlamin/WKbeam_simulations/TCV_86149_0.6_fluct/QLdiff.txt

echo " Quasilinear diffusion calculation done!"
