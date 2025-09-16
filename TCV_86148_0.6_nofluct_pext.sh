#!/bin/bash
command1=/home/devlamin/WKbeam_simulations/TCV_86148_0.6_nofluct_pext/RayTracing.txt
#mpiexec -np 32 python3 WKBeam.py trace $command1
#wait
#python3 WKBeam.py bin /home/devlamin/WKbeam_simulations/TCV_86148_0.6_nofluct_pext/Angular.txt
#python3 WKBeam.py bin /home/devlamin/WKbeam_simulations/TCV_86148_0.6_nofluct_pext/Absorption.txt
#python3 WKBeam.py bin /home/devlamin/WKbeam_simulations/TCV_86148_0.6_nofluct_pext/AbsorptionUni.txt
#python3 WKBeam.py bin /home/devlamin/WKbeam_simulations/TCV_86148_0.6_nofluct_pext/XZ.txt
#python3 WKBeam.py bin /home/devlamin/WKbeam_simulations/TCV_86148_0.6_nofluct_pext/RhoThetaN.txt
#wait

echo "Ray tracing and binning done!"
mpiexec -np 22 python3 WKBeam.py QLdiff /home/devlamin/WKbeam_simulations/TCV_86148_0.6_nofluct_pext/QLdiff.txt

echo " Quasilinear diffusion calculation done!"
