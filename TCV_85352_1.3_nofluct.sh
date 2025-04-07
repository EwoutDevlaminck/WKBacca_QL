#!/bin/bash
command1=/home/devlamin/WKbeam_simulations/TCV_85352_1.3_nofluct/RayTracing.txt
#mpiexec -np 20 python3 WKBeam.py trace $command1
wait
python3 WKBeam.py bin /home/devlamin/WKbeam_simulations/TCV_85352_1.3_nofluct/Angular.txt
python3 WKBeam.py bin /home/devlamin/WKbeam_simulations/TCV_85352_1.3_nofluct/Absorption.txt
python3 WKBeam.py bin /home/devlamin/WKbeam_simulations/TCV_85352_1.3_nofluct/AbsorptionUni.txt
python3 WKBeam.py bin /home/devlamin/WKbeam_simulations/TCV_85352_1.3_nofluct/XZ.txt
python3 WKBeam.py bin /home/devlamin/WKbeam_simulations/TCV_85352_1.3_nofluct/RhoThetaN.txt
wait
echo "All done!"
