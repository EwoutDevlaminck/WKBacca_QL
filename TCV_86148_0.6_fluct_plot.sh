#!/bin/bash
command1=/home/devlamin/WKbeam_simulations/TCV_86148_0.6_fluct/RayTracing.txt
python3 WKBeam.py plot2d /home/devlamin/WKbeam_simulations/TCV_86148_0.6_fluct/Angular.txt &
python3 WKBeam.py plotabs /home/devlamin/WKbeam_simulations/TCV_86148_0.6_fluct/Absorption.txt &
python3 WKBeam.py plotbin /home/devlamin/WKbeam_simulations/TCV_86148_0.6_fluct/output/XZ_binned.hdf5 $command1 &
python3 WKBeam.py beamFluct /home/devlamin/WKbeam_simulations/TCV_86148_0.6_fluct/output/XZ_binned.hdf5 $command1 &
wait
echo "All done!"
