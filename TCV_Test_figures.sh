#!/bin/bash
command1=/home/devlamin/WKBacca_LUKE_cases/TCV_Test/RayTracing.txt
python3 WKBeam.py plot2d /home/devlamin/WKBacca_LUKE_cases/TCV_Test/Angular.txt &
python3 WKBeam.py plotabs /home/devlamin/WKBacca_LUKE_cases/TCV_Test/Absorption.txt &
python3 WKBeam.py plotbin /home/devlamin/WKBacca_LUKE_cases/TCV_Test/output/XZ_binned.hdf5 $command1 &
python3 WKBeam.py beamFluct /home/devlamin/WKBacca_LUKE_cases/TCV_Test/output/XZ_binned.hdf5 $command1 &
wait
echo "All done!"