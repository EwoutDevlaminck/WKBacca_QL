#!/bin/bash
command1=/home/devlamin/WKbeam_simulations/TCV_86148_0.6_nofluct_pext/RayTracing.txt
python3 WKBeam.py plot2d /home/devlamin/WKbeam_simulations/TCV_86148_0.6_nofluct_pext/Angular.txt &
python3 WKBeam.py plotabs /home/devlamin/WKbeam_simulations/TCV_86148_0.6_nofluct_pext/Absorption.txt &
python3 WKBeam.py plotbin /home/devlamin/WKbeam_simulations/TCV_86148_0.6_nofluct_pext/output/XZ_binned.hdf5 $command1 &
python3 WKBeam.py beamFluct /home/devlamin/WKbeam_simulations/TCV_86148_0.6_nofluct_pext/output/XZ_binned.hdf5 $command1 &
wait
echo "All done!"
