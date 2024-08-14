#!/bin/bash

# Script to run multiple WKBeam simulations, where a given parameter can be changed in the config file.

LPerp_values=("0.3" "0.6" "1.5")

# Enter a loop where we execute the bash script

for val in "${LPerp_values[@]}"; do
	newline="scatteringLengthPerp = $val"
	echo $newline
	# Find and replace the whole line
	sed -i "/scatteringLengthPerp/ s/.*/$newline/" ./Benchmark_JC_Analytical/L2_raytracing.txt
	
	wait
	./Benchmark_JC_run.sh
	wait
	mkdir ./Benchmark_JC_Analytical/Output_fluct_S1.1/LPerp_$val
	cp ./Benchmark_JC_Analytical/output/L2_binned_abs.hdf5 ./Benchmark_JC_Analytical/Output_fluct_S1.1/LPerp_$val
	cp ./Benchmark_JC_Analytical/output/L2_binned_angular.hdf5 ./Benchmark_JC_Analytical/Output_fluct_S1.1/LPerp_$val
	cp ./Benchmark_JC_Analytical/output/L2_binned_XZ.hdf5 ./Benchmark_JC_Analytical/Output_fluct_S1.1/LPerp_$val
	
	rm -r ./Benchmark_JC_Analytical/output/*

done
echo "Both cases have been run."
