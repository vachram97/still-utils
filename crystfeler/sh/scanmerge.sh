#!/bin/bash

for pushres in $(python3 -c "print(*[i/10 for i in range(1,40,5)])"); do 
	for method in process_hkl; do
		for min_cc in -0.1 0.0 0.1 0.2; do
			for string_params in ' ' '--scale'; do
				analyzer -i laststream --merging "$method" --model "$model" --symmetry '2/m_uab' --iterations "$iterations" --lowres 30.0 --highres 2.6 --pushres "$pushres" --min_cc "$min_cc" --string_params "$string_params"
			done
		done
	done
done

for pushres in $(python3 -c "print(*[i/10 for i in range(1,40,5)])"); do 
	for method in partialator; do
		for iterations in 0 1 2; do
			for model in xsphere unity; do
				analyzer -i laststream --merging "$method" --model "$model" --symmetry '2/m_uab' --iterations "$iterations" --lowres 30.0 --highres 2.6 --pushres "$pushres" --min_cc "$min_cc" --string_params "$string_params"
			done
		done
	done
done

# srun -c 8 --comment "..." scanmerge.sh
