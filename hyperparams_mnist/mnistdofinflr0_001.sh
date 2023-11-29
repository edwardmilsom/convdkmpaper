#!/bin/bash

#It is recommended to run the following on a GPU cluster by inserting your own scheduler commands.


#This script runs the 18-layer model for 100 epochs on MNIST using infinite regularisaiton strength, but with a smaller learning rate as this experiment failed for the larger learning rate.

#DOF
for seed in 0 1 2 3; do
	for dof in inf; do
		python ../18layer.py --init_lr=0.001 --n_ind_scale=8 --dof=$dof --final_layer=GAP --likelihood=categorical --seed=$seed --bn_indnorm="global" --bn_indscale="global" --bn_tnorm="global" --bn_tscale="global" --device="cuda" --data_folder_path="../../data/" --dataset=MNIST
	done
done
