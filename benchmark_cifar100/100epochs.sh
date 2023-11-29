#!/bin/bash

#This script runs the 18-layer model for 100 epochs on CIFAR-100 with different numbers of inducing points.
#It is recommended to run the following on a GPU cluster by inserting your own scheduler commands.

#Largest model requires 40GB+ of GPU memory, e.g. A100.
for seed in 0 1 2 3; do
	for indscale in 32; do
		python ../18layer.py --init_lr=0.01 --n_ind_scale=$indscale --dof=0.01 --final_layer=GAP --likelihood=categorical --seed=$seed --bn_indnorm="global" --bn_indscale="global" --bn_tnorm="global" --bn_tscale="global" --device="cuda" --data_folder_path="../../data/" --dataset=CIFAR100
	done
done

for seed in 0 1 2 3; do
	for indscale in 16 8 4 2 1; do
		python ../18layer.py --init_lr=0.01 --n_ind_scale=$indscale --dof=0.01 --final_layer=GAP --likelihood=categorical --seed=$seed --bn_indnorm="global" --bn_indscale="global" --bn_tnorm="global" --bn_tscale="global" --device="cuda" --data_folder_path="../../data/" --dataset=CIFAR100
	done
done
