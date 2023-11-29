#!/bin/bash

#It is recommended to run the following on a GPU cluster by inserting your own scheduler commands.


#This script runs the 18-layer model for 100 epochs on CIFAR-100 while changing various hyperparameters one-by-one.

#DOF
for seed in 0 1 2 3; do
	for dof in 0 0.0001 0.001 0.01 0.1 1 10 100 1000 10000 inf; do
		python ../18layer.py --init_lr=0.01 --n_ind_scale=8 --dof=$dof --final_layer=GAP --likelihood=categorical --seed=$seed --bn_indnorm="global" --bn_indscale="global" --bn_tnorm="global" --bn_tscale="global" --device="cuda" --data_folder_path="../../data/" --dataset=CIFAR100
	done
done

#Normalisation and scaling schemes
for seed in 0 1 2 3; do

	norm_pairs=("none,none" "global,global" "local,local" "local,image" "global,location")
	for pair in "${norm_pairs[@]}"; do
		IFS=',' read -r indnorm tnorm <<< "$pair"
		python ../18layer.py --init_lr=0.01 --n_ind_scale=8 --dof=1 --final_layer=GAP --likelihood=categorical --seed=$seed --bn_indnorm=$indnorm --bn_indscale="global" --bn_tnorm=$tnorm --bn_tscale="global" --device="cuda" --data_folder_path="../../data/" --dataset=CIFAR100
	done

	scale_pairs=("none,none" "global,global" "global,location" "local,none" "local,global", "local,location")
	for pair in "${scale_pairs[@]}"; do
		IFS=',' read -r indscale tscale <<< "$pair"
		python ../18layer.py --init_lr=0.01 --n_ind_scale=8 --dof=1 --final_layer=GAP --likelihood=categorical --seed=$seed --bn_indnorm="global" --bn_indscale=$indscale --bn_tnorm="global" --bn_tscale=$tscale --device="cuda" --data_folder_path="../../data/" --dataset=CIFAR100
	done
done

#Top layer type
for seed in 0 1 2 3; do
	for final_layer in GAP GAPMixup BFC BFCMixup; do
		python ../18layer.py --init_lr=0.01 --n_ind_scale=8 --dof=1 --final_layer=$final_layer --likelihood=categorical --seed=$seed --bn_indnorm="global" --bn_indscale="global" --bn_tnorm="global" --bn_tscale="global" --device="cuda" --data_folder_path="../../data/" --dataset=CIFAR100
	done
done

#Likelihood
for seed in 0 1 2 3; do
	for likelihood in gaussian categorical; do
		python ../18layer.py --init_lr=0.01 --n_ind_scale=8 --dof=1 --final_layer=GAP --likelihood=$likelihood --seed=$seed --bn_indnorm="global" --bn_indscale="global" --bn_tnorm="global" --bn_tscale="global" --device="cuda" --data_folder_path="../../data/" --dataset=CIFAR100
	done
done

