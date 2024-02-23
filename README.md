# Convolutional Deep Kernel Machines
This repository contains code for the paper "Convolutional Deep Kernel Machines" [https://arxiv.org/abs/2309.09814] published at ICLR 2024.

A `convdkm.yml` file is included to replicate our conda environment. 

Included in each `benchmark...` and `hyperparams...` folder are bash scripts to run the exact experiments from the paper. We recommend modifying these for submission on a GPU cluster. Total GPU time will be measured in the order of weeks using A100 GPUs. The largest experiments (the `512/1024/2048 inducing point` version) may require an 80GB A100.

Also included are the raw outputs from those runs that constituted the results in the paper (see `runs` subfolders).

The most important part of the training script/s is the nn.Sequential which specifies the model architecture:
https://github.com/edwardmilsom/convdkmpaper/blob/4b68d09a68ffd4a86e5e1c4a8373aac74608e889/18layer.py#L166-L179

One can modify this to try out different architectures with relative ease.
