#!/bin/bash

#SBATCH --job-name=SpexPlus_train_run
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:a100:1    # Request 1 Nvidia Tesla A100 GPU
#SBATCH --time=12:00:00

python3 train.py