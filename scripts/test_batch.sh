#!/bin/bash

#SBATCH --account=large-sc-proj
#SBATCH --time=00:07:59
#SBATCH --job-name=lsai
#SBATCH --output=/iopsstor/scratch/cscs/%u/project/logs/%x-%j.out
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --mem=460000
#SBATCH --environment=/iopsstor/scratch/cscs/[User]/ngc_pt_jan.toml   # Vanilla 25.01 PyTorch NGC Image 
#SBATCH --no-requeue	# Prevent Slurm to requeue the job if the execution crashes (e.g. node failure) so we don't loose the logs



deepspeed /iopsstor/scratch/cscs/[User]/project/src/main.py --stage=0