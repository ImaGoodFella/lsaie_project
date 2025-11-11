#!/bin/bash

#SBATCH --account=large-sc-proj
#SBATCH --time=00:07:59
#SBATCH --job-name=lsai
#SBATCH --output=/iopsstor/scratch/cscs/%u/project/logs/%x-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --mem=460000
#SBATCH --environment=/iopsstor/scratch/cscs/[USER]/ngc_pt_jan.toml   # Vanilla 25.01 PyTorch NGC Image 
#SBATCH --no-requeue	# Prevent Slurm to requeue the job if the execution crashes (e.g. node failure) so we don't loose the logs

echo "START TIME: $(date)"

# Set up ENV
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
ASSIGNMENT_DIR="/iopsstor/scratch/cscs/[USER]/project"

CMD_PREFIX="numactl --membind=0-3"

TRAINING_CMD="nsys profile -s none -w true \
  --trace='nvtx,cudnn,cublas,cuda' \
  --output=/iopsstor/scratch/cscs/[USER]/project/nsys-trace.nsys-rep \
  --force-overwrite true \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop -x true numactl --membind=0-3 python3 $ASSIGNMENT_DIR/train.py --profile"

srun --cpus-per-task=$SLURM_CPUS_PER_TASK bash -c "$CMD_PREFIX $TRAINING_CMD"

echo "END TIME: $(date)"
