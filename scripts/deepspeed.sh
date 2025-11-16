#!/bin/bash

# Usage: sbatch --export=STAGE=0 deepspeed.sh
# Or: STAGE=0 sbatch deepspeed.sh
# Or set default: sbatch deepspeed.sh (uses stage 0)

#SBATCH --account=a-infra02
#SBATCH --time=00:07:59
#SBATCH --job-name=lsai
#SBATCH --output=/users/%u/LSAIE-Project/scripts/logs/deepspeed/%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --mem=460000
#SBATCH --environment=/users/rasteiger/LSAIE-Project/docker/lsaie_project.toml
#SBATCH --no-requeue	# Prevent Slurm to requeue the job if the execution crashes (e.g. node failure) so we don't loose the logs

# Get stage from environment variable, default to 0
STAGE=${STAGE:-0}

echo "START TIME: $(date)"
echo "Running DeepSpeed Stage: $STAGE"
echo "Job ID: $SLURM_JOB_ID"
echo "Output will be in: logs/deepspeed/$SLURM_JOB_ID.out"

# Create a symlink with stage info for easier identification
LOG_DIR="/users/$USER/LSAIE-Project/scripts/logs/deepspeed"
ln -sf "$LOG_DIR/$SLURM_JOB_ID.out" "$LOG_DIR/stage${STAGE}_latest.out"

# Set up ENV
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
ASSIGNMENT_DIR="/users/$USER/LSAIE-Project"

CMD_PREFIX="numactl --membind=0-3"

TRAINING_CMD="python3 -m torch.distributed.run \
    --standalone \
    --nproc_per_node=4 \
    $ASSIGNMENT_DIR/src/train.py \
    --batch-size 1 \
    --learning-rate 5e-5 \
    --lr-warmup-steps 100 \
    --training-steps 1000 \
    --sequence-length 2048 \
    --deepspeed \
    --deepspeed-config $ASSIGNMENT_DIR/configs/deepspeed/stage${STAGE}.json \
    "

srun --cpus-per-task=$SLURM_CPUS_PER_TASK bash -c "$CMD_PREFIX $TRAINING_CMD"

echo "END TIME: $(date)"
