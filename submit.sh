#!/bin/bash
#SBATCH --account=large-sc-2
#SBATCH --time=00:17:59
#SBATCH --job-name=lsai-deepspeed
#SBATCH --output=/iopsstor/scratch/cscs/ldionysiou/project/logs/%x-%j.out
#SBATCH --error=/iopsstor/scratch/cscs/ldionysiou/project/logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1     # <--- only ONE task
#SBATCH --gpus-per-task=4       # <--- all 4 GPUs to that task
#SBATCH --cpus-per-task=72
#SBATCH --mem=460000
#SBATCH --environment=/iopsstor/scratch/cscs/ldionysiou/env.toml
#SBATCH --no-requeue

echo "START TIME: $(date)"

export HF_HUB_ENABLE_HF_TRANSFER=0
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK


ASSIGNMENT_DIR=/iopsstor/scratch/cscs/ldionysiou/project

CMD_PREFIX=""  

TRAINING_CMD="python3 -m torch.distributed.run \
    --standalone \
    --nproc_per_node=4 \
    $ASSIGNMENT_DIR/train.py \
    --deepspeed \
    --deepspeed_config $ASSIGNMENT_DIR/ds_stage2.json \
    --batch-size=1 \
    --learning-rate=1e-5 \
    --lr-warmup-steps=10 \
    --grad-max-norm=1.0"

srun --mpi=none bash -lc "export HF_HUB_ENABLE_HF_TRANSFER=0; $CMD_PREFIX $TRAINING_CMD"

echo 'END TIME:' "$(date)"
