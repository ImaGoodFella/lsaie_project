#!/bin/bash
#SBATCH --account=large-sc-2
#SBATCH --time=00:30:00
#SBATCH --job-name=ds-bench
#SBATCH --output=/iopsstor/scratch/cscs/ldionysiou/project/logs/%x-%j.out
#SBATCH --error=/iopsstor/scratch/cscs/ldionysiou/project/logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=72
#SBATCH --mem=460000
#SBATCH --environment=/iopsstor/scratch/cscs/ldionysiou/env.toml
#SBATCH --no-requeue


export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "START TIME: $(date)"

ASSIGNMENT_DIR=/iopsstor/scratch/cscs/ldionysiou/project
LOG_DIR=$ASSIGNMENT_DIR/logs
mkdir -p "$LOG_DIR"

CMD_PREFIX="export HF_HUB_ENABLE_HF_TRANSFER=0;"
MODES=("stage0" "stage1" "stage2")
CONFIGS=("ds_stage0.json" "ds_stage1.json" "ds_stage2.json")
SEQ_LEN=2048

for i in "${!MODES[@]}"; do
    MODE=${MODES[$i]}
    CFG=${CONFIGS[$i]}
    RUN_NAME="ds_${MODE}"

    echo "============================================================"
    echo "Running DeepSpeed benchmark: $MODE (config: $CFG)"
    echo "============================================================"

    TRAINING_CMD="python3 -m torch.distributed.run \
        --standalone \
        --nproc_per_node=4 \
        $ASSIGNMENT_DIR/train.py \
        --sequence-length $SEQ_LEN \
        --deepspeed \
        --deepspeed_config $ASSIGNMENT_DIR/$CFG"

    srun --mpi=none bash -lc "$CMD_PREFIX $TRAINING_CMD" \
        2>&1 | tee "$LOG_DIR/${RUN_NAME}.log"

    echo "Finished mode: $MODE at $(date)"
done

echo "END TIME: $(date)"
