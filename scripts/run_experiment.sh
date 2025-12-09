#!/bin/bash

# Usage: ./run_experiment.sh [STAGE_NAME1 STAGE_NAME2 ...]
# If no arguments provided, runs all available configs
# Example: ./run_experiment.sh 3_deepcompile
#          ./run_experiment.sh 2 3 superoffload
#          ./run_experiment.sh   # runs all configs

CONFIG_DIR="/users/$USER/LSAIE-Project/configs/deepspeed"

# Get all available stage names
get_all_stages() {
    ls -1 "$CONFIG_DIR"/stage_*.json 2>/dev/null | xargs -n1 basename | sed 's/stage_//' | sed 's/.json//'
}

# Function to submit a single experiment
submit_experiment() {
    local STAGE_NAME="$1"
    local CONFIG_FILE="$CONFIG_DIR/stage_${STAGE_NAME}.json"
    
    # Check if config file exists
    if [ ! -f "$CONFIG_FILE" ]; then
        echo "Warning: Config file not found: $CONFIG_FILE"
        return 1
    fi
    
    echo "================================================"
    echo "Running experiment with stage: $STAGE_NAME"
    echo "Using config: $CONFIG_FILE"
    echo "================================================"
    
    # Submit the job
    STAGE=$STAGE_NAME sbatch deepspeed.sh
    
    # Get the job ID from the last submitted job
    sleep 1
    LATEST_JOB=$(squeue -u $USER --format="%.18i %.9P %.8j %.8u %.2t %.10M %.6D %R" -S -i | grep lsai | head -n 1 | awk '{print $1}')
    
    if [ -n "$LATEST_JOB" ]; then
        echo ""
        echo "Job submitted: $LATEST_JOB"
        echo "Monitor logs with: tail -f logs/deepspeed/${LATEST_JOB}.out"
        echo "Or use the symlink: tail -f logs/deepspeed/stage_${STAGE_NAME}_latest.out"
        echo ""
    fi
}

# Main logic
if [ $# -eq 0 ]; then
    echo "No stages specified. Running all available configs..."
    echo ""
    STAGES=($(get_all_stages))
    
    if [ ${#STAGES[@]} -eq 0 ]; then
        echo "Error: No config files found in $CONFIG_DIR"
        exit 1
    fi
    
    echo "Found ${#STAGES[@]} configs:"
    printf '  - %s\n' "${STAGES[@]}"
    echo ""
    read -p "Submit all experiments? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Cancelled."
        exit 0
    fi
    echo ""
    
    for stage in "${STAGES[@]}"; do
        submit_experiment "$stage"
        sleep 0.1  # Brief delay between submissions
    done
else
    # Run specified stages
    for stage in "$@"; do
        submit_experiment "$stage"
        sleep 2  # Brief delay between submissions
    done
fi

echo "================================================"
echo "All experiments submitted!"
echo "================================================"
