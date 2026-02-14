#!/bin/bash

# Create bash directory if it doesn't exist
mkdir -p bash

# Set log file paths
LOG_FILE="bash/pipeline_execution.log"
ERROR_LOG="bash/pipeline_errors.log"

# Clear previous logs
> "$LOG_FILE"
> "$ERROR_LOG"

# Record start time
echo "Pipeline execution started at $(date)" | tee -a "$LOG_FILE"

# Define script list
SCRIPTS=(
    "P1_data_statistics_traj_cleaning/p1_raw_data_inspection.py"
    "P1_data_statistics_traj_cleaning/p2_traj_cleaning.py"
    "P1_data_statistics_traj_cleaning/p3_cleaned_data_inspection.py"
)

# Run each script
for script in "${SCRIPTS[@]}"; do
    echo -e "
=== Running $script ===" | tee -a "$LOG_FILE"
    
    # Check if file exists
    if [ ! -f "$script" ]; then
        echo "Error: Script $script not found!" | tee -a "$ERROR_LOG"
        continue
    fi
    
    # Record start time
    start_time=$(date +%s)
    start_datetime=$(date)
    
    # Run Python script and wait for completion
    python "$script" 2>> "$ERROR_LOG" && wait
    
    if [ $? -eq 0 ]; then
        # Calculate execution time
        end_time=$(date +%s)
        end_datetime=$(date)
        duration=$((end_time - start_time))
        
        # Convert duration to readable format
        hours=$((duration / 3600))
        minutes=$(( (duration % 3600) / 60 ))
        seconds=$((duration % 60))
        
        echo "✓ Successfully completed $script" | tee -a "$LOG_FILE"
        echo "Started at: $start_datetime" | tee -a "$LOG_FILE"
        echo "Finished at: $end_datetime" | tee -a "$LOG_FILE"
        echo "Duration: ${hours}h ${minutes}m ${seconds}s" | tee -a "$LOG_FILE"
    else
        echo "✗ Failed to execute $script" | tee -a "$LOG_FILE"
        echo "Check $ERROR_LOG for details"
        
        read -p "Continue with next script? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Pipeline execution stopped by user" | tee -a "$LOG_FILE"
            exit 1
        fi
    fi
done

echo -e "
Pipeline execution completed at $(date)" | tee -a "$LOG_FILE"
