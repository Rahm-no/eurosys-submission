#!/bin/bash
#SBATCH --job-name=pytorch            # Job name
#SBATCH --output=pref12.log   # Standard output log
#SBATCH --error=error_%j.log          # Error log file
#SBATCH --nodes=1                     # Number of nodes
#SBATCH --partition=normal-a100-40      # Partition name
#SBATCH --time=48:00:00                # Time limit (hh:mm:ss)
#SBATCH --cpus-per-task=100           # Number of CPU cores
#SBATCH --account=i20240005g          # Account to charge

# Load required modules
module load Python

# # Run the monitoring script in the background & save output to a file
# bash gpu_script.sh > monitor_output_pytorchb16_losstest.log 2>&1 & 
# MONITOR_PID=$!   # Save the process ID of the monitoring script

# Run the script inside the Singularity container
singularity exec --nv \
    --bind /projects:/projects \
    --bind /projects/I20240005/coco:/projects/I20240005/coco \
    /projects/I20240005/rnouaj/object_detection_speedy/singularity_object_detection.sif \
    bash -c "./run_and_time.sh"

# # Stop the monitoring script after training finishes
# kill $MONITOR_PID
