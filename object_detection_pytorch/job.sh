#!/usr/bin/env bash
#SBATCH --job-name=object-pytorch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --output=object_pytorch.txt
#SBATCH --error=object_pytorch.err
#SBATCH --gpus=4
#SBATCH --account=i20240005g
#SBATCH --time=2:00:00
#SBATCH --partition=normal-a100-40



# Load required modules

# # Run the monitoring script in the background & save output to a file
# bash gpu_script.sh > monitor_output_pytorchb16_losstest.log 2>&1 & 
# MONITOR_PID=$!   # Save the process ID of the monitoring script

# Run the script inside the Singularity container
singularity exec --nv \
    --bind /projects:/projects \
    --bind /projects/I20240005/coco:/projects/I20240005/coco \
    /projects/I20240005/rnouaj/object_detection_speedy/singularity_object_detection.sif \
    bash -c "/projects/I20240005/rnouaj/recovered_from_git/Results_workloads_deucalion/object_detection_speedy/run_and_time.sh"

# # Stop the monitoring script after training finishes
# kill $MONITOR_PID
