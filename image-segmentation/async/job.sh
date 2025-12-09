#!/usr/bin/env bash
#SBATCH --job-name=imseg-speedy
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --output=imseg_speedy.txt
#SBATCH --error=imseg_speedy.err
#SBATCH --gpus=4
#SBATCH --account=i20240005g
#SBATCH --time=2:00:00
#SBATCH --partition=normal-a100-40
# Load required modules
module load Python
echo "Current working directory: $(pwd)"

# Run the monitoring script in the background & save output to a file
bash gpu_script.sh > gpu_usage.log 2>&1 & 
MONITOR_PID=$!   # Save the process ID of the monitoring script

# Run the script inside the Singularity container

singularity exec --nv \
    --bind /projects:/projects \
    --bind /projects/I20240005/rnouaj/image-segmentation/async/ckpts:/projects/I20240005/rnouaj/image-segmentation/async/ckpts \
    --bind /projects/I20240005/rnouaj/image-segmentation/async/output:/projects/I20240005/rnouaj/image-segmentation/async/results \
    --bind /projects/I20240005/raw-data-dir/kits19/data:/projects/I20240005/raw-data-dir/kits19/data \
    /projects/I20240005/rnouaj/singularity_images/singularity_async.sif \
    bash run_and_time.sh \


# Stop the monitoring script after training finishes
kill $MONITOR_PID
