#!/bin/bash
#SBATCH --job-name=imseg_pref6              # Job name
#SBATCH --output=output_imseg.log     # Standard output log
#SBATCH --error=error_%j.log          # Error log file
#SBATCH --nodes=1                     # Number of nodes
#SBATCH --partition=dev-a100-40       # Partition name
#SBATCH --time=4:00:00                # Time limit (hh:mm:ss)
#SBATCH --cpus-per-task=100           # Number of CPU cores
#SBATCH --account=i20240005g          # Account to charge

# Load required modules
module load Python
echo "Current working directory: $(pwd)"

# Run the monitoring script in the background & save output to a file
bash gpu_script.sh > /projects/I20240005/rnouaj/image-segmentation/imseg/GPUusage_16worker.csv 2>&1 & 
MONITOR_PID=$!   # Save the process ID of the monitoring script

# Run the script inside the Singularity container

singularity exec --nv \
    --bind /projects:/projects \
    --bind /projects/I20240005/rnouaj/image-segmentation/imseg/ckpts:/projects/I20240005/rnouaj/image-segmentation/imseg/ckpts \
    --bind /projects/I20240005/rnouaj/image-segmentation/imseg/output:/projects/I20240005/rnouaj/image-segmentation/imseg/results \
    --bind /projects/I20240005/raw-data-dir/kits19/data:/projects/I20240005/raw-data-dir/kits19/data \
    /projects/I20240005/rnouaj/singularity_images/singularity_imseg.sif \
    bash run_and_time.sh \


# Stop the monitoring script after training finishes
kill $MONITOR_PID
