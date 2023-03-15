#!/bin/bash

#SBATCH --job-name=dominoes-sweep
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:4
#SBATCH --time=00:10:00
#SBATCH --mem=8G

# Load the necessary modules
module purge
module load python/3.8.0 cuda/11.1.1 wandb

USER=$(whoami)
SCRATCH_DIR=/scratch/$USER/$SLURM_JOBID

mkdir -p $SCRATCH_DIR

# Specify the GPU devices to use
GPU_DEVICES="0 1 2 3"

# Split the devices into an array
IFS=' ' read -r -a GPU_DEVICES_ARRAY <<< "$GPU_DEVICES"

# Launch the WandB agents with the specified GPU devices
for ((i=0; i<${#GPU_DEVICES_ARRAY[@]}; i++)); do
  CUDA_VISIBLE_DEVICES=${GPU_DEVICES_ARRAY[$i]} wandb agent SWEEP_ID --count 1 --parallel &
done

# Wait for all jobs to finish
wait
