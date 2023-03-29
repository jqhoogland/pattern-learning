#!/bin/bash

#SBATCH --job-name=dominoes-sweep
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:4
#SBATCH --time=00:10:00
#SBATCH --mem=8G

# Required env variables: 
# - WANDB_API_KEY

# Load the necessary modules
module purge
module load python/3.9.0 cuda/11.1.1

# Create a virtual environment and activate it
python -m venv .venv
source .venv/bin/activate

# Install the required packages
pip install -r requirements.txt

# Log in to WandB
wandb login $WANDB_API_KEY

# Create a scratch directory
USER=$(whoami)
SCRATCH_DIR=/scratch/$USER/$SLURM_JOBID

mkdir -p $SCRATCH_DIR

# Specify the GPU devices to use
GPU_DEVICES="0 1 2 3"

# Split the devices into an array
IFS=' ' read -r -a GPU_DEVICES_ARRAY <<< "$GPU_DEVICES"

# Launch the WandB agents with the specified GPU devices
for ((i=0; i<${#GPU_DEVICES_ARRAY[@]}; i++)); do
  srun --exclusive -n1 -N1 -G1 -c2 CUDA_VISIBLE_DEVICES=${GPU_DEVICES_ARRAY[$i]} python dominoes/sweep.py &
done

# Wait for all jobs to finish
wait

# Deactivate the virtual environment
deactivate