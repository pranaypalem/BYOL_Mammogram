#!/bin/bash

#SBATCH -N 1                    # number of nodes
#SBATCH -c 16                   # number of cores 
#SBATCH -t 6-00:00:00           # time in d-hh:mm:ss (6 days)
#SBATCH -p general              # partition 
#SBATCH -q public               # QOS
#SBATCH --gres=gpu:a100:1       # request A100 GPU
#SBATCH --mem=128G              # memory
#SBATCH -o slurm.%j.out         # file to save job's STDOUT (%j = JobId)
#SBATCH -e slurm.%j.err         # file to save job's STDERR (%j = JobId)
#SBATCH --mail-type=ALL         # Send an e-mail when a job starts, stops, or fails
#SBATCH --mail-user="%u@asu.edu"
#SBATCH --export=NONE           # Purge the job-submitting shell environment

# Load required software
module load mamba/latest

# Activate our environment
source activate BYOL_Pranay

# Login to wandb
export WANDB_API_KEY=b7934b8839e5fb629070ea9fe209d8324d82d293
wandb login

# Change to the directory of our script
cd /home/pranaypalem/Documents/Breast_Cancer_Testing/Project/BYOL_Test

# Run the software/python script
python train_byol_mammo.py

exit