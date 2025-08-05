#!/bin/bash

#SBATCH --job-name=byol_training
#SBATCH --account=grp_ataxr
#SBATCH --partition=general
#SBATCH --qos=public
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G
#SBATCH --time=3-00:00:00
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=%u@asu.edu
#SBATCH --export=NONE

# Load required software
module load mamba/latest

# Activate our environment
source activate BYOL_Pranay

# Login to wandb
export WANDB_API_KEY=b7934b8839e5fb629070ea9fe209d8324d82d293
wandb login

# Change to the directory of our script
cd /scratch/ppalem1/BYOL_Mammogram

# Run the software/python script
python train_byol_mammo.py

conda deactivate

exit 0