#!/bin/sh

#SBATCH --job-name=train_49
#SBATCH --out=train_49.%j.out
#SBATCH --partition=broadwl
#SBATCH --time=4:00:00
#SBATCH --ntasks-per-node=1 # num cores to drive each gpu 
#SBATCH --exclusive

# LOAD MODULES
module load python 

# DO COMPUTE WORK
/project2/andrewferguson/Kirill/env_kirills/bin/python -u run_optim.py Experiments/ManyTrials/trial_49/params.yaml
