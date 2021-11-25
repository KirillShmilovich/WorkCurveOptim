#!/bin/sh
#SBATCH --job-name=trial_49
#SBATCH --output=Experiments/ManyTrials/trial_49/trial_49.%j.out
#SBATCH --partition=caslake
#SBATCH --account=pi-andrewferguson
#SBATCH --mail-type=ALL
#SBATCH --time=8:00:00
#SBATCH --ntasks-per-node=1  # number of tasks
#SBATCH --cpus-per-task=16    # number of threads per task


module load python
conda activate /project/andrewferguson/Kirill/env_kirill

/project/andrewferguson/Kirill/env_kirill/bin/python -u run_optim.py Experiments/ManyTrials/trial_49/params.yaml