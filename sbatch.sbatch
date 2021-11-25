#!/bin/sh
#SBATCH --job-name=/sep_dist=250-max_sep_fac=4-x_0=fixed-w_reg=2500-trial=9
#SBATCH --output=Experiments/Trials_sep_params/sep_dist=250-max_sep_fac=4-x_0=fixed-w_reg=2500-trial=9//sep_dist=250-max_sep_fac=4-x_0=fixed-w_reg=2500-trial=9.%j.out
#SBATCH --partition=caslake
#SBATCH --account=pi-andrewferguson
#SBATCH --mail-type=ALL
#SBATCH --time=24:00:00
#SBATCH --ntasks-per-node=1  # number of tasks
#SBATCH --cpus-per-task=16    # number of threads per task


module load python
conda activate /project/andrewferguson/Kirill/env_kirill

/project/andrewferguson/Kirill/env_kirill/bin/python -u run_optim.py Experiments/Trials_sep_params/sep_dist=250-max_sep_fac=4-x_0=fixed-w_reg=2500-trial=9/params.yaml