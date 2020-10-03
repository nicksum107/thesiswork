#!/bin/bash
#
#SBATCH --mail-type=FAIL
#SBATCH --ntasks=1

# Put your job commands after this line
srun hostname
srun ./ct_hyperparameter.sh ${1} ${2} ${3} ${4} ${5} ${6} ${7}
