#!/bin/bash
#
#SBATCH --mail-type=FAIL
#SBATCH --ntasks=1

# Put your job commands after this line
srun hostname
srun ./test.sh ${1} ${2} ${3} ${4} ${5}

