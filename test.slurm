#!/bin/bash
#SBATCH --job-name=bagnet        # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=12G         # memory per cpu-core (4G per cpu-core is default)
#SBATCH --gres=gpu:4             # number of gpus per node
#SBATCH --time=05:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=all          # send email on job start, end and fail
#SBATCH --mail-user=nsum@princeton.edu

module purge
module load anaconda3
conda activate thesis

#srun python train_cifar.py 
# srun python test_defense_cifar.py --patch_size=15
# srun python patch_attack_cifar.py --patch_size=15
srun python patch_attack_cifar_dual.py --patch_size=40