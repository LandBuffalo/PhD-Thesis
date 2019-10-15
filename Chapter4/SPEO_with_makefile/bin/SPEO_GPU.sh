#!/bin/bash
#SBATCH --ntasks-per-node=4
#SBATCH --ntasks-per-socket=2
#SBATCH --gres=gpu:2
#SBATCH --mem-per-cpu=5000
#SBATCH --time=20:00:00
#SBATCH --partition=skylake

module load gcc/5.5.0
module load openmpi/3.0.0
module load cuda/9.0.176

srun ./SPEO_GPU -dim 100