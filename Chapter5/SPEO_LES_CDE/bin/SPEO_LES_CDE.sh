#!/bin/bash
#SBATCH --ntasks=64
#SBATCH --mem-per-cpu=5000
#SBATCH --time=24:00:00
#SBATCH --partition=skylake

module load gcc/5.5.0
module load openmpi/3.0.0

srun ./SPEO_LES_CDE -dim 10 -total_functions 23-30 -total_runs 1-15 -max_base_FEs 10000 -interval 100 -connection_rate 0.25 -buffer_capacity 1 -global_pop_size 8192 -k_nearest 5