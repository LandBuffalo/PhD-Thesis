#!/bin/bash
rm -f core*
sbatch --ntasks=128 SPEO_CPU.sh
sbatch --ntasks=64 SPEO_GPU.sh
