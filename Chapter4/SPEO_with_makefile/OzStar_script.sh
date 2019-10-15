#!/bin/bash
module load cuda/9.0.176
module load gcc/5.5.0
module load openmpi/3.0.0
cd ./EA_library/EA_CPU
bash EA_CPU_script.sh
cd ../EA_CUDA
bash EA_CUDA_script.sh
cd ../../
pwd
mv libEA_CPU.a ./src
mv libEA_CUDA.a ./src
make clean
make
cd ./bin
