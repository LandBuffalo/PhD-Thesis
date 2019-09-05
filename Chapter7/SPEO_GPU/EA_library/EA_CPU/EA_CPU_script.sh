#!/bin/bash
module load gcc/5.5.0
g++ -c -I../../include EA_CPU.cc
g++ -c -I../../include DE_CPU.cc
g++ -c -I../../include CEC2014.cc
ar cru libEA_CPU.a *.o
mv -f libEA_CPU.a ../../
rm -f *.o