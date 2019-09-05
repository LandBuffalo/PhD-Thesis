# CPU-MBSO

This code is the implementation of sequential MBSO based on a single CPU core (Chapter 6 of Chen Jin's thesis).

## Getting Started

This code works on PC. As it supports generate random numbers using CURAND library, at least a GPU is necessary. The computing part is implemented based on C/C++ and the random number can be generated using CURAND APIs from CUDA. The GPU will not attend any computing jobs in this implementation

### Prerequisites

You need to install below software before you build and execute the code. 

```
Windows 10
Visual Studio 2015 Community
CUDA:8.0
```
The above versions are tested. Other versions of above software may also work on our code, but it is not guaranteed

## Running the code
If want to rebuild the code, please use x64 release mode and add the curand.lib into vs project properties. Then use F7 to build the program

Before running the code, enter the directories ```./x64/Release/``` fold. Then run ```CPU_MBSO.exe -f input.txt -d [x] -func [y] -id [z] -ip [m] -r [n] -c [o]```. The explanation of each argv are as follows:
```
input.txt contains the non-important parameters setting including random seed, maximal FEs and so on.
[x] is the index of GPU device, if you only have one GPU, x=0, otherwise, set the index based on your PC.
[y] is the function index from 1 to 30 based on the CEC2014 benchmark
[z] is the dimension of problem, you can use 10, 30, 50 and 100
[m] is the population size of MBSO
[n] is the instance index which is larger than 0
[o] is the number of centers which is a crucial parameters of MBSO
```
The shift, rotation and shuffle data is located at .input/ fold. It is provide by CEC2014 official website

## Output
The output is at the current fold named ```dim=[z]_pop=[m].out```. The first column is the function ID, the second column is the instance ID, the third column is the optimisation accuracy and the last column is the computing time.

## Authors

* **Chen Jin** - *Initial work*

## License

This project is licensed under the GPL License - see the [LICENSE.md]file for details

## Acknowledgments
