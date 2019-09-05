#include "../include/config.h"
#include "device_launch_parameters.h"
__global__ void global_setupRandomState(curandState * states, natural seed)
{
	int var_random = blockDim.x * blockDim.y * (blockIdx.x + blockIdx.y * gridDim.x) + threadIdx.x + threadIdx.y * blockDim.x;
	curand_init(seed, var_random, 0, &states[var_random]);
}
extern "C"
void API_setupRandomState(dim3 blocks, dim3 threads, curandState * states, natural seed)
{
	global_setupRandomState << <blocks, threads >> >(states, seed);
}


