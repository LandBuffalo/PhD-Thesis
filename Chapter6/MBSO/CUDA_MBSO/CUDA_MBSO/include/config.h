#ifndef __CONFIG_H__
#define __CONFIG_H__

#include <algorithm>  
#include <stdio.h>
#include "cuda_runtime.h"
#include "../include/error.h"

#include <curand_kernel.h>
//#include "device_launch_parameters.h"
//#include <cuda_runtime_api.h>
//#include "device_launch_parameters.h"
//#include <cuda.h>
//#include <device_functions.h>

#define MAX_DIM 100
#define TILE_WIDTH 16

#ifndef M_PI
#define M_PI 3.1415926535897932384626433832795029
#endif

#ifndef M_E
#define M_E 2.7182818284590452353602874713526625
#endif
#define MAX_CLUSTER 20

typedef unsigned int natural;

#define RECORD_OUT
#define HOST_RAND
//#define DEVICE_RAND
//#define	DEBUG
//#define IMPORT_RAND
#define KMEANS
#define GPU_KMEANS
//#define CPU_KMEANS
#define MAX_REGISTERS 32
//__constant__ real d_M[MAX_DIM * MAX_DIM * MAX_FUNC_COMPOSITION];

#endif

