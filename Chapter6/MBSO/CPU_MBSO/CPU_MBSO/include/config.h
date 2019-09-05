#ifndef __CONFIG_H__
#define __CONFIG_H__

#include <algorithm>  
#include <stdio.h>
#include "../include/error.h"
//#include "device_launch_parameters.h"
//#include <cuda_runtime_api.h>
//#include "device_launch_parameters.h"
//#include <cuda.h>
//#include <device_functions.h>

#define HOSTCURAND

typedef unsigned int natural;

#define RECORD_OUT
//#define	DEBUG
//#define IMPORT_RAND
#define KMEANS
//__constant__ real d_M[MAX_DIM * MAX_DIM * MAX_FUNC_COMPOSITION];
#define CURAND_UNIFORM_REAL(x) curand_uniform_double(x)

#endif

