#ifndef __EVAL_HH__
#define __EVAL_HH__

#include "../include/config.h"
#include <stdio.h>
#include"../include/rotation.h"


struct ConfigEval
{
	int i_func;
	int numfunc;
	int sigma[5];
	real *d_w;
	real *d_sumW;
	real *d_M;
	int *d_shuffle;
	int isshifted[5];
	int isrotated[5];
	real rate;
	real fixedshift;
	bool compositionFlag;
	real bias;
};

void calConfigEval(ConfigEval *funcconfig, ConfigDE config)
{
	funcconfig->d_M = NULL;
	funcconfig->d_shuffle = NULL;

	funcconfig->d_w = NULL;
	HANDLE_CUDA_ERROR(cudaMalloc(&funcconfig->d_w, sizeof(real) * config.i_popsize));
	funcconfig->d_sumW = NULL;
	HANDLE_CUDA_ERROR(cudaMalloc(&funcconfig->d_sumW, sizeof(real) * config.i_popsize));

	funcconfig->compositionFlag = true;
	if (config.i_func < 23)
	{
		funcconfig->numfunc = 1;
		funcconfig->compositionFlag = false;
		funcconfig->isrotated[0] = 1;
		funcconfig->isshifted[0] = 1;
		funcconfig->fixedshift = 0;
		funcconfig->rate = 1;
		switch (funcconfig->i_func)
		{
		case(4) :
			funcconfig->rate = 2.048 / (double)100;
			funcconfig->fixedshift = 1;
			break;
		case(6) :
			funcconfig->rate = 0.5 / (double)100;
			break;
		case(7) :
			funcconfig->rate = 6;
			break;
		case(8) :
			funcconfig->isrotated[0] = 0;
			funcconfig->rate = 5.12 / (double)100;
			break;
		case(9) :
			funcconfig->rate = 5.12 / (double)100;
			break;
		case(10) :
			funcconfig->isrotated[0] = 0;
			funcconfig->rate = 10;
			break;
		case(11) :
			funcconfig->rate = 10;
			break;
		case(12) :
			funcconfig->rate = 5 / (double)100;
			break;
		case(13) :
			funcconfig->rate = 5 / (double)100;
			break;
		case(14) :
			funcconfig->rate = 5 / (double)100;
			break;
		case(15) :
			funcconfig->rate = 5 / (double)100;
			funcconfig->fixedshift = 1;
			break;
		case(16) :
			funcconfig->fixedshift = 1;
			break;
		default:
			break;
		}
	}
	else
	{
		switch (config.i_func)
		{
		case(23) :
			funcconfig->numfunc = 5;
			funcconfig->isshifted[0] = funcconfig->isshifted[1] = funcconfig->isshifted[2] = funcconfig->isshifted[3] = funcconfig->isshifted[4] = 1;
			funcconfig->isrotated[0] = funcconfig->isrotated[2] = funcconfig->isrotated[3] = 1;
			funcconfig->isrotated[1] = funcconfig->isrotated[4] = 0;
			funcconfig->sigma[0] = 10;
			funcconfig->sigma[1] = 20;
			funcconfig->sigma[2] = 30;
			funcconfig->sigma[3] = 40;
			funcconfig->sigma[4] = 50;
			funcconfig->rate = 1;
			funcconfig->fixedshift = 0;
			funcconfig->bias = 2300;
			break;
		case(24) :
			funcconfig->numfunc = 3;
			funcconfig->isshifted[0] = funcconfig->isshifted[1] = funcconfig->isshifted[2] = 1;
			funcconfig->isrotated[1] = funcconfig->isrotated[2] = 1;
			funcconfig->isrotated[0] = 0;
			funcconfig->sigma[0] = 20;
			funcconfig->sigma[1] = 20;
			funcconfig->sigma[2] = 20;
			funcconfig->rate = 1;
			funcconfig->fixedshift = 0;
			funcconfig->bias = 2400;
			break;
		case(25) :
			funcconfig->numfunc = 3;
			funcconfig->isshifted[0] = funcconfig->isshifted[1] = funcconfig->isshifted[2] = 1;
			funcconfig->isrotated[0] = funcconfig->isrotated[1] = funcconfig->isrotated[2] = 1;
			funcconfig->sigma[0] = 10;
			funcconfig->sigma[1] = 30;
			funcconfig->sigma[2] = 50;
			funcconfig->rate = 1;
			funcconfig->fixedshift = 0;
			funcconfig->bias = 2500;
			break;
		case(26) :
			funcconfig->numfunc = 5;
			funcconfig->isshifted[0] = funcconfig->isshifted[1] = funcconfig->isshifted[2] = funcconfig->isshifted[3] = funcconfig->isshifted[4] = 1;
			funcconfig->isrotated[0] = funcconfig->isrotated[1] = funcconfig->isrotated[2] = funcconfig->isrotated[3] = funcconfig->isrotated[4] = 1;
			funcconfig->sigma[0] = 10;
			funcconfig->sigma[1] = 10;
			funcconfig->sigma[2] = 10;
			funcconfig->sigma[3] = 10;
			funcconfig->sigma[4] = 10;
			funcconfig->rate = 1;
			funcconfig->fixedshift = 0;
			funcconfig->bias = 2600;
			break;
		case(27) :
			funcconfig->numfunc = 5;
			funcconfig->isshifted[0] = funcconfig->isshifted[1] = funcconfig->isshifted[2] = funcconfig->isshifted[3] = funcconfig->isshifted[4] = 1;
			funcconfig->isrotated[0] = funcconfig->isrotated[1] = funcconfig->isrotated[2] = funcconfig->isrotated[3] = funcconfig->isrotated[4] = 1;
			funcconfig->sigma[0] = 10;
			funcconfig->sigma[1] = 10;
			funcconfig->sigma[2] = 10;
			funcconfig->sigma[3] = 20;
			funcconfig->sigma[4] = 20;
			funcconfig->rate = 1;
			funcconfig->fixedshift = 0;
			funcconfig->bias = 2700;
			break;
		case(28) :
			funcconfig->numfunc = 5;
			funcconfig->isshifted[0] = funcconfig->isshifted[1] = funcconfig->isshifted[2] = funcconfig->isshifted[3] = funcconfig->isshifted[4] = 1;
			funcconfig->isrotated[0] = funcconfig->isrotated[1] = funcconfig->isrotated[2] = funcconfig->isrotated[3] = funcconfig->isrotated[4] = 1;
			funcconfig->sigma[0] = 10;
			funcconfig->sigma[1] = 20;
			funcconfig->sigma[2] = 30;
			funcconfig->sigma[3] = 40;
			funcconfig->sigma[4] = 50;
			funcconfig->rate = 1;
			funcconfig->fixedshift = 0;
			funcconfig->bias = 2800;
			break;
		case(29) :
			funcconfig->numfunc = 3;
			funcconfig->isshifted[0] = funcconfig->isshifted[1] = funcconfig->isshifted[2] = 1;
			funcconfig->isrotated[0] = funcconfig->isrotated[1] = funcconfig->isrotated[2] = 1;
			funcconfig->sigma[0] = 10;
			funcconfig->sigma[1] = 30;
			funcconfig->sigma[2] = 50;
			funcconfig->rate = 1;
			funcconfig->fixedshift = 0;
			funcconfig->bias = 2900;
			break;
		case(30) :
			funcconfig->numfunc = 3;
			funcconfig->isshifted[0] = funcconfig->isshifted[1] = funcconfig->isshifted[2] = 1;
			funcconfig->isrotated[0] = funcconfig->isrotated[1] = funcconfig->isrotated[2] = 1;
			funcconfig->sigma[0] = 10;
			funcconfig->sigma[1] = 30;
			funcconfig->sigma[2] = 50;
			funcconfig->rate = 1;
			funcconfig->fixedshift = 0;
			funcconfig->bias = 3000;
			break;
		default:
			break;
		}
	}
	HANDLE_CUDA_ERROR(cudaMalloc(&funcconfig->d_M, sizeof(real) * config.i_dim * config.i_dim * funcconfig->numfunc));
	HANDLE_CUDA_ERROR(cudaMalloc(&funcconfig->d_shuffle, sizeof(int) * config.i_dim * MAX_FUNC_COMPOSITION));
};

static __device__ __forceinline__  void parallelsum(real * vector, real* result, int lengthSum) {
	//blockDim.y is the dimension of problem
	int olds = lengthSum;
	//maybe can be improved--------------------------------------//
	// if the olds can be divided by 2, use paralle sum
	for (int s = lengthSum / 2; olds == s * 2; s >>= 1) {
		olds = s;
		//~ if (blockIdx.x == 0 && threadIdx.x ==0 ) printf("T %d S %d OLDS %d\n", threadIdx.y, s, olds);
		//sum the two elements(index and index + s)
		if (threadIdx.y < s) vector[threadIdx.x + blockDim.x * threadIdx.y] += vector[threadIdx.x + blockDim.x * (threadIdx.y + s)];
		__syncthreads();

	}
	// if the olds can  not be divided by 2, use sequentially sum from threadIdx.y = 0
	if (threadIdx.y == 0) {
		real sum = vector[threadIdx.x];
		for (int i = 1; i < olds; i++) {
			//~ if (blockIdx.x == 0 && threadIdx.x ==0) printf("T %d I %d OLDS %d V %f\n", threadIdx.y, i, olds, vector[threadIdx.x + blockDim.x * i]);
			sum += vector[threadIdx.x + blockDim.x * i];
		}
		*result = sum;
	}

	__syncthreads();

};

static __device__ __forceinline__  void parallelmultiple(real * vector, real* result, int lengthSum) {
	//blockDim.y is the dimension of problem
	int olds = lengthSum;
	//maybe can be improved--------------------------------------//
	// if the olds can be divided by 2, use paralle sum
	for (int s = lengthSum / 2; olds == s * 2; s >>= 1) {
		olds = s;
		//~ if (blockIdx.x == 0 && threadIdx.x ==0 ) printf("T %d S %d OLDS %d\n", threadIdx.y, s, olds);
		//sum the two elements(index and index + s)
		if (threadIdx.y < s) vector[threadIdx.x + blockDim.x * threadIdx.y] *= vector[threadIdx.x + blockDim.x * (threadIdx.y + s)];
		__syncthreads();

	}
	// if the olds can  not be divided by 2, use sequentially sum from threadIdx.y = 0
	if (threadIdx.y == 0) {
		real sum = vector[threadIdx.x];
		for (int i = 1; i < olds; i++) {
			//~ if (blockIdx.x == 0 && threadIdx.x ==0) printf("T %d I %d OLDS %d V %f\n", threadIdx.y, i, olds, vector[threadIdx.x + blockDim.x * i]);
			sum *= vector[threadIdx.x + blockDim.x * i];
		}
		*result = sum;
	}

	__syncthreads();

};

__device__ void f1(real* fitnessValue, real * pop, size_t size, int ind, real* result, real* local, real bias, real *wi, int *d_shuffle, int startDim, int lengthDim)
{
	real var;
	int shuffleInd;
	if (threadIdx.y < lengthDim)
	{
		if (startDim == -1)
			shuffleInd = threadIdx.y;
		else
			shuffleInd = d_shuffle[threadIdx.y + startDim] - 1;
		var = pop[ind + shuffleInd * size];
	}
	else
	{
		var = 0;
	}
	local[threadIdx.x + blockDim.x * threadIdx.y] = pow((double)1000000.0, threadIdx.y / (double)(lengthDim - 1)) * var * var;
	__syncthreads();

	parallelsum(local, &result[threadIdx.x], blockDim.y);
	if (threadIdx.y == 0)
		fitnessValue[ind] += wi[threadIdx.x] * result[threadIdx.x] + bias;
};

__device__ void f2(real* fitnessValue, real * pop, size_t size, int ind, real* result, real* local, real bias, real *wi, int *d_shuffle, int startDim, int lengthDim)
{
	real var, unit1;
	int shuffleInd;
	if (threadIdx.y < lengthDim)
	{
		if (startDim == -1)
			shuffleInd = threadIdx.y;
		else
			shuffleInd = d_shuffle[threadIdx.y + startDim] - 1;
		var = pop[ind + shuffleInd * size];
		if (threadIdx.y == 0)
		{
			unit1 = var;
			var = 0;
		}
	}
	else
	{
		var = 0;
	}
	local[threadIdx.x + blockDim.x * threadIdx.y] = var * var;

	__syncthreads();
	parallelsum(local, &result[threadIdx.x], blockDim.y);

	if (threadIdx.y == 0)
		fitnessValue[ind] += wi[threadIdx.x] * (1e6 * result[threadIdx.x] + unit1 * unit1) + bias;
};

__device__ void f3(real* fitnessValue, real * pop, size_t size, int ind, real* result, real* local, real bias, real *wi, int *d_shuffle, int startDim, int lengthDim)
{
	real var, unit1;
	int shuffleInd;
	if (threadIdx.y < lengthDim)
	{
		if (startDim == -1)
			shuffleInd = threadIdx.y;
		else
			shuffleInd = d_shuffle[threadIdx.y + startDim] - 1;
		var = pop[ind + shuffleInd * size];
		if (threadIdx.y == 0)
			unit1 = var;
	}
	else
	{
		var = 0;
	}
	local[threadIdx.x + blockDim.x * threadIdx.y] = var * var;
	//	printf("%d\t%f\n", threadIdx.y, local[threadIdx.x + blockDim.x * threadIdx.y]);
	__syncthreads();
	parallelsum(local, &result[threadIdx.x], lengthDim);
	if (threadIdx.y == 0)
		fitnessValue[ind] += wi[threadIdx.x] * (result[threadIdx.x] + (1e6 - 1) * unit1 * unit1) + bias;
};

__device__ void f4(real* fitnessValue, real * pop, size_t size, int ind, real* result, real* local, real bias, real *wi, int *d_shuffle, int startDim, int lengthDim)
{
	real var1 = 0, var2 = 0, unit1 = 0, unit2 = 0, unit3 = 0;
	int shuffleInd1 = 0, shuffleInd2 = 0;
	if (threadIdx.y < lengthDim - 1)
	{
		if (startDim == -1)
		{
			shuffleInd1 = threadIdx.y;
			shuffleInd2 = threadIdx.y + 1;
		}
		else
		{
			shuffleInd1 = d_shuffle[threadIdx.y + startDim] - 1;
			shuffleInd2 = d_shuffle[threadIdx.y + 1 + startDim] - 1;
		}
		var1 = pop[ind + shuffleInd1 * size];
		var2 = pop[ind + shuffleInd2 * size];
		unit1 = (var1 * var1 - var2) * (var1 * var1 - var2);
		unit2 = (var1 - 1.0) * (var1 - 1.0);
		unit3 = 100.0 * unit1 + unit2;
	}
	else
	{
		unit3 = 0;
	}
	local[threadIdx.x + blockDim.x * threadIdx.y] = unit3;
	__syncthreads();
	parallelsum(local, &result[threadIdx.x], blockDim.y);
	if (threadIdx.y == 0)
		fitnessValue[ind] += wi[threadIdx.x] * result[threadIdx.x] + bias;

};

__device__ void f5(real* fitnessValue, real * pop, size_t size, int ind, real* result, real* local, real bias, real *wi, int *d_shuffle, int startDim, int lengthDim)
{
	real var, unit1 = 0, unit2 = 0, unit3, tmp;
	int shuffleInd;
	unit1 = unit2 = 0;

	if (threadIdx.y < lengthDim)
	{
		if (startDim == -1)
			shuffleInd = threadIdx.y;
		else
			shuffleInd = d_shuffle[threadIdx.y + startDim] - 1;
		var = pop[ind + shuffleInd * size];

		tmp = cos(2 * M_PI * var);
		unit1 = var * var;
	}
	else
	{
		tmp = 0;
		unit1 = 0;
	}
	local[threadIdx.x + blockDim.x * threadIdx.y] = tmp;
	local[threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y)] = unit1;

	__syncthreads();
	parallelsum(local, &result[threadIdx.x], blockDim.y);
	if (threadIdx.y == 0)
	{
		unit2 = result[threadIdx.x] / (double)lengthDim;
		unit3 = exp(unit2);
	}
	parallelsum(local + blockDim.x * blockDim.y, &result[threadIdx.x], blockDim.y);
	if (threadIdx.y == 0)
	{
		unit1 = -20 * exp(sqrt(result[threadIdx.x] / (double)lengthDim) * (-0.2));
		fitnessValue[ind] += wi[threadIdx.x] * (unit1 - unit3 + 20 + M_E) + bias;
	}
};

__device__ void f6(real* fitnessValue, real * pop, size_t size, int ind, real* result, real* local, real bias, real *wi, int *d_shuffle, int startDim, int lengthDim)
{
	real var, unit1, unit2, unit3;
	int shuffleInd;
	real a = 0.5;
	real b = 3;
	int kmax = 20;

	if (threadIdx.y < lengthDim)
	{
		if (startDim == -1)
			shuffleInd = threadIdx.y;
		else
			shuffleInd = d_shuffle[threadIdx.y + startDim] - 1;
		var = pop[ind + shuffleInd * size];
		unit2 = 0;
		unit3 = 0;
		for (int i = 0; i <= kmax; i++)
		{
			unit1 = 2 * M_PI * pow(b, (double)i);
			unit2 += pow(a, (double)i) * cos(unit1 * (var + 0.5));
			unit3 += pow(a, (double)i) * cos(unit1 * 0.5);
		}
	}
	else
	{
		unit2 = 0;
	}
	local[threadIdx.x + blockDim.x * threadIdx.y] = unit2;
	//	printf("%d\t%f\n", threadIdx.y, local[threadIdx.x + blockDim.x * threadIdx.y]);
	__syncthreads();
	parallelsum(local, &result[threadIdx.x], blockDim.y);
	if (threadIdx.y == 0)
		fitnessValue[ind] += wi[threadIdx.x] * (result[threadIdx.x] - unit3 * lengthDim) + bias;

};

__device__ void f7(real* fitnessValue, real * pop, size_t size, int ind, real* result, real* local, real bias, real *wi, int *d_shuffle, int startDim, int lengthDim)
{
	real var, unit1, unit2;
	int shuffleInd;
	if (threadIdx.y < lengthDim)
	{
		if (startDim == -1)
			shuffleInd = threadIdx.y;
		else
			shuffleInd = d_shuffle[threadIdx.y + startDim] - 1;
		var = pop[ind + shuffleInd * size];
		unit1 = var * var / (double)4000;
		unit2 = cos(var / sqrt((double)threadIdx.y + 1));

	}
	else
	{
		unit1 = 0;
		unit2 = 1;
	}
	local[threadIdx.x + blockDim.x * threadIdx.y] = unit1;
	local[threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y)] = unit2;

	__syncthreads();
	parallelsum(local, &result[threadIdx.x], blockDim.y);
	if (threadIdx.y == 0)
		unit1 = result[threadIdx.x];
	parallelmultiple(local + blockDim.x *  blockDim.y, &result[threadIdx.x], blockDim.y);


	if (threadIdx.y == 0)
	{
		unit2 = result[threadIdx.x];
		fitnessValue[ind] += wi[threadIdx.x] * (unit1 - unit2 + 1) + bias;
	}
};

__device__ void f8(real* fitnessValue, real * pop, size_t size, int ind, real* result, real* local, real bias, real *wi, int *d_shuffle, int startDim, int lengthDim)
{
	real var, unit1;
	int shuffleInd;
	if (threadIdx.y < lengthDim)
	{
		if (startDim == -1)
			shuffleInd = threadIdx.y;
		else
			shuffleInd = d_shuffle[threadIdx.y + startDim] - 1;
		var = pop[ind + shuffleInd * size];
		unit1 = var * var - 10 * cos(2 * M_PI * var) + 10;
	}
	else
	{
		unit1 = 0;
	}
	local[threadIdx.x + blockDim.x * threadIdx.y] = unit1;
	//	printf("%d\t%f\n", threadIdx.y, local[threadIdx.x + blockDim.x * threadIdx.y]);
	__syncthreads();
	parallelsum(local, &result[threadIdx.x], blockDim.y);
	if (threadIdx.y == 0)
		fitnessValue[ind] += wi[threadIdx.x] * result[threadIdx.x] + bias;
};

__device__ void f9(real* fitnessValue, real * pop, size_t size, int ind, real* result, real* local, real bias, real *wi, int *d_shuffle, int startDim, int lengthDim)
{
	real var, zvar, unit1;
	real sinValue = 0, cosValue = 0;
	int shuffleInd;
	if (threadIdx.y < lengthDim)
	{
		if (startDim == -1)
			shuffleInd = threadIdx.y;
		else
			shuffleInd = d_shuffle[threadIdx.y + startDim] - 1;

		zvar = pop[ind + shuffleInd * size] + 4.209687462275036e+002;

		if (zvar >= -500 && zvar <= 500)
		{
			sincos(sqrt(fabs(zvar)), &sinValue, &cosValue);
			unit1 = zvar * sinValue;
		}
		else if (zvar > 500)
		{
			sincos(sqrt(fabs(500 - fmod(zvar, 500.0))), &sinValue, &cosValue);
			unit1 = (500 - fmod(zvar, (double)500)) * sinValue - (zvar - 500) * (zvar - 500) / (10000.0 * lengthDim);
		}
		else if (zvar < -500)
		{
			sincos(sqrt(fabs(-500 + fmod(abs(zvar), 500.0))), &sinValue, &cosValue);

			unit1 = (-500 + fmod(abs(zvar), (double)500)) * sinValue - (zvar + 500) * (zvar + 500) / (10000.0 * lengthDim);
		}
	}
	else
	{
		unit1 = 0;
	}
	local[threadIdx.x + blockDim.x * threadIdx.y] = unit1;

	//	printf("%d\t%f\n", threadIdx.y, local[threadIdx.x + blockDim.x * threadIdx.y]);
	__syncthreads();
	parallelsum(local, &result[threadIdx.x], blockDim.y);
	if (threadIdx.y == 0)
		fitnessValue[ind] += wi[threadIdx.x] * (4.189828872724338e+002 * lengthDim - result[threadIdx.x]) + bias;
};

__device__ void f10(real* fitnessValue, real * pop, size_t size, int ind, real* result, real* local, real bias, real *wi, int *d_shuffle, int startDim, int lengthDim)
{
	real var, tmp, unit1, unit2, unit3 = 0;
	int shuffleInd;

	real tmp2 = (double)10 / pow((double)lengthDim, 1.2);
	real tmp3 = (double)10 / pow((double)lengthDim, 2);
	if (threadIdx.y < lengthDim)
	{
		if (startDim == -1)
			shuffleInd = threadIdx.y;
		else
			shuffleInd = d_shuffle[threadIdx.y + startDim] - 1;
		var = pop[ind + shuffleInd * size];
		unit3 = 0;

		for (int i = 1; i <= 32; i++)
		{
			tmp = pow((double)2, (double)i);
			unit1 = var * tmp;
			unit2 = abs(unit1 - floor(unit1 + 0.5));
			unit2 = unit2 / tmp;
			unit3 += unit2;
		}
		unit3 = pow((threadIdx.y + 1) * unit3 + 1, tmp2);
	}
	else
		unit3 = 1;
	local[threadIdx.x + blockDim.x * threadIdx.y] = unit3;

	__syncthreads();
	parallelmultiple(local, &result[threadIdx.x], blockDim.y);
	if (threadIdx.y == 0)
		fitnessValue[ind] += wi[threadIdx.x] * (tmp3 * result[threadIdx.x] - tmp3) + bias;
}

__device__ void f11(real* fitnessValue, real * pop, size_t size, int ind, real* result, real* local, real bias, real *wi, int *d_shuffle, int startDim, int lengthDim)
{
	real var, unit1, unit2, unit3;
	int shuffleInd;
	if (threadIdx.y < lengthDim)
	{
		if (startDim == -1)
			shuffleInd = threadIdx.y;
		else
			shuffleInd = d_shuffle[threadIdx.y + startDim] - 1;
		var = pop[ind + shuffleInd * size];
	}
	else
	{
		var = 0;
	}
	local[threadIdx.x + blockDim.x * threadIdx.y] = var * var;
	local[threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y)] = var;

	__syncthreads();
	parallelsum(local, &result[threadIdx.x], blockDim.y);
	if (threadIdx.y == 0)
	{
		unit1 = result[threadIdx.x];
		unit2 = pow(fabs(unit1 - lengthDim), 0.25);
	}
	parallelsum(local + blockDim.x * blockDim.y, &result[threadIdx.x], blockDim.y);
	if (threadIdx.y == 0)
	{
		unit3 = (0.5 * unit1 + result[threadIdx.x]) / (double)lengthDim;
		fitnessValue[ind] += wi[threadIdx.x] * (unit2 + unit3 + 0.5) + bias;
	}
		//	local[threadIdx.x + blockDim.x * threadIdx.y] = 0;
	//	local[threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y)] = 0;
		

}

__device__ void f12(real* fitnessValue, real * pop, size_t size, int ind, real* result, real* local, real bias, real *wi, int *d_shuffle, int startDim, int lengthDim)
{
	real var, unit1, unit2;
	int shuffleInd;
	if (threadIdx.y < lengthDim)
	{
		if (startDim == -1)
			shuffleInd = threadIdx.y;
		else
			shuffleInd = d_shuffle[threadIdx.y + startDim] - 1;
		var = pop[ind + shuffleInd * size];

	}
	else
	{
		var = 0;
	}
	local[threadIdx.x + blockDim.x * threadIdx.y] = var * var;
	local[threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y)] = var;
	__syncthreads();

	parallelsum(local, &result[threadIdx.x], blockDim.y);
	if (threadIdx.y == 0)
		unit1 = result[threadIdx.x];
	parallelsum(local + blockDim.x * blockDim.y, &result[threadIdx.x], blockDim.y);
	if (threadIdx.y == 0)
	{
		unit2 = result[threadIdx.x];
		fitnessValue[ind] += wi[threadIdx.x] * (sqrt(fabs(unit1 * unit1 - unit2 * unit2)) + (0.5 * unit1 + unit2) / (0.0 + lengthDim) + 0.5) + bias;
	}

}

__device__ void f13(real* fitnessValue, real * pop, size_t size, int ind, real* result, real* local, real bias, real *wi, int *d_shuffle, int startDim, int lengthDim)
{
	real var1, var2, unit1, unit2, unit3, unit4;
	int shuffleInd1, shuffleInd2;
	if (lengthDim >= 2)
	{
		if (threadIdx.y < lengthDim - 1)
		{
			if (startDim == -1)
			{
				shuffleInd1 = threadIdx.y;
				shuffleInd2 = threadIdx.y + 1;
			}
			else
			{
				shuffleInd1 = d_shuffle[threadIdx.y + startDim] - 1;
				shuffleInd2 = d_shuffle[threadIdx.y + 1 + startDim] - 1;
			}

			var1 = pop[ind + size * shuffleInd1];
			var2 = pop[ind + size * shuffleInd2];
			unit1 = 100 * (var1 * var1 - var2) * (var1 * var1 - var2) + (var1 - 1) * (var1 - 1);
			unit2 = unit1 * unit1 / (double)4000 - cos(unit1) + 1;
		}
		else if (threadIdx.y == lengthDim - 1)
		{
			if (startDim == -1)
			{
				var1 = pop[ind + size * (lengthDim - 1)];
				var2 = pop[ind];
			}
			else
			{
				var1 = pop[ind + size * (d_shuffle[lengthDim - 1 + startDim] - 1)];
				var2 = pop[ind + size * (d_shuffle[startDim] - 1)];
			}
			unit1 = 100 * (var1 * var1 - var2) * (var1 * var1 - var2) + (var1 - 1) * (var1 - 1);
			unit2 = unit1 * unit1 / (double)4000 - cos(unit1) + 1;

		}
		else
		{
			unit2 = 0;
		}
		local[threadIdx.x + blockDim.x * threadIdx.y] = unit2;
		__syncthreads();

		parallelsum(local, &result[threadIdx.x], blockDim.y);

		if (threadIdx.y == 0)
			fitnessValue[ind] += wi[threadIdx.x] * result[threadIdx.x] + bias;
	}
	else if (lengthDim == 1)
	{
		if (threadIdx.y == 0)
		{
			if (startDim == -1)
			{
				shuffleInd1 = threadIdx.y;
				shuffleInd2 = threadIdx.y + 1;

			}
			else
			{
				shuffleInd1 = d_shuffle[threadIdx.y + startDim] - 1;
				shuffleInd2 = d_shuffle[threadIdx.y + 1 + startDim] - 1;

			}
			var1 = pop[ind + size * shuffleInd1];
			var2 = pop[ind + size * shuffleInd2];
			unit1 = 100 * (var1 * var1 - var2) * (var1 * var1 - var2) + (var1 - 1) * (var1 - 1);
			unit2 = unit1 * unit1 / (double)4000 - cos(unit1) + 1;
			fitnessValue[ind] += wi[threadIdx.x] * unit2 + bias;
		}
	}
}

__device__ void f14(real* fitnessValue, real * pop, size_t size, int ind, real* result, real* local, real bias, real *wi, int *d_shuffle, int startDim, int lengthDim)
{
	real var1, var2, unit1, unit2, unit3 = 0;
	real sinValue = 0, cosValue = 0;
	int shuffleInd1, shuffleInd2;
	if (lengthDim != 1)
	{
		if (threadIdx.y < lengthDim - 1)
		{
			if (startDim == -1)
			{
				shuffleInd1 = threadIdx.y;
				shuffleInd2 = threadIdx.y + 1;
			}
			else
			{
				shuffleInd1 = d_shuffle[threadIdx.y + startDim] - 1;
				shuffleInd2 = d_shuffle[threadIdx.y + 1 + startDim] - 1;
			}

			var1 = pop[ind + size * shuffleInd1];
			var2 = pop[ind + size * shuffleInd2];
			unit1 = var1 * var1 + var2 * var2;
			sincos(sqrt(unit1), &sinValue, &cosValue);
			unit2 = sinValue * sinValue - 0.5;
			unit2 = unit2 / ((1 + 0.001 * unit1)*(1 + 0.001 * unit1)) + 0.5;

		}
		else if (threadIdx.y == lengthDim - 1)
		{
			if (startDim == -1)
			{
				var1 = pop[ind + size * (lengthDim - 1)];
				var2 = pop[ind];
			}
			else
			{
				var1 = pop[ind + size * (d_shuffle[lengthDim - 1 + startDim] - 1)];
				var2 = pop[ind + size * (d_shuffle[startDim] - 1)];
			}
			unit1 = var1 * var1 + var2 * var2;
			sincos(sqrt(unit1), &sinValue, &cosValue);
			unit2 = sinValue * sinValue - 0.5;
			unit2 = unit2 / ((1 + 0.001 * unit1)*(1 + 0.001 * unit1)) + 0.5;
		}
		else
		{
			unit2 = 0;
		}
		local[threadIdx.x + blockDim.x * threadIdx.y] = unit2;

		__syncthreads();

		parallelsum(local, &result[threadIdx.x], blockDim.y);

		if (threadIdx.y == 0)
			fitnessValue[ind] += wi[threadIdx.x] * result[threadIdx.x] + bias;
	}
	else
	{
		if (threadIdx.y == 0)
		{
			var1 = pop[ind + size * (d_shuffle[startDim] - 1)];
			sincos(var1, &sinValue, &cosValue);
			unit3 = var1 * var1;
			unit2 = (1.0 + 0.001 * unit3) * (1.0 + 0.001 * unit3);
			unit1 = sinValue * sinValue - 0.5;
			unit2 = unit1 / unit2 + 0.5;

			fitnessValue[ind] += wi[threadIdx.x] * unit2 + bias;
		}
	}
}

__device__ void hybirdFunc1(real* fitnessValue, real * pop, real* tmpValue, size_t size, int ind, real* result, real* local, real bias, real *wi, int *d_shuffle, int dim)
{
	real p[3] = { 0.3, 0.3, 0.4 };
	int lengthDim[3];

	lengthDim[0] = ceil(p[0] * dim);
	lengthDim[1] = ceil(p[1] * dim);
	lengthDim[2] = dim - lengthDim[0] - lengthDim[1];
	if (threadIdx.y == 0)
	{
		fitnessValue[ind] = 0;
	}
	int startDim, endDim = 0;
	startDim = endDim;
	endDim = lengthDim[0] + startDim;
	f9(fitnessValue, pop, size, ind, result, local, 0, wi, d_shuffle, startDim, lengthDim[0]);
	startDim = endDim;
	endDim = lengthDim[1] + startDim;
	f8(fitnessValue, pop, size, ind, result, local, 0, wi, d_shuffle, startDim, lengthDim[1]);
	startDim = endDim;
	f1(fitnessValue, pop, size, ind, result, local, 0, wi, d_shuffle, startDim, lengthDim[2]);

	if (threadIdx.y == 0)
		tmpValue[threadIdx.x] += fitnessValue[ind] + bias;
}

__device__ void hybirdFunc2(real* fitnessValue, real * pop, real* tmpValue, size_t size, int ind, real* result, real* local, real bias, real *wi, int *d_shuffle, int dim)
{

	real p[3] = { 0.3, 0.3, 0.4 };
	int lengthDim[3];

   	lengthDim[0] = ceil(p[0] * dim);
	lengthDim[1] = ceil(p[1] * dim);
	lengthDim[2] = dim - lengthDim[0] - lengthDim[1];
	if (threadIdx.y == 0)
		fitnessValue[ind] = 0;
	int startDim, endDim = 0;
	startDim = endDim;
	endDim = lengthDim[0] + startDim;
	f2(fitnessValue, pop, size, ind, result, local, 0, wi, d_shuffle, startDim, lengthDim[0]);
	startDim = endDim;
	endDim = lengthDim[1] + startDim;
	f12(fitnessValue, pop, size, ind, result, local, 0, wi, d_shuffle, startDim, lengthDim[1]);
	startDim = endDim;
	f8(fitnessValue, pop, size, ind, result, local, 0, wi, d_shuffle, startDim, lengthDim[2]);

	if (threadIdx.y == 0)
		tmpValue[threadIdx.x] += fitnessValue[ind] + bias;
}

__device__ void hybirdFunc3(real* fitnessValue, real * pop, real* tmpValue, size_t size, int ind, real* result, real* local, real bias, real *wi, int *d_shuffle, int dim)
{
	int funcList[4] = { 7, 6, 4, 14 };
	real p[4] = { 0.2, 0.2, 0.3, 0.3 };
	int lengthDim[4];
	lengthDim[0] = ceil(p[0] * dim);
	lengthDim[1] = ceil(p[1] * dim);
	lengthDim[2] = ceil(p[2] * dim);
	lengthDim[3] = dim - lengthDim[0] - lengthDim[1] - lengthDim[2];
	if (threadIdx.y == 0)
		fitnessValue[ind] = 0;
	int startDim, endDim = 0;

	startDim = endDim;
	endDim = lengthDim[0] + startDim;
	f7(fitnessValue, pop, size, ind, result, local, 0, wi, d_shuffle, startDim, lengthDim[0]);
	startDim = endDim;
	endDim = lengthDim[1] + startDim;
	f6(fitnessValue, pop, size, ind, result, local, 0, wi, d_shuffle, startDim, lengthDim[1]);
	startDim = endDim;
	endDim = lengthDim[2] + startDim;
	f4(fitnessValue, pop, size, ind, result, local, 0, wi, d_shuffle, startDim, lengthDim[2]);
	startDim = endDim;
	f14(fitnessValue, pop, size, ind, result, local, 0, wi, d_shuffle, startDim, lengthDim[3]);
	if (threadIdx.y == 0)
		tmpValue[threadIdx.x] += fitnessValue[ind] + bias;
}

__device__ void hybirdFunc4(real* fitnessValue, real * pop, real* tmpValue, size_t size, int ind, real* result, real* local, real bias, real *wi, int *d_shuffle, int dim)
{
	int funcList[4] = { 12, 3, 13, 8 };
	real p[4] = { 0.2, 0.2, 0.3, 0.3 };
	int lengthDim[4];
	lengthDim[0] = ceil(p[0] * dim);
	lengthDim[1] = ceil(p[1] * dim);
	lengthDim[2] = ceil(p[2] * dim);
	lengthDim[3] = dim - lengthDim[0] - lengthDim[1] - lengthDim[2];
	if (threadIdx.y == 0)
		fitnessValue[ind] = 0;
	int startDim, endDim = 0;

	startDim = endDim;
	endDim = lengthDim[0] + startDim;
	f12(fitnessValue, pop, size, ind, result, local, 0, wi, d_shuffle, startDim, lengthDim[0]);
	startDim = endDim;
	endDim = lengthDim[1] + startDim;
	f3(fitnessValue, pop, size, ind, result, local, 0, wi, d_shuffle, startDim, lengthDim[1]);
	startDim = endDim;
	endDim = lengthDim[2] + startDim;
	f13(fitnessValue, pop, size, ind, result, local, 0, wi, d_shuffle, startDim, lengthDim[2]);
	startDim = endDim;
	f8(fitnessValue, pop, size, ind, result, local, 0, wi, d_shuffle, startDim, lengthDim[3]);
	if (threadIdx.y == 0)
		tmpValue[threadIdx.x] += fitnessValue[ind] + bias;
}

__device__ void hybirdFunc5(real* fitnessValue, real * pop, real* tmpValue, size_t size, int ind, real* result, real* local, real bias, real *wi, int *d_shuffle, int dim)
{
	int funcList[5] = { 14, 12, 4, 9, 1 };
	real p[5] = { 0.1, 0.2, 0.2, 0.2, 0.3 };

	int lengthDim[5];
	lengthDim[0] = ceil(p[0] * dim);
	lengthDim[1] = ceil(p[1] * dim);
	lengthDim[2] = ceil(p[2] * dim);
	lengthDim[3] = ceil(p[3] * dim);
	lengthDim[4] = dim - lengthDim[0] - lengthDim[1] - lengthDim[2] - lengthDim[3];
	if (threadIdx.y == 0)
		fitnessValue[ind] = 0;
	int startDim, endDim = 0;

	startDim = endDim;
	endDim = lengthDim[0] + startDim;
	f14(fitnessValue, pop, size, ind, result, local, 0, wi, d_shuffle, startDim, lengthDim[0]);
	startDim = endDim;
	endDim = lengthDim[1] + startDim;
	f12(fitnessValue, pop, size, ind, result, local, 0, wi, d_shuffle, startDim, lengthDim[1]);
	startDim = endDim;
	endDim = lengthDim[2] + startDim;
	f4(fitnessValue, pop, size, ind, result, local, 0, wi, d_shuffle, startDim, lengthDim[2]);
	startDim = endDim;
	endDim = lengthDim[3] + startDim;
	f9(fitnessValue, pop, size, ind, result, local, 0, wi, d_shuffle, startDim, lengthDim[3]);
	startDim = endDim;
	f1(fitnessValue, pop, size, ind, result, local, 0, wi, d_shuffle, startDim, lengthDim[4]);
	if (threadIdx.y == 0)
		tmpValue[threadIdx.x] += fitnessValue[ind] + bias;
}

__device__ void hybirdFunc6(real* fitnessValue, real * pop, real* tmpValue, size_t size, int ind, real* result, real* local, real bias, real *wi, int *d_shuffle, int dim)
{
	int funcList[5] = { 10, 11, 13, 9, 5 };
	real p[5] = { 0.1, 0.2, 0.2, 0.2, 0.3 };
	int lengthDim[5];
	lengthDim[0] = ceil(p[0] * dim);
	lengthDim[1] = ceil(p[1] * dim);
	lengthDim[2] = ceil(p[2] * dim);
	lengthDim[3] = ceil(p[3] * dim);
	lengthDim[4] = dim - lengthDim[0] - lengthDim[1] - lengthDim[2] - lengthDim[3];
	if (threadIdx.y == 0)
		fitnessValue[ind] = 0;
	int startDim = 0;
	int endDim = 0;

	endDim = lengthDim[0] + startDim;
	f10(fitnessValue, pop, size, ind, result, local, 0, wi, d_shuffle, startDim, lengthDim[0]);
	startDim = endDim;
	endDim = lengthDim[1] + startDim;
	f11(fitnessValue, pop, size, ind, result, local, 0, wi, d_shuffle, startDim, lengthDim[1]);
	startDim = endDim;
	endDim = lengthDim[2] + startDim;
	f13(fitnessValue, pop, size, ind, result, local, 0, wi, d_shuffle, startDim, lengthDim[2]);
	startDim = endDim;
	endDim = lengthDim[3] + startDim;
	f9(fitnessValue, pop, size, ind, result, local, 0, wi, d_shuffle, startDim, lengthDim[3]);
	startDim = endDim;
	f5(fitnessValue, pop, size, ind, result, local, 0, wi, d_shuffle, startDim, lengthDim[4]);
	if (threadIdx.y == 0)
		tmpValue[threadIdx.x] += fitnessValue[ind] + bias;
}

extern __shared__ real shared[];
__global__ void global_f1(real* fitnessValue, real * pop, size_t size, int *d_shuffle, int dim)
{
	real bias = 100.0;
	int ind = threadIdx.x + blockIdx.x * blockDim.x;
	real * result = shared;
	real * local = result + blockDim.x;
	real * wi = local + blockDim.x * blockDim.y;

	if (threadIdx.y == 0)
	{
		fitnessValue[ind] = 0;
		wi[threadIdx.x] = 1;
	}
	f1(fitnessValue, pop, size, ind, result, local, bias, wi, NULL, -1, dim);
}

__global__ void global_f2(real* fitnessValue, real * pop, size_t size, int *d_shuffle, int dim)
{
	real bias = 200.0;
	int ind = threadIdx.x + blockIdx.x * blockDim.x;
	real * result = shared;
	real * local = result + blockDim.x;
	real * wi = local + blockDim.x * blockDim.y;

	if (threadIdx.y == 0)
	{
		fitnessValue[ind] = 0;
		wi[threadIdx.x] = 1;
	}
	f2(fitnessValue, pop, size, ind, result, local, bias, wi, NULL, -1, dim);
}

__global__ void global_f3(real* fitnessValue, real * pop, size_t size, int *d_shuffle, int dim)
{
	real bias = 300.0;
	int ind = threadIdx.x + blockIdx.x * blockDim.x;
	real * result = shared;
	real * local = result + blockDim.x;
	real * wi = local + blockDim.x * blockDim.y;

	if (threadIdx.y == 0)
	{
		fitnessValue[ind] = 0;
		wi[threadIdx.x] = 1;
	}
	f3(fitnessValue, pop, size, ind, result, local, bias, wi, NULL, -1, dim);
}

__global__ void global_f4(real* fitnessValue, real * pop, size_t size, int *d_shuffle, int dim)
{
	real bias = 400.0;
	int ind = threadIdx.x + blockIdx.x * blockDim.x;
	real * result = shared;
	real * local = result + blockDim.x;
	real * wi = local + blockDim.x * blockDim.y;

	if (threadIdx.y == 0)
	{
		fitnessValue[ind] = 0;
		wi[threadIdx.x] = 1;
	}
	f4(fitnessValue, pop, size, ind, result, local, bias, wi, NULL, -1, dim);
}

__global__ void global_f5(real* fitnessValue, real * pop, size_t size, int *d_shuffle, int dim)
{
	real bias = 500.0;
	int ind = threadIdx.x + blockIdx.x * blockDim.x;
	real * result = shared;
	real * wi = result + blockDim.x;
	real * local = wi + blockDim.x;


	if (threadIdx.y == 0)
	{
		fitnessValue[ind] = 0;
		wi[threadIdx.x] = 1;
	}
	f5(fitnessValue, pop, size, ind, result, local, bias, wi, NULL, -1, dim);
}

__global__ void global_f6(real* fitnessValue, real * pop, size_t size, int *d_shuffle, int dim)
{
	real bias = 600.0;
	int ind = threadIdx.x + blockIdx.x * blockDim.x;
	real * result = shared;
	real * local = result + blockDim.x;
	real * wi = local + blockDim.x * blockDim.y;

	if (threadIdx.y == 0)
	{
		fitnessValue[ind] = 0;
		wi[threadIdx.x] = 1;
	}
	f6(fitnessValue, pop, size, ind, result, local, bias, wi, NULL, -1, dim);
}

__global__ void global_f7(real* fitnessValue, real * pop, size_t size, int *d_shuffle, int dim)
{
	real bias = 700.0;
	int ind = threadIdx.x + blockIdx.x * blockDim.x;
	real * result = shared;
	real * wi = result + blockDim.x;
	real * local = wi + blockDim.x;

	if (threadIdx.y == 0)
	{
		fitnessValue[ind] = 0;
		wi[threadIdx.x] = 1;
	}
	f7(fitnessValue, pop, size, ind, result, local, bias, wi, NULL, -1, dim);
}

__global__ void global_f8(real* fitnessValue, real * pop, size_t size, int *d_shuffle, int dim)
{
	real bias = 800.0;
	int ind = threadIdx.x + blockIdx.x * blockDim.x;
	real * result = shared;
	real * local = result + blockDim.x;
	real * wi = local + blockDim.x * blockDim.y;

	if (threadIdx.y == 0)
	{
		fitnessValue[ind] = 0;
		wi[threadIdx.x] = 1;
	}
	f8(fitnessValue, pop, size, ind, result, local, bias, wi, NULL, -1, dim);
}

__global__ void global_f9(real* fitnessValue, real * pop, size_t size, int *d_shuffle, int dim)
{
	real bias = 900.0;
	int ind = threadIdx.x + blockIdx.x * blockDim.x;
	real * result = shared;
	real * local = result + blockDim.x;
	real * wi = local + blockDim.x * blockDim.y;

	if (threadIdx.y == 0)
	{
		fitnessValue[ind] = 0;
		wi[threadIdx.x] = 1;
	}
	f8(fitnessValue, pop, size, ind, result, local, bias, wi, NULL, -1, dim);
}

__global__ void global_f10(real* fitnessValue, real * pop, size_t size, int *d_shuffle, int dim)
{
	real bias = 1000.0;
	int ind = threadIdx.x + blockIdx.x * blockDim.x;
	real * result = shared;
	real * local = result + blockDim.x;
	real * wi = local + blockDim.x * blockDim.y;

	if (threadIdx.y == 0)
	{
		fitnessValue[ind] = 0;
		wi[threadIdx.x] = 1;
	}
	f9(fitnessValue, pop, size, ind, result, local, bias, wi, NULL, -1, dim);
}

__global__ void global_f11(real* fitnessValue, real * pop, size_t size, int *d_shuffle, int dim)
{
	real bias = 1100.0;
	int ind = threadIdx.x + blockIdx.x * blockDim.x;
	real * result = shared;
	real * local = result + blockDim.x;
	real * wi = local + blockDim.x * blockDim.y;

	if (threadIdx.y == 0)
	{
		fitnessValue[ind] = 0;
		wi[threadIdx.x] = 1;
	}
	f9(fitnessValue, pop, size, ind, result, local, bias, wi, NULL, -1, dim);
}

__global__ void global_f12(real* fitnessValue, real * pop, size_t size, int *d_shuffle, int dim)
{
	real bias = 1200.0;
	int ind = threadIdx.x + blockIdx.x * blockDim.x;
	real * result = shared;
	real * local = result + blockDim.x;
	real * wi = local + blockDim.x * blockDim.y;

	if (threadIdx.y == 0)
	{
		fitnessValue[ind] = 0;
		wi[threadIdx.x] = 1;
	}
	f10(fitnessValue, pop, size, ind, result, local, bias, wi, NULL, -1, dim);
}

__global__ void global_f13(real* fitnessValue, real * pop, size_t size, int *d_shuffle, int dim)
{
	real bias = 1300.0;
	int ind = threadIdx.x + blockIdx.x * blockDim.x;
	real * result = shared;
	real * wi = result + blockDim.x;
	real * local = wi + blockDim.x;

	if (threadIdx.y == 0)
	{
		fitnessValue[ind] = 0;
		wi[threadIdx.x] = 1;
	}
	f11(fitnessValue, pop, size, ind, result, local, bias, wi, NULL, -1, dim);
}

__global__ void global_f14(real* fitnessValue, real * pop, size_t size, int *d_shuffle, int dim)
{
	real bias = 1400.0;
	int ind = threadIdx.x + blockIdx.x * blockDim.x;
	real * result = shared;
	real * wi = result + blockDim.x;
	real * local = wi + blockDim.x;

	if (threadIdx.y == 0)
	{
		fitnessValue[ind] = 0;
		wi[threadIdx.x] = 1;
	}
	f12(fitnessValue, pop, size, ind, result, local, bias, wi, NULL, -1, dim);
}

__global__ void global_f15(real* fitnessValue, real * pop, size_t size, int *d_shuffle, int dim)
{
	real bias = 1500.0;
	int ind = threadIdx.x + blockIdx.x * blockDim.x;
	real * result = shared;
	real * local = result + blockDim.x;
	real * wi = local + blockDim.x * blockDim.y;

	if (threadIdx.y == 0)
	{
		fitnessValue[ind] = 0;
		wi[threadIdx.x] = 1;
	}
	f13(fitnessValue, pop, size, ind, result, local, bias, wi, NULL, -1, dim);
}

__global__ void global_f16(real* fitnessValue, real * pop, size_t size, int *d_shuffle, int dim)
{
	real bias = 1600.0;
	int ind = threadIdx.x + blockIdx.x * blockDim.x;
	real * result = shared;
	real * local = result + blockDim.x;
	real * wi = local + blockDim.x * blockDim.y;

	if (threadIdx.y == 0)
	{
		fitnessValue[ind] = 0;
		wi[threadIdx.x] = 1;
	}
	f14(fitnessValue, pop, size, ind, result, local, bias, wi, NULL, -1, dim);
}

__global__ void global_f17(real* fitnessValue, real * pop, size_t size, int *d_shuffle, int dim)
{
	real bias = 1700.0;
	int ind = threadIdx.x + blockIdx.x * blockDim.x;
	real * result = shared;
	real * wi = result + blockDim.x * blockDim.y;
	real * tmpValue = wi + blockDim.x;
	real * local = tmpValue + blockDim.x;
	if (threadIdx.y == 0)
	{
		fitnessValue[ind] = 0;
		wi[threadIdx.x] = 1;
		tmpValue[threadIdx.x] = 0;
	}
	hybirdFunc1(fitnessValue, pop, tmpValue, size, ind, result, local, 0, wi, d_shuffle, dim);
	if (threadIdx.y == 0)
		fitnessValue[ind] = tmpValue[threadIdx.x] + bias;
}

__global__ void global_f18(real* fitnessValue, real * pop, size_t size, int *d_shuffle, int dim)
{
	real bias = 1800.0;
	int ind = threadIdx.x + blockIdx.x * blockDim.x;
	real * result = shared;
	real * wi = result + blockDim.x * blockDim.y;
	real * tmpValue = wi + blockDim.x;
	real * local = tmpValue + blockDim.x;
	if (threadIdx.y == 0)
	{
		fitnessValue[ind] = 0;
		wi[threadIdx.x] = 1;
		tmpValue[threadIdx.x] = 0;
	}
	hybirdFunc2(fitnessValue, pop, tmpValue, size, ind, result, local, 0, wi, d_shuffle, dim);
	if (threadIdx.y == 0)
		fitnessValue[ind] = tmpValue[threadIdx.x] + bias;
}

__global__ void global_f19(real* fitnessValue, real * pop, size_t size, int *d_shuffle, int dim)
{
	real bias = 1900.0;
	int ind = threadIdx.x + blockIdx.x * blockDim.x;
	real * result = shared;
	real * wi = result + blockDim.x * blockDim.y;
	real * tmpValue = wi + blockDim.x;
	real * local = tmpValue + blockDim.x;
	if (threadIdx.y == 0)
	{
		fitnessValue[ind] = 0;
		wi[threadIdx.x] = 1;
		tmpValue[threadIdx.x] = 0;
	}
	hybirdFunc3(fitnessValue, pop, tmpValue, size, ind, result, local, 0, wi, d_shuffle, dim);
	if (threadIdx.y == 0)
		fitnessValue[ind] = tmpValue[threadIdx.x] + bias;
}

__global__ void global_f20(real* fitnessValue, real * pop, size_t size, int *d_shuffle, int dim)
{
	real bias = 2000.0;
	int ind = threadIdx.x + blockIdx.x * blockDim.x;
	real * result = shared;
	real * wi = result + blockDim.x * blockDim.y;
	real * tmpValue = wi + blockDim.x;
	real * local = tmpValue + blockDim.x;
	if (threadIdx.y == 0)
	{
		fitnessValue[ind] = 0;
		wi[threadIdx.x] = 1;
		tmpValue[threadIdx.x] = 0;
	}
	hybirdFunc4(fitnessValue, pop, tmpValue, size, ind, result, local, 0, wi, d_shuffle, dim);
	if (threadIdx.y == 0)
		fitnessValue[ind] = tmpValue[threadIdx.x] + bias;
}

__global__ void global_f21(real* fitnessValue, real * pop, size_t size, int *d_shuffle, int dim)
{
	real bias = 2100.0;
	int ind = threadIdx.x + blockIdx.x * blockDim.x;
	real * result = shared;
	real * wi = result + blockDim.x * blockDim.y;
	real * tmpValue = wi + blockDim.x;
	real * local = tmpValue + blockDim.x;
	if (threadIdx.y == 0)
	{
		fitnessValue[ind] = 0;
		wi[threadIdx.x] = 1;
		tmpValue[threadIdx.x] = 0;
	}
	hybirdFunc5(fitnessValue, pop, tmpValue, size, ind, result, local, 0, wi, d_shuffle, dim);
	if (threadIdx.y == 0)
		fitnessValue[ind] = tmpValue[threadIdx.x] + bias;
}

__global__ void global_f22(real* fitnessValue, real * pop, size_t size, int *d_shuffle, int dim)
{
	real bias = 2200.0;
	int ind = threadIdx.x + blockIdx.x * blockDim.x;
	real * result = shared;
	real * wi = result + blockDim.x * blockDim.y;
	real * tmpValue = wi + blockDim.x;
	real * local = tmpValue + blockDim.x;
	if (threadIdx.y == 0)
	{
		fitnessValue[ind] = 0;
		wi[threadIdx.x] = 1;
		tmpValue[threadIdx.x] = 0;
	}
	hybirdFunc6(fitnessValue, pop, tmpValue, size, ind, result, local, 0, wi, d_shuffle, dim);
	if (threadIdx.y == 0)
		fitnessValue[ind] = tmpValue[threadIdx.x] + bias;
}

__global__ void global_f23(real* fitnessValue, real * pop, real * rotpop, size_t size, int dim)
{
	real subbias;
	real bias = 2300;
	int numFunc = 5;
	real * result = shared;
	real * wi = result + blockDim.x;
	real * wi2 = wi + blockDim.x;
	real * sumWi = wi2 + blockDim.x;
	real * local = sumWi + blockDim.x;
	int ind = threadIdx.x + blockIdx.x * blockDim.x;

	real unit1, unit2, unit3, unit4;
	if (threadIdx.y == 0)
	{
		sumWi[threadIdx.x] = 0;
		fitnessValue[ind] = 0;
	}
	//shift data for calWi;
	unit1 = pop[ind + size * threadIdx.y] - d_shift[threadIdx.y];
	local[threadIdx.x + blockDim.x * threadIdx.y] = unit1 * unit1;
	__syncthreads();
	parallelsum(local, &result[threadIdx.x], blockDim.y);
	if (threadIdx.y == 0)
	{
		unit2 = result[threadIdx.x];
		unit3 = unit2 / (double)(2 * dim * 100.0);
		wi[threadIdx.x] = 1 / sqrt(unit2) * exp(-unit3);
		sumWi[threadIdx.x] += wi[threadIdx.x];
	}
	__syncthreads();

	if (threadIdx.y == 0)
	{
		wi2[threadIdx.x] = wi[threadIdx.x];
		subbias = 0;
	}
	f4(fitnessValue, rotpop, size * numFunc, ind, result, local, subbias, wi2, NULL, -1, dim);

	__syncthreads();
	unit1 = pop[ind + size * threadIdx.y] - d_shift[threadIdx.y + MAX_DIM];
	local[threadIdx.x + blockDim.x * threadIdx.y] = unit1 * unit1;
	__syncthreads();
	parallelsum(local, &result[threadIdx.x], blockDim.y);
	if (threadIdx.y == 0)
	{
		unit2 = result[threadIdx.x];
		unit3 = unit2 / (double)(2 * dim * 400.0);
		wi[threadIdx.x] = 1 / sqrt(unit2) * exp(-unit3);
		sumWi[threadIdx.x] += wi[threadIdx.x];
	}
	__syncthreads();


	if (threadIdx.y == 0)
	{
		wi2[threadIdx.x] = 1e-6 * wi[threadIdx.x];
		subbias = 100.0 * wi[threadIdx.x];
	}
	f1(fitnessValue, rotpop + size, size * numFunc, ind, result, local, subbias, wi2, NULL, -1, dim);


	__syncthreads();
	unit1 = pop[ind + size * threadIdx.y] - d_shift[threadIdx.y + 2 * MAX_DIM];
	local[threadIdx.x + blockDim.x * threadIdx.y] = unit1 * unit1;
	__syncthreads();
	parallelsum(local, &result[threadIdx.x], blockDim.y);
	if (threadIdx.y == 0)
	{
		unit2 = result[threadIdx.x];
		unit3 = unit2 / (double)(2 * dim * 900.0);
		wi[threadIdx.x] = 1 / sqrt(unit2) * exp(-unit3);
		sumWi[threadIdx.x] += wi[threadIdx.x];
	}
	__syncthreads();
	if (threadIdx.y == 0)
	{
		wi2[threadIdx.x] = 1e-26 * wi[threadIdx.x];
		subbias =  200.0 * wi[threadIdx.x];

	}
	f2(fitnessValue, rotpop + 2 * size, size * numFunc, ind, result, local, subbias, wi2, NULL, -1, dim);


	__syncthreads();
	unit1 = pop[ind + size * threadIdx.y] - d_shift[threadIdx.y + 3 * MAX_DIM];
	local[threadIdx.x + blockDim.x * threadIdx.y] = unit1 * unit1;
	__syncthreads();
	parallelsum(local, &result[threadIdx.x], blockDim.y);
	if (threadIdx.y == 0)
	{
		unit2 = result[threadIdx.x];
		unit3 = unit2 / (double)(2 * dim * 1600.0);
		wi[threadIdx.x] = 1 / sqrt(unit2) * exp(-unit3);
		sumWi[threadIdx.x] += wi[threadIdx.x];
	}
	__syncthreads();
	if (threadIdx.y == 0)
	{
		wi2[threadIdx.x] =  1e-6 * wi[threadIdx.x];
		subbias =  300.0 * wi[threadIdx.x];

	}
	f3(fitnessValue, rotpop + 3 * size, size * numFunc, ind, result, local, subbias, wi2, NULL, -1, dim);


	__syncthreads();
	unit1 = pop[ind + size * threadIdx.y] - d_shift[threadIdx.y + 4 * MAX_DIM];
	local[threadIdx.x + blockDim.x * threadIdx.y] = unit1 * unit1;
	__syncthreads();
	parallelsum(local, &result[threadIdx.x], blockDim.y);
	if (threadIdx.y == 0)
	{
		unit2 = result[threadIdx.x];
		unit3 = unit2 / (double)(2 * dim * 2500.0);
		wi[threadIdx.x] = 1 / sqrt(unit2) * exp(-unit3);
		sumWi[threadIdx.x] += wi[threadIdx.x];
	}
	__syncthreads();
	if (threadIdx.y == 0)
	{
		wi2[threadIdx.x] = 1e-6 * wi[threadIdx.x];
		subbias = 400.0 * wi[threadIdx.x];
	}
	f1(fitnessValue, rotpop + 4 * size, size * numFunc, ind, result, local, subbias, wi2, NULL, -1, dim);
	if (threadIdx.y == 0)
		fitnessValue[ind] = fitnessValue[ind] / sumWi[threadIdx.x] + bias;
}
__global__ void global_f24(real* fitnessValue, real * pop, real * rotpop, size_t size, int dim)
{
	real subbias;
	real bias = 2400;
	int numFunc = 3;
	real * result = shared;
	real * wi = result + blockDim.x;
	real * wi2 = wi + blockDim.x;
	real * sumWi = wi2 + blockDim.x;
	real * local = sumWi + blockDim.x;
	int ind = threadIdx.x + blockIdx.x * blockDim.x;

	real unit1, unit2, unit3, unit4;
	if (threadIdx.y == 0)
	{
		sumWi[threadIdx.x] = 0;
		fitnessValue[ind] = 0;
	}
	//shift data for calWi;
	unit1 = pop[ind + size * threadIdx.y] - d_shift[threadIdx.y];
	local[threadIdx.x + blockDim.x * threadIdx.y] = unit1 * unit1;
	__syncthreads();
	parallelsum(local, &result[threadIdx.x], blockDim.y);
	if (threadIdx.y == 0)
	{
		unit2 = result[threadIdx.x];
		unit3 = unit2 / (double)(2 * dim * 400.0);
		wi[threadIdx.x] = 1 / sqrt(unit2) * exp(-unit3);
		sumWi[threadIdx.x] += wi[threadIdx.x];
	}
	__syncthreads();

	if (threadIdx.y == 0)
	{
		wi2[threadIdx.x] = wi[threadIdx.x];
		subbias = 0;
	}
	f9(fitnessValue, rotpop, size * numFunc, ind, result, local, subbias, wi2, NULL, -1, dim);

	__syncthreads();
	unit1 = pop[ind + size * threadIdx.y] - d_shift[threadIdx.y + MAX_DIM];
	local[threadIdx.x + blockDim.x * threadIdx.y] = unit1 * unit1;
	__syncthreads();
	parallelsum(local, &result[threadIdx.x], blockDim.y);
	if (threadIdx.y == 0)
	{
		unit2 = result[threadIdx.x];
		unit3 = unit2 / (double)(2 * dim * 400.0);
		wi[threadIdx.x] = 1 / sqrt(unit2) * exp(-unit3);
		sumWi[threadIdx.x] += wi[threadIdx.x];
	}
	__syncthreads();


	if (threadIdx.y == 0)
	{
		wi2[threadIdx.x] = wi[threadIdx.x];
		subbias = 100.0 * wi[threadIdx.x];
	}
	f8(fitnessValue, rotpop + size, size * numFunc, ind, result, local, subbias, wi2, NULL, -1, dim);


	__syncthreads();
	unit1 = pop[ind + size * threadIdx.y] - d_shift[threadIdx.y + 2 * MAX_DIM];
	local[threadIdx.x + blockDim.x * threadIdx.y] = unit1 * unit1;
	__syncthreads();
	parallelsum(local, &result[threadIdx.x], blockDim.y);
	if (threadIdx.y == 0)
	{
		unit2 = result[threadIdx.x];
		unit3 = unit2 / (double)(2 * dim * 400.0);
		wi[threadIdx.x] = 1 / sqrt(unit2) * exp(-unit3);
		sumWi[threadIdx.x] += wi[threadIdx.x];
	}
	__syncthreads();
	if (threadIdx.y == 0)
	{
		wi2[threadIdx.x] = wi[threadIdx.x];
		subbias = 200.0 * wi[threadIdx.x];

	}
	f12(fitnessValue, rotpop + 2 * size, size * numFunc, ind, result, local, subbias, wi2, NULL, -1, dim);
	if (threadIdx.y == 0)
		fitnessValue[ind] = fitnessValue[ind] / sumWi[threadIdx.x] + bias;
}
__global__ void global_f25(real* fitnessValue, real * pop, real * rotpop, size_t size, int dim)
{
	real subbias;
	real bias = 2500;
	int numFunc = 3;
	real * result = shared;
	real * wi = result + blockDim.x;
	real * wi2 = wi + blockDim.x;
	real * sumWi = wi2 + blockDim.x;
	real * local = sumWi + blockDim.x;
	int ind = threadIdx.x + blockIdx.x * blockDim.x;

	real unit1, unit2, unit3, unit4;
	if (threadIdx.y == 0)
	{
		sumWi[threadIdx.x] = 0;
		fitnessValue[ind] = 0;
	}
	//shift data for calWi;
	unit1 = pop[ind + size * threadIdx.y] - d_shift[threadIdx.y];
	local[threadIdx.x + blockDim.x * threadIdx.y] = unit1 * unit1;
	__syncthreads();
	parallelsum(local, &result[threadIdx.x], blockDim.y);
	if (threadIdx.y == 0)
	{
		unit2 = result[threadIdx.x];
		unit3 = unit2 / (double)(2 * dim * 100.0);
		wi[threadIdx.x] = 1 / sqrt(unit2) * exp(-unit3);
		sumWi[threadIdx.x] += wi[threadIdx.x];
	}
	__syncthreads();

	if (threadIdx.y == 0)
	{
		wi2[threadIdx.x] = 0.25 * wi[threadIdx.x];
		subbias = 0;
	}
	f9(fitnessValue, rotpop, size * numFunc, ind, result, local, subbias, wi2, NULL, -1, dim);

	__syncthreads();
	unit1 = pop[ind + size * threadIdx.y] - d_shift[threadIdx.y + MAX_DIM];
	local[threadIdx.x + blockDim.x * threadIdx.y] = unit1 * unit1;
	__syncthreads();
	parallelsum(local, &result[threadIdx.x], blockDim.y);
	if (threadIdx.y == 0)
	{
		unit2 = result[threadIdx.x];
		unit3 = unit2 / (double)(2 * dim * 900.0);
		wi[threadIdx.x] = 1 / sqrt(unit2) * exp(-unit3);
		sumWi[threadIdx.x] += wi[threadIdx.x];
	}
	__syncthreads();


	if (threadIdx.y == 0)
	{
		wi2[threadIdx.x] = wi[threadIdx.x];
		subbias = 100.0 * wi[threadIdx.x];
	}
	f8(fitnessValue, rotpop + size * 1, size * numFunc, ind, result, local, subbias, wi2, NULL, -1, dim);


	__syncthreads();
	unit1 = pop[ind + size * threadIdx.y] - d_shift[threadIdx.y + 2 * MAX_DIM];
	local[threadIdx.x + blockDim.x * threadIdx.y] = unit1 * unit1;
	__syncthreads();
	parallelsum(local, &result[threadIdx.x], blockDim.y);
	if (threadIdx.y == 0)
	{
		unit2 = result[threadIdx.x];
		unit3 = unit2 / (double)(2 * dim * 2500.0);
		wi[threadIdx.x] = 1 / sqrt(unit2) * exp(-unit3);
		sumWi[threadIdx.x] += wi[threadIdx.x];
	}
	__syncthreads();
	if (threadIdx.y == 0)
	{
		wi2[threadIdx.x] = 1e-7 * wi[threadIdx.x];
		subbias = 200.0 * wi[threadIdx.x];
	}
	f1(fitnessValue, rotpop + 2 * size, size * numFunc, ind, result, local, subbias, wi2, NULL, -1, dim);
	if (threadIdx.y == 0)
		fitnessValue[ind] = fitnessValue[ind] / sumWi[threadIdx.x] + bias;
}
__global__ void global_f26(real* fitnessValue, real * pop, real * rotpop, size_t size, int dim)
{
	real subbias;
	real bias = 2600;
	int numFunc = 5;
	real * result = shared;
	real * wi = result + blockDim.x;
	real * wi2 = wi + blockDim.x;
	real * sumWi = wi2 + blockDim.x;
	real * local = sumWi + blockDim.x;
	int ind = threadIdx.x + blockIdx.x * blockDim.x;

	real unit1, unit2, unit3, unit4;
	if (threadIdx.y == 0)
	{
		sumWi[threadIdx.x] = 0;
		fitnessValue[ind] = 0;
	}
	//shift data for calWi;
	unit1 = pop[ind + size * threadIdx.y] - d_shift[threadIdx.y];
	local[threadIdx.x + blockDim.x * threadIdx.y] = unit1 * unit1;
	__syncthreads();
	parallelsum(local, &result[threadIdx.x], blockDim.y);
	if (threadIdx.y == 0)
	{
		unit2 = result[threadIdx.x];
		unit3 = unit2 / (double)(2 * dim * 100.0);
		wi[threadIdx.x] = 1 / sqrt(unit2) * exp(-unit3);
		sumWi[threadIdx.x] += wi[threadIdx.x];
	}
	__syncthreads();

	if (threadIdx.y == 0)
	{
		wi2[threadIdx.x] = 0.25 * wi[threadIdx.x];
		subbias = 0;
	}
	f9(fitnessValue, rotpop, size * numFunc, ind, result, local, subbias, wi2, NULL, -1, dim);

	__syncthreads();
	unit1 = pop[ind + size * threadIdx.y] - d_shift[threadIdx.y + MAX_DIM];
	local[threadIdx.x + blockDim.x * threadIdx.y] = unit1 * unit1;
	__syncthreads();
	parallelsum(local, &result[threadIdx.x], blockDim.y);
	if (threadIdx.y == 0)
	{
		unit2 = result[threadIdx.x];
		unit3 = unit2 / (double)(2 * dim * 100.0);
		wi[threadIdx.x] = 1 / sqrt(unit2) * exp(-unit3);
		sumWi[threadIdx.x] += wi[threadIdx.x];
	}
	__syncthreads();


	if (threadIdx.y == 0)
	{
		wi2[threadIdx.x] = 1 * wi[threadIdx.x];
		subbias = 100.0 * wi[threadIdx.x];
	}
	f11(fitnessValue, rotpop + size * 1, size * numFunc, ind, result, local, subbias, wi2, NULL, -1, dim);


	__syncthreads();
	unit1 = pop[ind + size * threadIdx.y] - d_shift[threadIdx.y + 2 * MAX_DIM];
	local[threadIdx.x + blockDim.x * threadIdx.y] = unit1 * unit1;
	__syncthreads();
	parallelsum(local, &result[threadIdx.x], blockDim.y);
	if (threadIdx.y == 0)
	{
		unit2 = result[threadIdx.x];
		unit3 = unit2 / (double)(2 * dim * 100.0);
		wi[threadIdx.x] = 1 / sqrt(unit2) * exp(-unit3);
		sumWi[threadIdx.x] += wi[threadIdx.x];
	}
	__syncthreads();
	if (threadIdx.y == 0)
	{
		wi2[threadIdx.x] = 1e-7 * wi[threadIdx.x];
		subbias = 200.0 * wi[threadIdx.x];
	}
	f1(fitnessValue, rotpop + 2 * size, size * numFunc, ind, result, local, subbias, wi2, NULL, -1, dim);

	__syncthreads();
	unit1 = pop[ind + size * threadIdx.y] - d_shift[threadIdx.y + 3 * MAX_DIM];
	local[threadIdx.x + blockDim.x * threadIdx.y] = unit1 * unit1;
	__syncthreads();
	parallelsum(local, &result[threadIdx.x], blockDim.y);
	if (threadIdx.y == 0)
	{
		unit2 = result[threadIdx.x];
		unit3 = unit2 / (double)(2 * dim * 100.0);
		wi[threadIdx.x] = 1 / sqrt(unit2) * exp(-unit3);
		sumWi[threadIdx.x] += wi[threadIdx.x];
	}
	__syncthreads();
	if (threadIdx.y == 0)
	{
		wi2[threadIdx.x] = 2.5 * wi[threadIdx.x];
		subbias = 300.0 * wi[threadIdx.x];
	}
	f6(fitnessValue, rotpop + 3 * size, size * numFunc, ind, result, local, subbias, wi2, NULL, -1, dim);

	__syncthreads();
	unit1 = pop[ind + size * threadIdx.y] - d_shift[threadIdx.y + 4 * MAX_DIM];
	local[threadIdx.x + blockDim.x * threadIdx.y] = unit1 * unit1;
	__syncthreads();
	parallelsum(local, &result[threadIdx.x], blockDim.y);
	if (threadIdx.y == 0)
	{
		unit2 = result[threadIdx.x];
		unit3 = unit2 / (double)(2 * dim * 100.0);
		wi[threadIdx.x] = 1 / sqrt(unit2) * exp(-unit3);
		sumWi[threadIdx.x] += wi[threadIdx.x];
	}
	__syncthreads();
	if (threadIdx.y == 0)
	{
		wi2[threadIdx.x] = 10 * wi[threadIdx.x];
		subbias = 400.0 * wi[threadIdx.x];
	}
	f7(fitnessValue, rotpop + 4 * size, size * numFunc, ind, result, local, subbias, wi2, NULL, -1, dim);
	if (threadIdx.y == 0)
		fitnessValue[ind] = fitnessValue[ind] / sumWi[threadIdx.x] + bias;
}
__global__ void global_f27(real* fitnessValue, real * pop, real * rotpop, size_t size, int dim)
{
	real subbias;
	real bias = 2700;
	int numFunc = 5;
	real * result = shared;
	real * wi = result + blockDim.x;
	real * wi2 = wi + blockDim.x;
	real * sumWi = wi2 + blockDim.x;
	real * local = sumWi + blockDim.x;
	int ind = threadIdx.x + blockIdx.x * blockDim.x;

	real unit1, unit2, unit3, unit4;
	if (threadIdx.y == 0)
	{
		sumWi[threadIdx.x] = 0;
		fitnessValue[ind] = 0;
	}
	//shift data for calWi;
	unit1 = pop[ind + size * threadIdx.y] - d_shift[threadIdx.y];
	local[threadIdx.x + blockDim.x * threadIdx.y] = unit1 * unit1;
	__syncthreads();
	parallelsum(local, &result[threadIdx.x], blockDim.y);
	if (threadIdx.y == 0)
	{
		unit2 = result[threadIdx.x];
		unit3 = unit2 / (double)(2 * dim * 100.0);
		wi[threadIdx.x] = 1 / sqrt(unit2) * exp(-unit3);
		sumWi[threadIdx.x] += wi[threadIdx.x];
	}
	__syncthreads();

	if (threadIdx.y == 0)
	{
		wi2[threadIdx.x] = 10 * wi[threadIdx.x];
		subbias = 0;
	}
	f12(fitnessValue, rotpop, size * numFunc, ind, result, local, subbias, wi2, NULL, -1, dim);

	__syncthreads();
	unit1 = pop[ind + size * threadIdx.y] - d_shift[threadIdx.y + MAX_DIM];
	local[threadIdx.x + blockDim.x * threadIdx.y] = unit1 * unit1;
	__syncthreads();
	parallelsum(local, &result[threadIdx.x], blockDim.y);
	if (threadIdx.y == 0)
	{
		unit2 = result[threadIdx.x];
		unit3 = unit2 / (double)(2 * dim * 100.0);
		wi[threadIdx.x] = 1 / sqrt(unit2) * exp(-unit3);
		sumWi[threadIdx.x] += wi[threadIdx.x];
	}
	__syncthreads();


	if (threadIdx.y == 0)
	{
		wi2[threadIdx.x] = 10 * wi[threadIdx.x];
		subbias = 100.0 * wi[threadIdx.x];
	}
	f8(fitnessValue, rotpop + size * 1, size * numFunc, ind, result, local, subbias, wi2, NULL, -1, dim);


	__syncthreads();
	unit1 = pop[ind + size * threadIdx.y] - d_shift[threadIdx.y + 2 * MAX_DIM];
	local[threadIdx.x + blockDim.x * threadIdx.y] = unit1 * unit1;
	__syncthreads();
	parallelsum(local, &result[threadIdx.x], blockDim.y);
	if (threadIdx.y == 0)
	{
		unit2 = result[threadIdx.x];
		unit3 = unit2 / (double)(2 * dim * 100.0);
		wi[threadIdx.x] = 1 / sqrt(unit2) * exp(-unit3);
		sumWi[threadIdx.x] += wi[threadIdx.x];
	}
	__syncthreads();
	if (threadIdx.y == 0)
	{;
		wi2[threadIdx.x] = 2.5 * wi[threadIdx.x];
		subbias = 200.0 * wi[threadIdx.x];
	}
	f9(fitnessValue, rotpop + 2 * size, size * numFunc, ind, result, local, subbias, wi2, NULL, -1, dim);

	__syncthreads();
	unit1 = pop[ind + size * threadIdx.y] - d_shift[threadIdx.y + 3 * MAX_DIM];
	local[threadIdx.x + blockDim.x * threadIdx.y] = unit1 * unit1;
	__syncthreads();
	parallelsum(local, &result[threadIdx.x], blockDim.y);
	if (threadIdx.y == 0)
	{
		unit2 = result[threadIdx.x];
		unit3 = unit2 / (double)(2 * dim * 400.0);
		wi[threadIdx.x] = 1 / sqrt(unit2) * exp(-unit3);
		sumWi[threadIdx.x] += wi[threadIdx.x];
	}
	__syncthreads();
	if (threadIdx.y == 0)
	{
		wi2[threadIdx.x] = 25 * wi[threadIdx.x];
		subbias = 300.0 * wi[threadIdx.x];
	}
	f6(fitnessValue, rotpop + 3 * size, size * numFunc, ind, result, local, subbias, wi2, NULL, -1, dim);

	__syncthreads();
	unit1 = pop[ind + size * threadIdx.y] - d_shift[threadIdx.y + 4 * MAX_DIM];
	local[threadIdx.x + blockDim.x * threadIdx.y] = unit1 * unit1;
	__syncthreads();
	parallelsum(local, &result[threadIdx.x], blockDim.y);
	if (threadIdx.y == 0)
	{
		unit2 = result[threadIdx.x];
		unit3 = unit2 / (double)(2 * dim * 400.0);
		wi[threadIdx.x] = 1 / sqrt(unit2) * exp(-unit3);
		sumWi[threadIdx.x] += wi[threadIdx.x];
	}
	__syncthreads();
	if (threadIdx.y == 0)
	{
		wi2[threadIdx.x] = 1e-6 * wi[threadIdx.x];
		subbias = 400.0 * wi[threadIdx.x];
	}
	f1(fitnessValue, rotpop + 4 * size, size * numFunc, ind, result, local, subbias, wi2, NULL, -1, dim);
	if (threadIdx.y == 0)
		fitnessValue[ind] = fitnessValue[ind] / sumWi[threadIdx.x] + bias;
}
__global__ void global_f28(real* fitnessValue, real * pop, real * rotpop, size_t size, int dim)
{
	real subbias;
	real bias = 2800;
	int numFunc = 5;
	real * result = shared;
	real * wi = result + blockDim.x;
	real * wi2 = wi + blockDim.x;
	real * sumWi = wi2 + blockDim.x;
	real * local = sumWi + blockDim.x;
	int ind = threadIdx.x + blockIdx.x * blockDim.x;

	real unit1, unit2, unit3, unit4;
	if (threadIdx.y == 0)
	{
		sumWi[threadIdx.x] = 0;
		fitnessValue[ind] = 0;
	}
	//shift data for calWi;
	unit1 = pop[ind + size * threadIdx.y] - d_shift[threadIdx.y];
	local[threadIdx.x + blockDim.x * threadIdx.y] = unit1 * unit1;
	__syncthreads();
	parallelsum(local, &result[threadIdx.x], blockDim.y);
	if (threadIdx.y == 0)
	{
		unit2 = result[threadIdx.x];
		unit3 = unit2 / (double)(2 * dim * 100.0);
		wi[threadIdx.x] = 1 / sqrt(unit2) * exp(-unit3);
		sumWi[threadIdx.x] += wi[threadIdx.x];
	}
	__syncthreads();

	if (threadIdx.y == 0)
	{
		wi2[threadIdx.x] = 2.5 * wi[threadIdx.x];
		subbias = 0;
	}
	f13(fitnessValue, rotpop, size * numFunc, ind, result, local, subbias, wi2, NULL, -1, dim);

	__syncthreads();
	unit1 = pop[ind + size * threadIdx.y] - d_shift[threadIdx.y + MAX_DIM];
	local[threadIdx.x + blockDim.x * threadIdx.y] = unit1 * unit1;
	__syncthreads();
	parallelsum(local, &result[threadIdx.x], blockDim.y);
	if (threadIdx.y == 0)
	{
		unit2 = result[threadIdx.x];
		unit3 = unit2 / (double)(2 * dim * 400.0);
		wi[threadIdx.x] = 1 / sqrt(unit2) * exp(-unit3);
		sumWi[threadIdx.x] += wi[threadIdx.x];
	}
	__syncthreads();


	if (threadIdx.y == 0)
	{
		wi2[threadIdx.x] = 10 * wi[threadIdx.x];
		subbias = 100.0 * wi[threadIdx.x];

	}
	f11(fitnessValue, rotpop + size * 1, size * numFunc, ind, result, local, subbias, wi2, NULL, -1, dim);


	__syncthreads();
	unit1 = pop[ind + size * threadIdx.y] - d_shift[threadIdx.y + 2 * MAX_DIM];
	local[threadIdx.x + blockDim.x * threadIdx.y] = unit1 * unit1;
	__syncthreads();
	parallelsum(local, &result[threadIdx.x], blockDim.y);
	if (threadIdx.y == 0)
	{
		unit2 = result[threadIdx.x];
		unit3 = unit2 / (double)(2 * dim * 900.0);
		wi[threadIdx.x] = 1 / sqrt(unit2) * exp(-unit3);
		sumWi[threadIdx.x] += wi[threadIdx.x];
	}
	__syncthreads();
	if (threadIdx.y == 0)
	{
		wi2[threadIdx.x] = 2.5 * wi[threadIdx.x];
		subbias = 200.0 * wi[threadIdx.x];
	}
	f9(fitnessValue, rotpop + 2 * size, size * numFunc, ind, result, local, subbias, wi2, NULL, -1, dim);

	__syncthreads();
	unit1 = pop[ind + size * threadIdx.y] - d_shift[threadIdx.y + 3 * MAX_DIM];
	local[threadIdx.x + blockDim.x * threadIdx.y] = unit1 * unit1;
	__syncthreads();
	parallelsum(local, &result[threadIdx.x], blockDim.y);
	if (threadIdx.y == 0)
	{
		unit2 = result[threadIdx.x];
		unit3 = unit2 / (double)(2 * dim * 1600.0);
		wi[threadIdx.x] = 1 / sqrt(unit2) * exp(-unit3);
		sumWi[threadIdx.x] += wi[threadIdx.x];
	}
	__syncthreads();
	if (threadIdx.y == 0)
	{
		wi2[threadIdx.x] = 5e-4 * wi[threadIdx.x];
		subbias = 300.0 * wi[threadIdx.x];
	}
	f14(fitnessValue, rotpop + 3 * size, size * numFunc, ind, result, local, subbias, wi2, NULL, -1, dim);

	__syncthreads();
	unit1 = pop[ind + size * threadIdx.y] - d_shift[threadIdx.y + 4 * MAX_DIM];
	local[threadIdx.x + blockDim.x * threadIdx.y] = unit1 * unit1;
	__syncthreads();
	parallelsum(local, &result[threadIdx.x], blockDim.y);
	if (threadIdx.y == 0)
	{
		unit2 = result[threadIdx.x];
		unit3 = unit2 / (double)(2 * dim * 2500.0);
		wi[threadIdx.x] = 1 / sqrt(unit2) * exp(-unit3);
		sumWi[threadIdx.x] += wi[threadIdx.x];
	}
	__syncthreads();
	if (threadIdx.y == 0)
	{
		wi2[threadIdx.x] = 1e-6 * wi[threadIdx.x];
		subbias = 400.0 * wi[threadIdx.x];
	}
	f1(fitnessValue, rotpop + 4 * size, size * numFunc, ind, result, local, subbias, wi2, NULL, -1, dim);
	if (threadIdx.y == 0)
		fitnessValue[ind] = fitnessValue[ind] / sumWi[threadIdx.x] + bias;
}
__global__ void global_f29(real* fitnessValue, real * pop, real * rotpop, size_t size, int *d_shuffle, int dim)
{
	real subbias;
	real bias = 2900;
	int numFunc = 3;
	real * result = shared;
	real * wi = result + blockDim.x;
	real * wi2 = wi + blockDim.x;
	real * sumWi = wi2 + blockDim.x;
	real * tmpValue = sumWi + blockDim.x;
	real * local = tmpValue + blockDim.x;
	int ind = threadIdx.x + blockIdx.x * blockDim.x;

	real unit1, unit2, unit3, unit4;
	if (threadIdx.y == 0)
	{
		sumWi[threadIdx.x] = 0;
		fitnessValue[ind] = 0;
		tmpValue[threadIdx.x] = 0;
	}
	//shift data for calWi;
	unit1 = pop[ind + size * threadIdx.y] - d_shift[threadIdx.y];
	local[threadIdx.x + blockDim.x * threadIdx.y] = unit1 * unit1;
	__syncthreads();
	parallelsum(local, &result[threadIdx.x], blockDim.y);
	if (threadIdx.y == 0)
	{
		unit2 = result[threadIdx.x];
		unit3 = unit2 / (double)(2 * dim * 100.0);
		wi[threadIdx.x] = 1 / sqrt(unit2) * exp(-unit3);
		sumWi[threadIdx.x] += wi[threadIdx.x];
	}
	__syncthreads();

	if (threadIdx.y == 0)
	{
		wi2[threadIdx.x] = wi[threadIdx.x];
		subbias = 0;
	}
	hybirdFunc1(fitnessValue, rotpop, tmpValue, size * numFunc, ind, result, local, subbias, wi2, d_shuffle, dim);
	
	__syncthreads();
	unit1 = pop[ind + size * threadIdx.y] - d_shift[threadIdx.y + MAX_DIM];
	local[threadIdx.x + blockDim.x * threadIdx.y] = unit1 * unit1;
	__syncthreads();
	parallelsum(local, &result[threadIdx.x], blockDim.y);
	if (threadIdx.y == 0)
	{
		unit2 = result[threadIdx.x];
		unit3 = unit2 / (double)(2 * dim * 900.0);
		wi[threadIdx.x] = 1 / sqrt(unit2) * exp(-unit3);
		sumWi[threadIdx.x] += wi[threadIdx.x];
	}
	__syncthreads();


	if (threadIdx.y == 0)
	{
		wi2[threadIdx.x] = wi[threadIdx.x];
		subbias = 100.0 * wi[threadIdx.x];
	}
	__syncthreads();
	hybirdFunc2(fitnessValue, rotpop + size * 1, tmpValue, size * numFunc, ind, result, local, subbias, wi2, d_shuffle + dim, dim);


	__syncthreads();
	unit1 = pop[ind + size * threadIdx.y] - d_shift[threadIdx.y + 2 * MAX_DIM];
	local[threadIdx.x + blockDim.x * threadIdx.y] = unit1 * unit1;
	__syncthreads();
	parallelsum(local, &result[threadIdx.x], blockDim.y);
	if (threadIdx.y == 0)
	{
		unit2 = result[threadIdx.x];
		unit3 = unit2 / (double)(2 * dim * 2500.0);
		wi[threadIdx.x] = 1 / sqrt(unit2) * exp(-unit3);
		sumWi[threadIdx.x] += wi[threadIdx.x];
	}
	__syncthreads();
	if (threadIdx.y == 0)
	{
		wi2[threadIdx.x] = wi[threadIdx.x];
		subbias = 200.0 * wi[threadIdx.x];

	}
	hybirdFunc3(fitnessValue, rotpop + size * 2, tmpValue, size * numFunc, ind, result, local, subbias, wi2, d_shuffle + dim * 2, dim);

	__syncthreads();
	
	if (threadIdx.y == 0)
		fitnessValue[ind] = tmpValue[threadIdx.x] / sumWi[threadIdx.x] + bias;
}
__global__ void global_f30(real* fitnessValue, real * pop, real * rotpop, size_t size, int *d_shuffle, int dim)
{
	real subbias;
	real bias = 3000;
	int numFunc = 3;
	real * result = shared;
	real * wi = result + blockDim.x;
	real * wi2 = wi + blockDim.x;
	real * sumWi = wi2 + blockDim.x;
	real * tmpValue = sumWi + blockDim.x;
	real * local = tmpValue + blockDim.x;
	int ind = threadIdx.x + blockIdx.x * blockDim.x;

	real unit1, unit2, unit3, unit4;
	if (threadIdx.y == 0)
	{
		sumWi[threadIdx.x] = 0;
		fitnessValue[ind] = 0;
		tmpValue[threadIdx.x] = 0;
	}
	//shift data for calWi;
	unit1 = pop[ind + size * threadIdx.y] - d_shift[threadIdx.y];
	local[threadIdx.x + blockDim.x * threadIdx.y] = unit1 * unit1;
	__syncthreads();
	parallelsum(local, &result[threadIdx.x], blockDim.y);
	if (threadIdx.y == 0)
	{
		unit2 = result[threadIdx.x];
		unit3 = unit2 / (double)(2 * dim * 100.0);
		wi[threadIdx.x] = 1 / sqrt(unit2) * exp(-unit3);
		sumWi[threadIdx.x] += wi[threadIdx.x];
	}
	__syncthreads();

	if (threadIdx.y == 0)
	{
		wi2[threadIdx.x] = wi[threadIdx.x];
		subbias = 0;
	}
	hybirdFunc4(fitnessValue, rotpop, tmpValue, size * numFunc, ind, result, local, subbias, wi2, d_shuffle, dim);

	__syncthreads();
	unit1 = pop[ind + size * threadIdx.y] - d_shift[threadIdx.y + MAX_DIM];
	local[threadIdx.x + blockDim.x * threadIdx.y] = unit1 * unit1;
	__syncthreads();
	parallelsum(local, &result[threadIdx.x], blockDim.y);
	if (threadIdx.y == 0)
	{
		unit2 = result[threadIdx.x];
		unit3 = unit2 / (double)(2 * dim * 900.0);
		wi[threadIdx.x] = 1 / sqrt(unit2) * exp(-unit3);
		sumWi[threadIdx.x] += wi[threadIdx.x];
	}
	__syncthreads();


	if (threadIdx.y == 0)
	{
		wi2[threadIdx.x] = wi[threadIdx.x];
		subbias = 100.0 * wi[threadIdx.x];
	}
	__syncthreads();
	hybirdFunc5(fitnessValue, rotpop + size * 1, tmpValue, size * numFunc, ind, result, local, subbias, wi2, d_shuffle + dim, dim);


	__syncthreads();
	unit1 = pop[ind + size * threadIdx.y] - d_shift[threadIdx.y + 2 * MAX_DIM];
	local[threadIdx.x + blockDim.x * threadIdx.y] = unit1 * unit1;
	__syncthreads();
	parallelsum(local, &result[threadIdx.x], blockDim.y);
	if (threadIdx.y == 0)
	{
		unit2 = result[threadIdx.x];
		unit3 = unit2 / (double)(2 * dim * 2500.0);
		wi[threadIdx.x] = 1 / sqrt(unit2) * exp(-unit3);
		sumWi[threadIdx.x] += wi[threadIdx.x];
	}
	__syncthreads();
	if (threadIdx.y == 0)
	{
		wi2[threadIdx.x] = wi[threadIdx.x];
		subbias = 200.0 * wi[threadIdx.x];

	}
	hybirdFunc6(fitnessValue, rotpop + size * 2, tmpValue, size * numFunc, ind, result, local, subbias, wi2, d_shuffle + dim * 2, dim);

	__syncthreads();

	if (threadIdx.y == 0)
		fitnessValue[ind] = tmpValue[threadIdx.x] / sumWi[threadIdx.x] + bias;
}

//__global__ void global_f4(real* fitnessValue, real * pop, size_t size, real* result, int ind, real* local, real bias, int *d_shuffle, int startDim, int lengthDim, int sharedMemStart)
//{
//	f4(fitnessValue, pop, size, result, ind, local, bias, d_shuffle, startDim, lengthDim, sharedMemStart);
//}
//__global__ void global_f5(real* fitnessValue, real * pop, size_t size, real* result, int ind, real* local, real bias, int *d_shuffle, int startDim, int lengthDim, int sharedMemStart)
//{
//	f5(fitnessValue, pop, size, result, ind, local, bias, d_shuffle, startDim, lengthDim, sharedMemStart);
//}
//__global__ void global_f6(real* fitnessValue, real * pop, size_t size, real* result, int ind, real* local, real bias, int *d_shuffle, int startDim, int lengthDim, int sharedMemStart)
//{
//	f6(fitnessValue, pop, size, result, ind, local, bias, d_shuffle, startDim, lengthDim, sharedMemStart);
//}
//__global__ void global_f7(real* fitnessValue, real * pop, size_t size, real* result, int ind, real* local, real bias, int *d_shuffle, int startDim, int lengthDim, int sharedMemStart)
//{
//	f7(fitnessValue, pop, size, result, ind, local, bias, d_shuffle, startDim, lengthDim, sharedMemStart);
//}
//__global__ void global_f8(real* fitnessValue, real * pop, size_t size, real* result, int ind, real* local, real bias, int *d_shuffle, int startDim, int lengthDim, int sharedMemStart)
//{
//	f8(fitnessValue, pop, size, result, ind, local, bias, d_shuffle, startDim, lengthDim, sharedMemStart);
//}
//__global__ void global_f9(real* fitnessValue, real * pop, size_t size, real* result, int ind, real* local, real bias, int *d_shuffle, int startDim, int lengthDim, int sharedMemStart)
//{
//	f8(fitnessValue, pop, size, result, ind, local, bias, d_shuffle, startDim, lengthDim, sharedMemStart);
//}
//__global__ void global_f10(real* fitnessValue, real * pop, size_t size, real* result, int ind, real* local, real bias, int *d_shuffle, int startDim, int lengthDim, int sharedMemStart)
//{
//	f9(fitnessValue, pop, size, result, ind, local, bias, d_shuffle, startDim, lengthDim, sharedMemStart);
//}
//__global__ void global_f11(real* fitnessValue, real * pop, size_t size, real* result, int ind, real* local, real bias, int *d_shuffle, int startDim, int lengthDim, int sharedMemStart)
//{
//	f9(fitnessValue, pop, size, result, ind, local, bias, d_shuffle, startDim, lengthDim, sharedMemStart);
//}
//__global__ void global_f12(real* fitnessValue, real * pop, size_t size, real* result, int ind, real* local, real bias, int *d_shuffle, int startDim, int lengthDim, int sharedMemStart)
//{
//	f10(fitnessValue, pop, size, result, ind, local, bias, d_shuffle, startDim, lengthDim, sharedMemStart);
//}
//__global__ void global_f13(real* fitnessValue, real * pop, size_t size, real* result, int ind, real* local, real bias, int *d_shuffle, int startDim, int lengthDim, int sharedMemStart)
//{
//	f11(fitnessValue, pop, size, result, ind, local, bias, d_shuffle, startDim, lengthDim, sharedMemStart);
//}
//__global__ void global_f14(real* fitnessValue, real * pop, size_t size, real* result, int ind, real* local, real bias, int *d_shuffle, int startDim, int lengthDim, int sharedMemStart)
//{
//	f12(fitnessValue, pop, size, result, ind, local, bias, d_shuffle, startDim, lengthDim, sharedMemStart);
//}
//__global__ void global_f15(real* fitnessValue, real * pop, size_t size, real* result, int ind, real* local, real bias, int *d_shuffle, int startDim, int lengthDim, int sharedMemStart)
//{
//	f13(fitnessValue, pop, size, result, ind, local, bias, d_shuffle, startDim, lengthDim, sharedMemStart);
//}
//__global__ void global_f16(real* fitnessValue, real * pop, size_t size, real* result, int ind, real* local, real bias, int *d_shuffle, int startDim, int lengthDim, int sharedMemStart)
//{
//	f14(fitnessValue, pop, size, result, ind, local, bias, d_shuffle, startDim, lengthDim, sharedMemStart);
//}
//__global__ void global_f17(real* fitnessValue, real * pop, size_t size, real* result, int ind, real* local, real bias, int *d_shuffle, int startDim, int lengthDim, int sharedMemStart)
//{
//	hybirdFunc1(fitnessValue, pop, size, result, ind, local, bias, d_shuffle, startDim, lengthDim, sharedMemStart);
//}
//__global__ void global_f18(real* fitnessValue, real * pop, size_t size, real* result, int ind, real* local, real bias, int *d_shuffle, int startDim, int lengthDim, int sharedMemStart)
//{
//	hybirdFunc2(fitnessValue, pop, size, result, ind, local, bias, d_shuffle, startDim, lengthDim, sharedMemStart);
//}
//__global__ void global_f19(real* fitnessValue, real * pop, size_t size, real* result, int ind, real* local, real bias, int *d_shuffle, int startDim, int lengthDim, int sharedMemStart)
//{
//	hybirdFunc3(fitnessValue, pop, size, result, ind, local, bias, d_shuffle, startDim, lengthDim, sharedMemStart);
//}
//__global__ void global_f20(real* fitnessValue, real * pop, size_t size, real* result, int ind, real* local, real bias, int *d_shuffle, int startDim, int lengthDim, int sharedMemStart)
//{
//	hybirdFunc4(fitnessValue, pop, size, result, ind, local, bias, d_shuffle, startDim, lengthDim, sharedMemStart);
//}
//__global__ void global_f21(real* fitnessValue, real * pop, size_t size, real* result, int ind, real* local, real bias, int *d_shuffle, int startDim, int lengthDim, int sharedMemStart)
//{
//	hybirdFunc5(fitnessValue, pop, size, result, ind, local, bias, d_shuffle, startDim, lengthDim, sharedMemStart);
//}
//__global__ void global_f22(real* fitnessValue, real * pop, size_t size, real* result, int ind, real* local, real bias, int *d_shuffle, int startDim, int lengthDim, int sharedMemStart)
//{
//	hybirdFunc6(fitnessValue, pop, size, result, ind, local, bias, d_shuffle, startDim, lengthDim, sharedMemStart);
//}
//__global__ void global_f23(real* fitnessValue, real * pop, real * rotpop, real * tmppop, ConfigEval funcconfig, size_t size, int dim, int ind, real bias)
//{
//	int funcList[5] = { 4, 1, 2, 3, 1 };
//	real lambda[5] = { 1, 1e-6, 1e-26, 1e-6, 1e-6 };
//	real subbias[5] = { 0, 100, 200, 300, 400 };
//	real tmp = 0;;
//	__shared__ real w[MAX_BLOCK_X];
//	__shared__ real sumW[MAX_BLOCK_X];
//	int ind = threadIdx.x + blockIdx.x * blockDim.x;
//	if (threadIdx.y == 0)
//	{
//		sumW[threadIdx.x] = 0;
//		fitnessValue[ind] = 0;
//	}
//	tmppop[ind + size * threadIdx.y] = funcconfig.rate * (pop[ind + size * threadIdx.y] - d_shift[threadIdx.y]);
//	__syncthreads();
//	calW(w, sumW, tmppop, size, ind, dim, 10.0);
//	__syncthreads();
//
//	f4(fitnessValue, rotpop, size, ind, 0, NULL, -1, dim, w[threadIdx.x], 0);
//
//	f1(fitnessValue, rotpop, size, ind, 0, NULL, -1, dim, w[threadIdx.x] * 1e-6, 100);
//
//	f2(fitnessValue, rotpop, size, ind, 0, NULL, -1, dim, w[threadIdx.x] * 1e-26, 200);
//
//	f3(fitnessValue, rotpop, size, ind, 0, NULL, -1, dim, w[threadIdx.x] * 1e-6, 300);
//
//	f1(fitnessValue, rotpop, size, ind, 0, NULL, -1, dim, w[threadIdx.x] * 1e-6, 400);
//
//}
//__global__ void global_f24(real* fitnessValue, real * pop, size_t size, real* result, int ind, real* local, real bias, int *d_shuffle, int startDim, int lengthDim, int sharedMemStart)
//{
//	f1(fitnessValue, pop, size, result, ind, local, bias, d_shuffle, startDim, lengthDim, sharedMemStart);
//}
//__global__ void global_f25(real* fitnessValue, real * pop, size_t size, real* result, int ind, real* local, real bias, int *d_shuffle, int startDim, int lengthDim, int sharedMemStart)
//{
//	f1(fitnessValue, pop, size, result, ind, local, bias, d_shuffle, startDim, lengthDim, sharedMemStart);
//}
//__global__ void global_f26(real* fitnessValue, real * pop, size_t size, real* result, int ind, real* local, real bias, int *d_shuffle, int startDim, int lengthDim, int sharedMemStart)
//{
//	f1(fitnessValue, pop, size, result,ind, local, bias, d_shuffle, startDim, lengthDim, sharedMemStart);
//}
//__global__ void global_f27(real* fitnessValue, real * pop, size_t size, real* result, int ind, real* local, real bias, int *d_shuffle, int startDim, int lengthDim, int sharedMemStart)
//{
//	f1(fitnessValue, pop, size, result, ind, local, bias, d_shuffle, startDim, lengthDim, sharedMemStart);
//}
//__global__ void global_f28(real* fitnessValue, real * pop, size_t size, real* result, int ind, real* local, real bias, int *d_shuffle, int startDim, int lengthDim, int sharedMemStart)
//{
//	f1(fitnessValue, pop, size, result, ind, local, bias, d_shuffle, startDim, lengthDim, sharedMemStart);
//}
//__global__ void global_f29(real* fitnessValue, real * pop, size_t size, real* result, int ind, real* local, real bias, int *d_shuffle, int startDim, int lengthDim, int sharedMemStart)
//{
//	f1(fitnessValue, pop, size, result, ind, local, bias, d_shuffle, startDim, lengthDim, sharedMemStart);
//}
//__global__ void global_f30(real* fitnessValue, real * pop, size_t size, real* result, int ind, real* local, real bias, int *d_shuffle, int startDim, int lengthDim, int sharedMemStart)
//{
//	f1(fitnessValue, pop, size, result, ind, local, bias, d_shuffle, startDim, lengthDim, sharedMemStart);
//}

void evaluateFitness(population *pop, population* rotpop, ConfigEval configEval, ConfigDE configDE, ConfigCUDA configCUDA, cudaStream_t stream)
{

//	cudaShiftRotation(rotpop->d_pop, pop->d_pop, funcconfig.d_M, funcconfig.fixedshift, funcconfig.rate, pop->size, config.i_dim, 0);
//	global_f23 << <kernelconfig.blocks, kernelconfig.threads, 48000 >> >(pop->d_fval, rotpop->d_pop, pop->size, funcconfig.d_shuffle, config.i_dim);
	
	cudaContinueRotation(rotpop->d_pop, pop->d_pop, configEval.d_M, configEval.fixedshift, configEval.rate, pop->size, configDE.i_dim, configEval.numfunc);
//	HANDLE_CUDA_ERROR(cudaMemcpy(rotpop->h_pop, rotpop->d_pop, pop->size * sizeof(real) * pop->dim * funcconfig.numfunc, cudaMemcpyDeviceToHost));
	switch (configDE.i_func)
	{
	case(1) :
		global_f1 << <kernelconfig.blocks, kernelconfig.threads, 48000 >> >(pop->d_fval, rotpop->d_pop, pop->size, funcconfig.d_shuffle, config.i_dim);
		break;
	case(2) :
		global_f2 << <kernelconfig.blocks, kernelconfig.threads, 48000 >> >(pop->d_fval, rotpop->d_pop, pop->size, funcconfig.d_shuffle, config.i_dim);
		break;
	case(3) :
		global_f3 << <kernelconfig.blocks, kernelconfig.threads, 48000 >> >(pop->d_fval, rotpop->d_pop, pop->size, funcconfig.d_shuffle, config.i_dim);
		break;
	case(4) :
		global_f4 << <kernelconfig.blocks, kernelconfig.threads, 48000 >> >(pop->d_fval, rotpop->d_pop, pop->size, funcconfig.d_shuffle, config.i_dim);
		break;
	case(5) :
		global_f5 << <kernelconfig.blocks, kernelconfig.threads, 48000 >> >(pop->d_fval, rotpop->d_pop, pop->size, funcconfig.d_shuffle, config.i_dim);
		break;
	case(6) :
		global_f6 << <kernelconfig.blocks, kernelconfig.threads, 48000 >> >(pop->d_fval, rotpop->d_pop, pop->size, funcconfig.d_shuffle, config.i_dim);
		break;
	case(7) :
		global_f7 << <kernelconfig.blocks, kernelconfig.threads, 48000 >> >(pop->d_fval, rotpop->d_pop, pop->size, funcconfig.d_shuffle, config.i_dim);
		break;
	case(8) :
		global_f8 << <kernelconfig.blocks, kernelconfig.threads, 48000 >> >(pop->d_fval, rotpop->d_pop, pop->size, funcconfig.d_shuffle, config.i_dim);
		break;
	case(9) :
		global_f9 << <kernelconfig.blocks, kernelconfig.threads, 48000 >> >(pop->d_fval, rotpop->d_pop, pop->size, funcconfig.d_shuffle, config.i_dim);
		break;
	case(10) :
		global_f10 << <kernelconfig.blocks, kernelconfig.threads, 48000 >> >(pop->d_fval, rotpop->d_pop, pop->size, funcconfig.d_shuffle, config.i_dim);
		break;
	case(11) :
		global_f11 << <kernelconfig.blocks, kernelconfig.threads, 48000 >> >(pop->d_fval, rotpop->d_pop, pop->size, funcconfig.d_shuffle, config.i_dim);
		break;
	case(12) :
		global_f12 << <kernelconfig.blocks, kernelconfig.threads, 48000 >> >(pop->d_fval, rotpop->d_pop, pop->size, funcconfig.d_shuffle, config.i_dim);
		break;
	case(13) :
		global_f13 << <kernelconfig.blocks, kernelconfig.threads, 48000 >> >(pop->d_fval, rotpop->d_pop, pop->size, funcconfig.d_shuffle, config.i_dim);
		break;
	case(14) :
		global_f14 << <kernelconfig.blocks, kernelconfig.threads, 48000 >> >(pop->d_fval, rotpop->d_pop, pop->size, funcconfig.d_shuffle, config.i_dim);
		break;
	case(15) :
		global_f15 << <kernelconfig.blocks, kernelconfig.threads, 48000 >> >(pop->d_fval, rotpop->d_pop, pop->size, funcconfig.d_shuffle, config.i_dim);
		break;
	case(16) :
		global_f16 << <kernelconfig.blocks, kernelconfig.threads, 48000 >> >(pop->d_fval, rotpop->d_pop, pop->size, funcconfig.d_shuffle, config.i_dim);
		break;
	case(17) :
		global_f17 << <kernelconfig.blocks, kernelconfig.threads, 48000 >> >(pop->d_fval, rotpop->d_pop, pop->size, funcconfig.d_shuffle, config.i_dim);
		break;
	case(18) :
		global_f18 << <kernelconfig.blocks, kernelconfig.threads, 48000 >> >(pop->d_fval, rotpop->d_pop, pop->size, funcconfig.d_shuffle, config.i_dim);
		break;
	case(19) :
		global_f19 << <kernelconfig.blocks, kernelconfig.threads, 48000 >> >(pop->d_fval, rotpop->d_pop, pop->size, funcconfig.d_shuffle, config.i_dim);
		break;
	case(20) :
		global_f20 << <kernelconfig.blocks, kernelconfig.threads, 48000 >> >(pop->d_fval, rotpop->d_pop, pop->size, funcconfig.d_shuffle, config.i_dim);
		break;
	case(21) :
		global_f21 << <kernelconfig.blocks, kernelconfig.threads, 48000 >> >(pop->d_fval, rotpop->d_pop, pop->size, funcconfig.d_shuffle, config.i_dim);
		break;
	case(22) :
		global_f22 << <kernelconfig.blocks, kernelconfig.threads, 48000 >> >(pop->d_fval, rotpop->d_pop, pop->size, funcconfig.d_shuffle, config.i_dim);
		break;
	case(23) :
		global_f23 << <kernelconfig.blocks, kernelconfig.threads, 48000 >> >(pop->d_fval, pop->d_pop, rotpop->d_pop, pop->size, config.i_dim);
		break;
	case(24) :
		global_f24 << <kernelconfig.blocks, kernelconfig.threads, 48000 >> >(pop->d_fval, pop->d_pop, rotpop->d_pop, pop->size, config.i_dim);
		break;
	case(25) :
		global_f25 << <kernelconfig.blocks, kernelconfig.threads, 48000 >> >(pop->d_fval, pop->d_pop, rotpop->d_pop, pop->size, config.i_dim);
		break;
	case(26) :
		global_f26 << <kernelconfig.blocks, kernelconfig.threads, 48000 >> >(pop->d_fval, pop->d_pop, rotpop->d_pop, pop->size, config.i_dim);		
		break;
	case(27) :
		global_f27 << <kernelconfig.blocks, kernelconfig.threads, 48000 >> >(pop->d_fval, pop->d_pop, rotpop->d_pop, pop->size, config.i_dim);		
		break;
	case(28) :
		global_f28 << <kernelconfig.blocks, kernelconfig.threads, 48000 >> >(pop->d_fval, pop->d_pop, rotpop->d_pop, pop->size, config.i_dim);
		break;
	case(29) :
		global_f29 << <kernelconfig.blocks, kernelconfig.threads, 48000 >> >(pop->d_fval, pop->d_pop, rotpop->d_pop, pop->size, funcconfig.d_shuffle, config.i_dim);
		break;
	case(30) :
		global_f30 << <kernelconfig.blocks, kernelconfig.threads, 48000 >> >(pop->d_fval, pop->d_pop, rotpop->d_pop, pop->size, funcconfig.d_shuffle, config.i_dim);
		break;
	default:
		break;
	}
//	global_f29 << <kernelconfig.blocks, kernelconfig.threads, 48000 >> >(pop->d_fval, pop->d_pop, rotpop->d_pop, pop->size, config.i_dim);
//		HANDLE_CUDA_ERROR(cudaMemcpy(rotpop->h_pop, rotpop->d_pop, pop->size * sizeof(real) * pop->dim * funcconfig.numfunc, cudaMemcpyDeviceToHost));
//	loadfromdev(pop);

//	global_f18 << <kernelconfig.blocks, kernelconfig.threads, 48000 >> >(pop->d_fval, rotpop->d_pop, pop->size, funcconfig.d_shuffle, config.i_dim);

//	loadfromdev(pop);

};
#endif
