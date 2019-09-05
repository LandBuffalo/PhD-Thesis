#include "../include/config.h"
#include "device_launch_parameters.h"


static __device__ __forceinline__  void device_parallelsum(double * vector, double* result, int lengthSum)
{
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
		double sum = vector[threadIdx.x];
		for (int i = 1; i < olds; i++) {
			//~ if (blockIdx.x == 0 && threadIdx.x ==0) printf("T %d I %d OLDS %d V %f\n", threadIdx.y, i, olds, vector[threadIdx.x + blockDim.x * i]);
			sum += vector[threadIdx.x + blockDim.x * i];
		}
		*result = sum;
	}

	__syncthreads();

};

static __device__ __forceinline__  void device_parallelmultiple(double * vector, double* result, int lengthSum) {
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
		double sum = vector[threadIdx.x];
		for (int i = 1; i < olds; i++) {
			//~ if (blockIdx.x == 0 && threadIdx.x ==0) printf("T %d I %d OLDS %d V %f\n", threadIdx.y, i, olds, vector[threadIdx.x + blockDim.x * i]);
			sum *= vector[threadIdx.x + blockDim.x * i];
		}
		*result = sum;
	}

	__syncthreads();

};

__global__ void global_matrixMultiply_comp_function(double * A, double * B, double * C, int num_A_rows, int num_A_columns, int num_B_rows, int num_B_columns, int num_C_rows, int num_C_columns, double * d_shift, double shift, double rate, int num_comp_func)
{
	//@@ Insert code to implement matrix multiplication here
	__shared__ double ds_M[TILE_WIDTH][TILE_WIDTH];
	__shared__ double ds_N[TILE_WIDTH][TILE_WIDTH];

	int bx = blockIdx.x, by = blockIdx.y,
		tx = threadIdx.x, ty = threadIdx.y,
		row = by * TILE_WIDTH + ty,
		col = bx * TILE_WIDTH + tx;

	for (int repeat = 0; repeat < num_comp_func; repeat++)
	{
		double Pvalue = 0;

		for (int m = 0; m < (num_A_columns - 1) / TILE_WIDTH + 1; ++m) {
			if (row < num_A_rows && m*TILE_WIDTH + tx < num_A_columns)
				ds_M[ty][tx] = A[repeat * num_A_rows * num_A_columns + row*num_A_columns + m*TILE_WIDTH + tx];
			else
				ds_M[ty][tx] = 0;
			if (col < num_B_columns && m*TILE_WIDTH + ty < num_B_rows)
				ds_N[ty][tx] = rate * (B[(m*TILE_WIDTH + ty)*num_B_columns + col] - d_shift[m*TILE_WIDTH + ty + repeat * MAX_DIM]);
			else
				ds_N[ty][tx] = 0;
			__syncthreads();
			for (int k = 0; k < TILE_WIDTH; ++k)
				Pvalue += ds_M[ty][k] * ds_N[k][tx];
			__syncthreads();
		}
		if (row < num_C_rows && col < num_C_columns)
			C[row * (num_C_columns * num_comp_func) + col + repeat * num_C_columns] = Pvalue + shift;
		__syncthreads();
	}

}


__device__ void device_f1(double* value_fitness, double * pop, size_t size, int ind, double* result, double* local, double bias, double *wi, int *d_shuffle, int startDim, int lengthDim)
{
	double var;
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

	device_parallelsum(local, &result[threadIdx.x], blockDim.y);
	if (threadIdx.y == 0)
		value_fitness[ind] += wi[threadIdx.x] * result[threadIdx.x] + bias;
};

__device__ void device_f2(double* value_fitness, double * pop, size_t size, int ind, double* result, double* local, double bias, double *wi, int *d_shuffle, int startDim, int lengthDim)
{
	double var, unit1;
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
	device_parallelsum(local, &result[threadIdx.x], blockDim.y);

	if (threadIdx.y == 0)
		value_fitness[ind] += wi[threadIdx.x] * (1e6 * result[threadIdx.x] + unit1 * unit1) + bias;
};

__device__ void device_f3(double* value_fitness, double * pop, size_t size, int ind, double* result, double* local, double bias, double *wi, int *d_shuffle, int startDim, int lengthDim)
{
	double var, unit1;
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
	device_parallelsum(local, &result[threadIdx.x], lengthDim);
	if (threadIdx.y == 0)
		value_fitness[ind] += wi[threadIdx.x] * (result[threadIdx.x] + (1e6 - 1) * unit1 * unit1) + bias;
};

__device__ void device_f4(double* value_fitness, double * pop, size_t size, int ind, double* result, double* local, double bias, double *wi, int *d_shuffle, int startDim, int lengthDim)
{
	double var1 = 0, var2 = 0, unit1 = 0, unit2 = 0, unit3 = 0;
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
	device_parallelsum(local, &result[threadIdx.x], blockDim.y);
	if (threadIdx.y == 0)
		value_fitness[ind] += wi[threadIdx.x] * result[threadIdx.x] + bias;

};

__device__ void device_f5(double* value_fitness, double * pop, size_t size, int ind, double* result, double* local, double bias, double *wi, int *d_shuffle, int startDim, int lengthDim)
{
	double var, unit1 = 0, unit2 = 0, unit3, tmp;
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
	device_parallelsum(local, &result[threadIdx.x], blockDim.y);
	if (threadIdx.y == 0)
	{
		unit2 = result[threadIdx.x] / (double)lengthDim;
		unit3 = exp(unit2);
	}
	device_parallelsum(local + blockDim.x * blockDim.y, &result[threadIdx.x], blockDim.y);
	if (threadIdx.y == 0)
	{
		unit1 = -20 * exp(sqrt(result[threadIdx.x] / (double)lengthDim) * (-0.2));
		value_fitness[ind] += wi[threadIdx.x] * (unit1 - unit3 + 20 + M_E) + bias;
	}
};

__device__ void device_f6(double* value_fitness, double * pop, size_t size, int ind, double* result, double* local, double bias, double *wi, int *d_shuffle, int startDim, int lengthDim)
{
	double var, unit1, unit2, unit3;
	int shuffleInd;
	double a = 0.5;
	double b = 3;
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
	device_parallelsum(local, &result[threadIdx.x], blockDim.y);
	if (threadIdx.y == 0)
		value_fitness[ind] += wi[threadIdx.x] * (result[threadIdx.x] - unit3 * lengthDim) + bias;

};

__device__ void device_f7(double* value_fitness, double * pop, size_t size, int ind, double* result, double* local, double bias, double *wi, int *d_shuffle, int startDim, int lengthDim)
{
	double var, unit1, unit2;
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
	device_parallelsum(local, &result[threadIdx.x], blockDim.y);
	if (threadIdx.y == 0)
		unit1 = result[threadIdx.x];
	device_parallelmultiple(local + blockDim.x *  blockDim.y, &result[threadIdx.x], blockDim.y);


	if (threadIdx.y == 0)
	{
		unit2 = result[threadIdx.x];
		value_fitness[ind] += wi[threadIdx.x] * (unit1 - unit2 + 1) + bias;
	}
};

__device__ void device_f8(double* value_fitness, double * pop, size_t size, int ind, double* result, double* local, double bias, double *wi, int *d_shuffle, int startDim, int lengthDim)
{
	double var, unit1;
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
	device_parallelsum(local, &result[threadIdx.x], blockDim.y);
	if (threadIdx.y == 0)
		value_fitness[ind] += wi[threadIdx.x] * result[threadIdx.x] + bias;
};

__device__ void device_f9(double* value_fitness, double * pop, size_t size, int ind, double* result, double* local, double bias, double *wi, int *d_shuffle, int startDim, int lengthDim)
{
	double var, zvar, unit1;
	double sinValue = 0, cosValue = 0;
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
	device_parallelsum(local, &result[threadIdx.x], blockDim.y);
	if (threadIdx.y == 0)
		value_fitness[ind] += wi[threadIdx.x] * (4.189828872724338e+002 * lengthDim - result[threadIdx.x]) + bias;
};

__device__ void device_f10(double* value_fitness, double * pop, size_t size, int ind, double* result, double* local, double bias, double *wi, int *d_shuffle, int startDim, int lengthDim)
{
	double var, tmp, unit1, unit2, unit3 = 0;
	int shuffleInd;

	double tmp2 = (double)10 / pow((double)lengthDim, 1.2);
	double tmp3 = (double)10 / pow((double)lengthDim, 2);
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
	device_parallelmultiple(local, &result[threadIdx.x], blockDim.y);
	if (threadIdx.y == 0)
		value_fitness[ind] += wi[threadIdx.x] * (tmp3 * result[threadIdx.x] - tmp3) + bias;
}

__device__ void device_f11(double* value_fitness, double * pop, size_t size, int ind, double* result, double* local, double bias, double *wi, int *d_shuffle, int startDim, int lengthDim)
{
	double var, unit1, unit2, unit3;
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
	device_parallelsum(local, &result[threadIdx.x], blockDim.y);
	if (threadIdx.y == 0)
	{
		unit1 = result[threadIdx.x];
		unit2 = pow(fabs(unit1 - lengthDim), 0.25);
	}
	device_parallelsum(local + blockDim.x * blockDim.y, &result[threadIdx.x], blockDim.y);
	if (threadIdx.y == 0)
	{
		unit3 = (0.5 * unit1 + result[threadIdx.x]) / (double)lengthDim;
		value_fitness[ind] += wi[threadIdx.x] * (unit2 + unit3 + 0.5) + bias;
	}
	//	local[threadIdx.x + blockDim.x * threadIdx.y] = 0;
	//	local[threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y)] = 0;


}

__device__ void device_f12(double* value_fitness, double * pop, size_t size, int ind, double* result, double* local, double bias, double *wi, int *d_shuffle, int startDim, int lengthDim)
{
	double var, unit1, unit2;
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

	device_parallelsum(local, &result[threadIdx.x], blockDim.y);
	if (threadIdx.y == 0)
		unit1 = result[threadIdx.x];
	device_parallelsum(local + blockDim.x * blockDim.y, &result[threadIdx.x], blockDim.y);
	if (threadIdx.y == 0)
	{
		unit2 = result[threadIdx.x];
		value_fitness[ind] += wi[threadIdx.x] * (sqrt(fabs(unit1 * unit1 - unit2 * unit2)) + (0.5 * unit1 + unit2) / (0.0 + lengthDim) + 0.5) + bias;
	}

}

__device__ void device_f13(double* value_fitness, double * pop, size_t size, int ind, double* result, double* local, double bias, double *wi, int *d_shuffle, int startDim, int lengthDim)
{
	double var1, var2, unit1, unit2, unit3, unit4;
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

		device_parallelsum(local, &result[threadIdx.x], blockDim.y);

		if (threadIdx.y == 0)
			value_fitness[ind] += wi[threadIdx.x] * result[threadIdx.x] + bias;
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
			value_fitness[ind] += wi[threadIdx.x] * unit2 + bias;
		}
	}
}

__device__ void device_f14(double* value_fitness, double * pop, size_t size, int ind, double* result, double* local, double bias, double *wi, int *d_shuffle, int startDim, int lengthDim)
{
	double var1, var2, unit1, unit2, unit3 = 0;
	double sinValue = 0, cosValue = 0;
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

		device_parallelsum(local, &result[threadIdx.x], blockDim.y);

		if (threadIdx.y == 0)
			value_fitness[ind] += wi[threadIdx.x] * result[threadIdx.x] + bias;
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

			value_fitness[ind] += wi[threadIdx.x] * unit2 + bias;
		}
	}
}

__device__ void device_hybirdFunc1(double* value_fitness, double * pop, double* tmpValue, size_t size, int ind, double* result, double* local, double bias, double *wi, int *d_shuffle, int dim)
{
	double p[3] = { 0.3, 0.3, 0.4 };
	int lengthDim[3];

	lengthDim[0] = ceil(p[0] * dim);
	lengthDim[1] = ceil(p[1] * dim);
	lengthDim[2] = dim - lengthDim[0] - lengthDim[1];
	if (threadIdx.y == 0)
	{
		value_fitness[ind] = 0;
	}
	int startDim, endDim = 0;
	startDim = endDim;
	endDim = lengthDim[0] + startDim;
	device_f9(value_fitness, pop, size, ind, result, local, 0, wi, d_shuffle, startDim, lengthDim[0]);
	startDim = endDim;
	endDim = lengthDim[1] + startDim;
	device_f8(value_fitness, pop, size, ind, result, local, 0, wi, d_shuffle, startDim, lengthDim[1]);
	startDim = endDim;
	device_f1(value_fitness, pop, size, ind, result, local, 0, wi, d_shuffle, startDim, lengthDim[2]);

	if (threadIdx.y == 0)
		tmpValue[threadIdx.x] += value_fitness[ind] + bias;
}

__device__ void device_hybirdFunc2(double* value_fitness, double * pop, double* tmpValue, size_t size, int ind, double* result, double* local, double bias, double *wi, int *d_shuffle, int dim)
{

	double p[3] = { 0.3, 0.3, 0.4 };
	int lengthDim[3];

	lengthDim[0] = ceil(p[0] * dim);
	lengthDim[1] = ceil(p[1] * dim);
	lengthDim[2] = dim - lengthDim[0] - lengthDim[1];
	if (threadIdx.y == 0)
		value_fitness[ind] = 0;
	int startDim, endDim = 0;
	startDim = endDim;
	endDim = lengthDim[0] + startDim;
	device_f2(value_fitness, pop, size, ind, result, local, 0, wi, d_shuffle, startDim, lengthDim[0]);
	startDim = endDim;
	endDim = lengthDim[1] + startDim;
	device_f12(value_fitness, pop, size, ind, result, local, 0, wi, d_shuffle, startDim, lengthDim[1]);
	startDim = endDim;
	device_f8(value_fitness, pop, size, ind, result, local, 0, wi, d_shuffle, startDim, lengthDim[2]);

	if (threadIdx.y == 0)
		tmpValue[threadIdx.x] += value_fitness[ind] + bias;
}

__device__ void device_hybirdFunc3(double* value_fitness, double * pop, double* tmpValue, size_t size, int ind, double* result, double* local, double bias, double *wi, int *d_shuffle, int dim)
{
	int funcList[4] = { 7, 6, 4, 14 };
	double p[4] = { 0.2, 0.2, 0.3, 0.3 };
	int lengthDim[4];
	lengthDim[0] = ceil(p[0] * dim);
	lengthDim[1] = ceil(p[1] * dim);
	lengthDim[2] = ceil(p[2] * dim);
	lengthDim[3] = dim - lengthDim[0] - lengthDim[1] - lengthDim[2];
	if (threadIdx.y == 0)
		value_fitness[ind] = 0;
	int startDim, endDim = 0;

	startDim = endDim;
	endDim = lengthDim[0] + startDim;
	device_f7(value_fitness, pop, size, ind, result, local, 0, wi, d_shuffle, startDim, lengthDim[0]);
	startDim = endDim;
	endDim = lengthDim[1] + startDim;
	device_f6(value_fitness, pop, size, ind, result, local, 0, wi, d_shuffle, startDim, lengthDim[1]);
	startDim = endDim;
	endDim = lengthDim[2] + startDim;
	device_f4(value_fitness, pop, size, ind, result, local, 0, wi, d_shuffle, startDim, lengthDim[2]);
	startDim = endDim;
	device_f14(value_fitness, pop, size, ind, result, local, 0, wi, d_shuffle, startDim, lengthDim[3]);
	if (threadIdx.y == 0)
		tmpValue[threadIdx.x] += value_fitness[ind] + bias;
}

__device__ void device_hybirdFunc4(double* value_fitness, double * pop, double* tmpValue, size_t size, int ind, double* result, double* local, double bias, double *wi, int *d_shuffle, int dim)
{
	int funcList[4] = { 12, 3, 13, 8 };
	double p[4] = { 0.2, 0.2, 0.3, 0.3 };
	int lengthDim[4];
	lengthDim[0] = ceil(p[0] * dim);
	lengthDim[1] = ceil(p[1] * dim);
	lengthDim[2] = ceil(p[2] * dim);
	lengthDim[3] = dim - lengthDim[0] - lengthDim[1] - lengthDim[2];
	if (threadIdx.y == 0)
		value_fitness[ind] = 0;
	int startDim, endDim = 0;

	startDim = endDim;
	endDim = lengthDim[0] + startDim;
	device_f12(value_fitness, pop, size, ind, result, local, 0, wi, d_shuffle, startDim, lengthDim[0]);
	startDim = endDim;
	endDim = lengthDim[1] + startDim;
	device_f3(value_fitness, pop, size, ind, result, local, 0, wi, d_shuffle, startDim, lengthDim[1]);
	startDim = endDim;
	endDim = lengthDim[2] + startDim;
	device_f13(value_fitness, pop, size, ind, result, local, 0, wi, d_shuffle, startDim, lengthDim[2]);
	startDim = endDim;
	device_f8(value_fitness, pop, size, ind, result, local, 0, wi, d_shuffle, startDim, lengthDim[3]);
	if (threadIdx.y == 0)
		tmpValue[threadIdx.x] += value_fitness[ind] + bias;
}

__device__ void device_hybirdFunc5(double* value_fitness, double * pop, double* tmpValue, size_t size, int ind, double* result, double* local, double bias, double *wi, int *d_shuffle, int dim)
{
	int funcList[5] = { 14, 12, 4, 9, 1 };
	double p[5] = { 0.1, 0.2, 0.2, 0.2, 0.3 };

	int lengthDim[5];
	lengthDim[0] = ceil(p[0] * dim);
	lengthDim[1] = ceil(p[1] * dim);
	lengthDim[2] = ceil(p[2] * dim);
	lengthDim[3] = ceil(p[3] * dim);
	lengthDim[4] = dim - lengthDim[0] - lengthDim[1] - lengthDim[2] - lengthDim[3];
	if (threadIdx.y == 0)
		value_fitness[ind] = 0;
	int startDim, endDim = 0;

	startDim = endDim;
	endDim = lengthDim[0] + startDim;
	device_f14(value_fitness, pop, size, ind, result, local, 0, wi, d_shuffle, startDim, lengthDim[0]);
	startDim = endDim;
	endDim = lengthDim[1] + startDim;
	device_f12(value_fitness, pop, size, ind, result, local, 0, wi, d_shuffle, startDim, lengthDim[1]);
	startDim = endDim;
	endDim = lengthDim[2] + startDim;
	device_f4(value_fitness, pop, size, ind, result, local, 0, wi, d_shuffle, startDim, lengthDim[2]);
	startDim = endDim;
	endDim = lengthDim[3] + startDim;
	device_f9(value_fitness, pop, size, ind, result, local, 0, wi, d_shuffle, startDim, lengthDim[3]);
	startDim = endDim;
	device_f1(value_fitness, pop, size, ind, result, local, 0, wi, d_shuffle, startDim, lengthDim[4]);
	if (threadIdx.y == 0)
		tmpValue[threadIdx.x] += value_fitness[ind] + bias;
}

__device__ void device_hybirdFunc6(double* value_fitness, double * pop, double* tmpValue, size_t size, int ind, double* result, double* local, double bias, double *wi, int *d_shuffle, int dim)
{
	int funcList[5] = { 10, 11, 13, 9, 5 };
	double p[5] = { 0.1, 0.2, 0.2, 0.2, 0.3 };
	int lengthDim[5];
	lengthDim[0] = ceil(p[0] * dim);
	lengthDim[1] = ceil(p[1] * dim);
	lengthDim[2] = ceil(p[2] * dim);
	lengthDim[3] = ceil(p[3] * dim);
	lengthDim[4] = dim - lengthDim[0] - lengthDim[1] - lengthDim[2] - lengthDim[3];
	if (threadIdx.y == 0)
		value_fitness[ind] = 0;
	int startDim = 0;
	int endDim = 0;

	endDim = lengthDim[0] + startDim;
	device_f10(value_fitness, pop, size, ind, result, local, 0, wi, d_shuffle, startDim, lengthDim[0]);
	startDim = endDim;
	endDim = lengthDim[1] + startDim;
	device_f11(value_fitness, pop, size, ind, result, local, 0, wi, d_shuffle, startDim, lengthDim[1]);
	startDim = endDim;
	endDim = lengthDim[2] + startDim;
	device_f13(value_fitness, pop, size, ind, result, local, 0, wi, d_shuffle, startDim, lengthDim[2]);
	startDim = endDim;
	endDim = lengthDim[3] + startDim;
	device_f9(value_fitness, pop, size, ind, result, local, 0, wi, d_shuffle, startDim, lengthDim[3]);
	startDim = endDim;
	device_f5(value_fitness, pop, size, ind, result, local, 0, wi, d_shuffle, startDim, lengthDim[4]);
	if (threadIdx.y == 0)
		tmpValue[threadIdx.x] += value_fitness[ind] + bias;
}

extern __shared__ double shared[];
__global__ void global_f1(double* value_fitness, double * pop, size_t size, int *d_shuffle, int dim)
{
	double bias = 100.0;
	int ind = threadIdx.x + blockIdx.x * blockDim.x;
	double * result = shared;
	double * local = result + blockDim.x;
	double * wi = local + blockDim.x * blockDim.y;

	if (threadIdx.y == 0)
	{
		value_fitness[ind] = 0;
		wi[threadIdx.x] = 1;
	}
	device_f1(value_fitness, pop, size, ind, result, local, bias, wi, NULL, -1, dim);
}

__global__ void global_f2(double* value_fitness, double * pop, size_t size, int *d_shuffle, int dim)
{
	double bias = 200.0;
	int ind = threadIdx.x + blockIdx.x * blockDim.x;
	double * result = shared;
	double * local = result + blockDim.x;
	double * wi = local + blockDim.x * blockDim.y;

	if (threadIdx.y == 0)
	{
		value_fitness[ind] = 0;
		wi[threadIdx.x] = 1;
	}
	device_f2(value_fitness, pop, size, ind, result, local, bias, wi, NULL, -1, dim);
}

__global__ void global_f3(double* value_fitness, double * pop, size_t size, int *d_shuffle, int dim)
{
	double bias = 300.0;
	int ind = threadIdx.x + blockIdx.x * blockDim.x;
	double * result = shared;
	double * local = result + blockDim.x;
	double * wi = local + blockDim.x * blockDim.y;

	if (threadIdx.y == 0)
	{
		value_fitness[ind] = 0;
		wi[threadIdx.x] = 1;
	}
	device_f3(value_fitness, pop, size, ind, result, local, bias, wi, NULL, -1, dim);
}

__global__ void global_f4(double* value_fitness, double * pop, size_t size, int *d_shuffle, int dim)
{
	double bias = 400.0;
	int ind = threadIdx.x + blockIdx.x * blockDim.x;
	double * result = shared;
	double * local = result + blockDim.x;
	double * wi = local + blockDim.x * blockDim.y;

	if (threadIdx.y == 0)
	{
		value_fitness[ind] = 0;
		wi[threadIdx.x] = 1;
	}
	device_f4(value_fitness, pop, size, ind, result, local, bias, wi, NULL, -1, dim);
}

__global__ void global_f5(double* value_fitness, double * pop, size_t size, int *d_shuffle, int dim)
{
	double bias = 500.0;
	int ind = threadIdx.x + blockIdx.x * blockDim.x;
	double * result = shared;
	double * wi = result + blockDim.x;
	double * local = wi + blockDim.x;


	if (threadIdx.y == 0)
	{
		value_fitness[ind] = 0;
		wi[threadIdx.x] = 1;
	}
	device_f5(value_fitness, pop, size, ind, result, local, bias, wi, NULL, -1, dim);
}

__global__ void global_f6(double* value_fitness, double * pop, size_t size, int *d_shuffle, int dim)
{
	double bias = 600.0;
	int ind = threadIdx.x + blockIdx.x * blockDim.x;
	double * result = shared;
	double * local = result + blockDim.x;
	double * wi = local + blockDim.x * blockDim.y;

	if (threadIdx.y == 0)
	{
		value_fitness[ind] = 0;
		wi[threadIdx.x] = 1;
	}
	device_f6(value_fitness, pop, size, ind, result, local, bias, wi, NULL, -1, dim);
}

__global__ void global_f7(double* value_fitness, double * pop, size_t size, int *d_shuffle, int dim)
{
	double bias = 700.0;
	int ind = threadIdx.x + blockIdx.x * blockDim.x;
	double * result = shared;
	double * wi = result + blockDim.x;
	double * local = wi + blockDim.x;

	if (threadIdx.y == 0)
	{
		value_fitness[ind] = 0;
		wi[threadIdx.x] = 1;
	}
	device_f7(value_fitness, pop, size, ind, result, local, bias, wi, NULL, -1, dim);
}

__global__ void global_f8(double* value_fitness, double * pop, size_t size, int *d_shuffle, int dim)
{
	double bias = 800.0;
	int ind = threadIdx.x + blockIdx.x * blockDim.x;
	double * result = shared;
	double * local = result + blockDim.x;
	double * wi = local + blockDim.x * blockDim.y;

	if (threadIdx.y == 0)
	{
		value_fitness[ind] = 0;
		wi[threadIdx.x] = 1;
	}
	device_f8(value_fitness, pop, size, ind, result, local, bias, wi, NULL, -1, dim);
}

__global__ void global_f9(double* value_fitness, double * pop, size_t size, int *d_shuffle, int dim)
{
	double bias = 900.0;
	int ind = threadIdx.x + blockIdx.x * blockDim.x;
	double * result = shared;
	double * local = result + blockDim.x;
	double * wi = local + blockDim.x * blockDim.y;

	if (threadIdx.y == 0)
	{
		value_fitness[ind] = 0;
		wi[threadIdx.x] = 1;
	}
	device_f8(value_fitness, pop, size, ind, result, local, bias, wi, NULL, -1, dim);
}

__global__ void global_f10(double* value_fitness, double * pop, size_t size, int *d_shuffle, int dim)
{
	double bias = 1000.0;
	int ind = threadIdx.x + blockIdx.x * blockDim.x;
	double * result = shared;
	double * local = result + blockDim.x;
	double * wi = local + blockDim.x * blockDim.y;

	if (threadIdx.y == 0)
	{
		value_fitness[ind] = 0;
		wi[threadIdx.x] = 1;
	}
	device_f9(value_fitness, pop, size, ind, result, local, bias, wi, NULL, -1, dim);
}

__global__ void global_f11(double* value_fitness, double * pop, size_t size, int *d_shuffle, int dim)
{
	double bias = 1100.0;
	int ind = threadIdx.x + blockIdx.x * blockDim.x;
	double * result = shared;
	double * local = result + blockDim.x;
	double * wi = local + blockDim.x * blockDim.y;

	if (threadIdx.y == 0)
	{
		value_fitness[ind] = 0;
		wi[threadIdx.x] = 1;
	}
	device_f9(value_fitness, pop, size, ind, result, local, bias, wi, NULL, -1, dim);
}

__global__ void global_f12(double* value_fitness, double * pop, size_t size, int *d_shuffle, int dim)
{
	double bias = 1200.0;
	int ind = threadIdx.x + blockIdx.x * blockDim.x;
	double * result = shared;
	double * local = result + blockDim.x;
	double * wi = local + blockDim.x * blockDim.y;

	if (threadIdx.y == 0)
	{
		value_fitness[ind] = 0;
		wi[threadIdx.x] = 1;
	}
	device_f10(value_fitness, pop, size, ind, result, local, bias, wi, NULL, -1, dim);
}

__global__ void global_f13(double* value_fitness, double * pop, size_t size, int *d_shuffle, int dim)
{
	double bias = 1300.0;
	int ind = threadIdx.x + blockIdx.x * blockDim.x;
	double * result = shared;
	double * wi = result + blockDim.x;
	double * local = wi + blockDim.x;

	if (threadIdx.y == 0)
	{
		value_fitness[ind] = 0;
		wi[threadIdx.x] = 1;
	}
	device_f11(value_fitness, pop, size, ind, result, local, bias, wi, NULL, -1, dim);
}

__global__ void global_f14(double* value_fitness, double * pop, size_t size, int *d_shuffle, int dim)
{
	double bias = 1400.0;
	int ind = threadIdx.x + blockIdx.x * blockDim.x;
	double * result = shared;
	double * wi = result + blockDim.x;
	double * local = wi + blockDim.x;

	if (threadIdx.y == 0)
	{
		value_fitness[ind] = 0;
		wi[threadIdx.x] = 1;
	}
	device_f12(value_fitness, pop, size, ind, result, local, bias, wi, NULL, -1, dim);
}

__global__ void global_f15(double* value_fitness, double * pop, size_t size, int *d_shuffle, int dim)
{
	double bias = 1500.0;
	int ind = threadIdx.x + blockIdx.x * blockDim.x;
	double * result = shared;
	double * local = result + blockDim.x;
	double * wi = local + blockDim.x * blockDim.y;

	if (threadIdx.y == 0)
	{
		value_fitness[ind] = 0;
		wi[threadIdx.x] = 1;
	}
	device_f13(value_fitness, pop, size, ind, result, local, bias, wi, NULL, -1, dim);
}

__global__ void global_f16(double* value_fitness, double * pop, size_t size, int *d_shuffle, int dim)
{
	double bias = 1600.0;
	int ind = threadIdx.x + blockIdx.x * blockDim.x;
	double * result = shared;
	double * local = result + blockDim.x;
	double * wi = local + blockDim.x * blockDim.y;

	if (threadIdx.y == 0)
	{
		value_fitness[ind] = 0;
		wi[threadIdx.x] = 1;
	}
	device_f14(value_fitness, pop, size, ind, result, local, bias, wi, NULL, -1, dim);
}

__global__ void global_f17(double* value_fitness, double * pop, size_t size, int *d_shuffle, int dim)
{
	double bias = 1700.0;
	int ind = threadIdx.x + blockIdx.x * blockDim.x;
	double * result = shared;
	double * wi = result + blockDim.x * blockDim.y;
	double * tmpValue = wi + blockDim.x;
	double * local = tmpValue + blockDim.x;
	if (threadIdx.y == 0)
	{
		value_fitness[ind] = 0;
		wi[threadIdx.x] = 1;
		tmpValue[threadIdx.x] = 0;
	}
	device_hybirdFunc1(value_fitness, pop, tmpValue, size, ind, result, local, 0, wi, d_shuffle, dim);
	if (threadIdx.y == 0)
		value_fitness[ind] = tmpValue[threadIdx.x] + bias;
}

__global__ void global_f18(double* value_fitness, double * pop, size_t size, int *d_shuffle, int dim)
{
	double bias = 1800.0;
	int ind = threadIdx.x + blockIdx.x * blockDim.x;
	double * result = shared;
	double * wi = result + blockDim.x * blockDim.y;
	double * tmpValue = wi + blockDim.x;
	double * local = tmpValue + blockDim.x;
	if (threadIdx.y == 0)
	{
		value_fitness[ind] = 0;
		wi[threadIdx.x] = 1;
		tmpValue[threadIdx.x] = 0;
	}
	device_hybirdFunc2(value_fitness, pop, tmpValue, size, ind, result, local, 0, wi, d_shuffle, dim);
	if (threadIdx.y == 0)
		value_fitness[ind] = tmpValue[threadIdx.x] + bias;
}

__global__ void global_f19(double* value_fitness, double * pop, size_t size, int *d_shuffle, int dim)
{
	double bias = 1900.0;
	int ind = threadIdx.x + blockIdx.x * blockDim.x;
	double * result = shared;
	double * wi = result + blockDim.x * blockDim.y;
	double * tmpValue = wi + blockDim.x;
	double * local = tmpValue + blockDim.x;
	if (threadIdx.y == 0)
	{
		value_fitness[ind] = 0;
		wi[threadIdx.x] = 1;
		tmpValue[threadIdx.x] = 0;
	}
	device_hybirdFunc3(value_fitness, pop, tmpValue, size, ind, result, local, 0, wi, d_shuffle, dim);
	if (threadIdx.y == 0)
		value_fitness[ind] = tmpValue[threadIdx.x] + bias;
}

__global__ void global_f20(double* value_fitness, double * pop, size_t size, int *d_shuffle, int dim)
{
	double bias = 2000.0;
	int ind = threadIdx.x + blockIdx.x * blockDim.x;
	double * result = shared;
	double * wi = result + blockDim.x * blockDim.y;
	double * tmpValue = wi + blockDim.x;
	double * local = tmpValue + blockDim.x;
	if (threadIdx.y == 0)
	{
		value_fitness[ind] = 0;
		wi[threadIdx.x] = 1;
		tmpValue[threadIdx.x] = 0;
	}
	device_hybirdFunc4(value_fitness, pop, tmpValue, size, ind, result, local, 0, wi, d_shuffle, dim);
	if (threadIdx.y == 0)
		value_fitness[ind] = tmpValue[threadIdx.x] + bias;
}

__global__ void global_f21(double* value_fitness, double * pop, size_t size, int *d_shuffle, int dim)
{
	double bias = 2100.0;
	int ind = threadIdx.x + blockIdx.x * blockDim.x;
	double * result = shared;
	double * wi = result + blockDim.x * blockDim.y;
	double * tmpValue = wi + blockDim.x;
	double * local = tmpValue + blockDim.x;
	if (threadIdx.y == 0)
	{
		value_fitness[ind] = 0;
		wi[threadIdx.x] = 1;
		tmpValue[threadIdx.x] = 0;
	}
	device_hybirdFunc5(value_fitness, pop, tmpValue, size, ind, result, local, 0, wi, d_shuffle, dim);
	if (threadIdx.y == 0)
		value_fitness[ind] = tmpValue[threadIdx.x] + bias;
}

__global__ void global_f22(double* value_fitness, double * pop, size_t size, int *d_shuffle, int dim)
{
	double bias = 2200.0;
	int ind = threadIdx.x + blockIdx.x * blockDim.x;
	double * result = shared;
	double * wi = result + blockDim.x * blockDim.y;
	double * tmpValue = wi + blockDim.x;
	double * local = tmpValue + blockDim.x;
	if (threadIdx.y == 0)
	{
		value_fitness[ind] = 0;
		wi[threadIdx.x] = 1;
		tmpValue[threadIdx.x] = 0;
	}
	device_hybirdFunc6(value_fitness, pop, tmpValue, size, ind, result, local, 0, wi, d_shuffle, dim);
	if (threadIdx.y == 0)
		value_fitness[ind] = tmpValue[threadIdx.x] + bias;
}

__global__ void global_f23(double* value_fitness, double * pop, double * rotpop, double * d_shift, size_t size, int dim)
{
	double subbias;
	double bias = 2300;
	int numFunc = 5;
	double * result = shared;
	double * wi = result + blockDim.x;
	double * wi2 = wi + blockDim.x;
	double * sumWi = wi2 + blockDim.x;
	double * local = sumWi + blockDim.x;
	int ind = threadIdx.x + blockIdx.x * blockDim.x;

	double unit1, unit2, unit3, unit4;
	if (threadIdx.y == 0)
	{
		sumWi[threadIdx.x] = 0;
		value_fitness[ind] = 0;
	}
	//shift data for calWi;
	unit1 = pop[ind + size * threadIdx.y] - d_shift[threadIdx.y];
	local[threadIdx.x + blockDim.x * threadIdx.y] = unit1 * unit1;
	__syncthreads();
	device_parallelsum(local, &result[threadIdx.x], blockDim.y);
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
	device_f4(value_fitness, rotpop, size * numFunc, ind, result, local, subbias, wi2, NULL, -1, dim);

	__syncthreads();
	unit1 = pop[ind + size * threadIdx.y] - d_shift[threadIdx.y + MAX_DIM];
	local[threadIdx.x + blockDim.x * threadIdx.y] = unit1 * unit1;
	__syncthreads();
	device_parallelsum(local, &result[threadIdx.x], blockDim.y);
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
	device_f1(value_fitness, rotpop + size, size * numFunc, ind, result, local, subbias, wi2, NULL, -1, dim);


	__syncthreads();
	unit1 = pop[ind + size * threadIdx.y] - d_shift[threadIdx.y + 2 * MAX_DIM];
	local[threadIdx.x + blockDim.x * threadIdx.y] = unit1 * unit1;
	__syncthreads();
	device_parallelsum(local, &result[threadIdx.x], blockDim.y);
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
		subbias = 200.0 * wi[threadIdx.x];

	}
	device_f2(value_fitness, rotpop + 2 * size, size * numFunc, ind, result, local, subbias, wi2, NULL, -1, dim);


	__syncthreads();
	unit1 = pop[ind + size * threadIdx.y] - d_shift[threadIdx.y + 3 * MAX_DIM];
	local[threadIdx.x + blockDim.x * threadIdx.y] = unit1 * unit1;
	__syncthreads();
	device_parallelsum(local, &result[threadIdx.x], blockDim.y);
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
		wi2[threadIdx.x] = 1e-6 * wi[threadIdx.x];
		subbias = 300.0 * wi[threadIdx.x];

	}
	device_f3(value_fitness, rotpop + 3 * size, size * numFunc, ind, result, local, subbias, wi2, NULL, -1, dim);


	__syncthreads();
	unit1 = pop[ind + size * threadIdx.y] - d_shift[threadIdx.y + 4 * MAX_DIM];
	local[threadIdx.x + blockDim.x * threadIdx.y] = unit1 * unit1;
	__syncthreads();
	device_parallelsum(local, &result[threadIdx.x], blockDim.y);
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
	device_f1(value_fitness, rotpop + 4 * size, size * numFunc, ind, result, local, subbias, wi2, NULL, -1, dim);
	if (threadIdx.y == 0)
		value_fitness[ind] = value_fitness[ind] / sumWi[threadIdx.x] + bias;
}

__global__ void global_f24(double* value_fitness, double * pop, double * rotpop, double * d_shift, size_t size, int dim)
{
	double subbias;
	double bias = 2400;
	int numFunc = 3;
	double * result = shared;
	double * wi = result + blockDim.x;
	double * wi2 = wi + blockDim.x;
	double * sumWi = wi2 + blockDim.x;
	double * local = sumWi + blockDim.x;
	int ind = threadIdx.x + blockIdx.x * blockDim.x;

	double unit1, unit2, unit3, unit4;
	if (threadIdx.y == 0)
	{
		sumWi[threadIdx.x] = 0;
		value_fitness[ind] = 0;
	}
	//shift data for calWi;
	unit1 = pop[ind + size * threadIdx.y] - d_shift[threadIdx.y];
	local[threadIdx.x + blockDim.x * threadIdx.y] = unit1 * unit1;
	__syncthreads();
	device_parallelsum(local, &result[threadIdx.x], blockDim.y);
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
	device_f9(value_fitness, rotpop, size * numFunc, ind, result, local, subbias, wi2, NULL, -1, dim);

	__syncthreads();
	unit1 = pop[ind + size * threadIdx.y] - d_shift[threadIdx.y + MAX_DIM];
	local[threadIdx.x + blockDim.x * threadIdx.y] = unit1 * unit1;
	__syncthreads();
	device_parallelsum(local, &result[threadIdx.x], blockDim.y);
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
	device_f8(value_fitness, rotpop + size, size * numFunc, ind, result, local, subbias, wi2, NULL, -1, dim);


	__syncthreads();
	unit1 = pop[ind + size * threadIdx.y] - d_shift[threadIdx.y + 2 * MAX_DIM];
	local[threadIdx.x + blockDim.x * threadIdx.y] = unit1 * unit1;
	__syncthreads();
	device_parallelsum(local, &result[threadIdx.x], blockDim.y);
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
	device_f12(value_fitness, rotpop + 2 * size, size * numFunc, ind, result, local, subbias, wi2, NULL, -1, dim);
	if (threadIdx.y == 0)
		value_fitness[ind] = value_fitness[ind] / sumWi[threadIdx.x] + bias;
}

__global__ void global_f25(double* value_fitness, double * pop, double * rotpop, double * d_shift, size_t size, int dim)
{
	double subbias;
	double bias = 2500;
	int numFunc = 3;
	double * result = shared;
	double * wi = result + blockDim.x;
	double * wi2 = wi + blockDim.x;
	double * sumWi = wi2 + blockDim.x;
	double * local = sumWi + blockDim.x;
	int ind = threadIdx.x + blockIdx.x * blockDim.x;

	double unit1, unit2, unit3, unit4;
	if (threadIdx.y == 0)
	{
		sumWi[threadIdx.x] = 0;
		value_fitness[ind] = 0;
	}
	//shift data for calWi;
	unit1 = pop[ind + size * threadIdx.y] - d_shift[threadIdx.y];
	local[threadIdx.x + blockDim.x * threadIdx.y] = unit1 * unit1;
	__syncthreads();
	device_parallelsum(local, &result[threadIdx.x], blockDim.y);
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
	device_f9(value_fitness, rotpop, size * numFunc, ind, result, local, subbias, wi2, NULL, -1, dim);

	__syncthreads();
	unit1 = pop[ind + size * threadIdx.y] - d_shift[threadIdx.y + MAX_DIM];
	local[threadIdx.x + blockDim.x * threadIdx.y] = unit1 * unit1;
	__syncthreads();
	device_parallelsum(local, &result[threadIdx.x], blockDim.y);
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
	device_f8(value_fitness, rotpop + size * 1, size * numFunc, ind, result, local, subbias, wi2, NULL, -1, dim);


	__syncthreads();
	unit1 = pop[ind + size * threadIdx.y] - d_shift[threadIdx.y + 2 * MAX_DIM];
	local[threadIdx.x + blockDim.x * threadIdx.y] = unit1 * unit1;
	__syncthreads();
	device_parallelsum(local, &result[threadIdx.x], blockDim.y);
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
	device_f1(value_fitness, rotpop + 2 * size, size * numFunc, ind, result, local, subbias, wi2, NULL, -1, dim);
	if (threadIdx.y == 0)
		value_fitness[ind] = value_fitness[ind] / sumWi[threadIdx.x] + bias;
}

__global__ void global_f26(double* value_fitness, double * pop, double * rotpop, double * d_shift, size_t size, int dim)
{
	double subbias;
	double bias = 2600;
	int numFunc = 5;
	double * result = shared;
	double * wi = result + blockDim.x;
	double * wi2 = wi + blockDim.x;
	double * sumWi = wi2 + blockDim.x;
	double * local = sumWi + blockDim.x;
	int ind = threadIdx.x + blockIdx.x * blockDim.x;

	double unit1, unit2, unit3, unit4;
	if (threadIdx.y == 0)
	{
		sumWi[threadIdx.x] = 0;
		value_fitness[ind] = 0;
	}
	//shift data for calWi;
	unit1 = pop[ind + size * threadIdx.y] - d_shift[threadIdx.y];
	local[threadIdx.x + blockDim.x * threadIdx.y] = unit1 * unit1;
	__syncthreads();
	device_parallelsum(local, &result[threadIdx.x], blockDim.y);
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
	device_f9(value_fitness, rotpop, size * numFunc, ind, result, local, subbias, wi2, NULL, -1, dim);

	__syncthreads();
	unit1 = pop[ind + size * threadIdx.y] - d_shift[threadIdx.y + MAX_DIM];
	local[threadIdx.x + blockDim.x * threadIdx.y] = unit1 * unit1;
	__syncthreads();
	device_parallelsum(local, &result[threadIdx.x], blockDim.y);
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
	device_f11(value_fitness, rotpop + size * 1, size * numFunc, ind, result, local, subbias, wi2, NULL, -1, dim);


	__syncthreads();
	unit1 = pop[ind + size * threadIdx.y] - d_shift[threadIdx.y + 2 * MAX_DIM];
	local[threadIdx.x + blockDim.x * threadIdx.y] = unit1 * unit1;
	__syncthreads();
	device_parallelsum(local, &result[threadIdx.x], blockDim.y);
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
	device_f1(value_fitness, rotpop + 2 * size, size * numFunc, ind, result, local, subbias, wi2, NULL, -1, dim);

	__syncthreads();
	unit1 = pop[ind + size * threadIdx.y] - d_shift[threadIdx.y + 3 * MAX_DIM];
	local[threadIdx.x + blockDim.x * threadIdx.y] = unit1 * unit1;
	__syncthreads();
	device_parallelsum(local, &result[threadIdx.x], blockDim.y);
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
	device_f6(value_fitness, rotpop + 3 * size, size * numFunc, ind, result, local, subbias, wi2, NULL, -1, dim);

	__syncthreads();
	unit1 = pop[ind + size * threadIdx.y] - d_shift[threadIdx.y + 4 * MAX_DIM];
	local[threadIdx.x + blockDim.x * threadIdx.y] = unit1 * unit1;
	__syncthreads();
	device_parallelsum(local, &result[threadIdx.x], blockDim.y);
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
	device_f7(value_fitness, rotpop + 4 * size, size * numFunc, ind, result, local, subbias, wi2, NULL, -1, dim);
	if (threadIdx.y == 0)
		value_fitness[ind] = value_fitness[ind] / sumWi[threadIdx.x] + bias;
}

__global__ void global_f27(double* value_fitness, double * pop, double * rotpop, double * d_shift, size_t size, int dim)
{
	double subbias;
	double bias = 2700;
	int numFunc = 5;
	double * result = shared;
	double * wi = result + blockDim.x;
	double * wi2 = wi + blockDim.x;
	double * sumWi = wi2 + blockDim.x;
	double * local = sumWi + blockDim.x;
	int ind = threadIdx.x + blockIdx.x * blockDim.x;

	double unit1, unit2, unit3, unit4;
	if (threadIdx.y == 0)
	{
		sumWi[threadIdx.x] = 0;
		value_fitness[ind] = 0;
	}
	//shift data for calWi;
	unit1 = pop[ind + size * threadIdx.y] - d_shift[threadIdx.y];
	local[threadIdx.x + blockDim.x * threadIdx.y] = unit1 * unit1;
	__syncthreads();
	device_parallelsum(local, &result[threadIdx.x], blockDim.y);
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
	device_f12(value_fitness, rotpop, size * numFunc, ind, result, local, subbias, wi2, NULL, -1, dim);

	__syncthreads();
	unit1 = pop[ind + size * threadIdx.y] - d_shift[threadIdx.y + MAX_DIM];
	local[threadIdx.x + blockDim.x * threadIdx.y] = unit1 * unit1;
	__syncthreads();
	device_parallelsum(local, &result[threadIdx.x], blockDim.y);
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
	device_f8(value_fitness, rotpop + size * 1, size * numFunc, ind, result, local, subbias, wi2, NULL, -1, dim);


	__syncthreads();
	unit1 = pop[ind + size * threadIdx.y] - d_shift[threadIdx.y + 2 * MAX_DIM];
	local[threadIdx.x + blockDim.x * threadIdx.y] = unit1 * unit1;
	__syncthreads();
	device_parallelsum(local, &result[threadIdx.x], blockDim.y);
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
		;
		wi2[threadIdx.x] = 2.5 * wi[threadIdx.x];
		subbias = 200.0 * wi[threadIdx.x];
	}
	device_f9(value_fitness, rotpop + 2 * size, size * numFunc, ind, result, local, subbias, wi2, NULL, -1, dim);

	__syncthreads();
	unit1 = pop[ind + size * threadIdx.y] - d_shift[threadIdx.y + 3 * MAX_DIM];
	local[threadIdx.x + blockDim.x * threadIdx.y] = unit1 * unit1;
	__syncthreads();
	device_parallelsum(local, &result[threadIdx.x], blockDim.y);
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
	device_f6(value_fitness, rotpop + 3 * size, size * numFunc, ind, result, local, subbias, wi2, NULL, -1, dim);

	__syncthreads();
	unit1 = pop[ind + size * threadIdx.y] - d_shift[threadIdx.y + 4 * MAX_DIM];
	local[threadIdx.x + blockDim.x * threadIdx.y] = unit1 * unit1;
	__syncthreads();
	device_parallelsum(local, &result[threadIdx.x], blockDim.y);
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
	device_f1(value_fitness, rotpop + 4 * size, size * numFunc, ind, result, local, subbias, wi2, NULL, -1, dim);
	if (threadIdx.y == 0)
		value_fitness[ind] = value_fitness[ind] / sumWi[threadIdx.x] + bias;
}

__global__ void global_f28(double* value_fitness, double * pop, double * rotpop, double * d_shift, size_t size, int dim)
{
	double subbias;
	double bias = 2800;
	int numFunc = 5;
	double * result = shared;
	double * wi = result + blockDim.x;
	double * wi2 = wi + blockDim.x;
	double * sumWi = wi2 + blockDim.x;
	double * local = sumWi + blockDim.x;
	int ind = threadIdx.x + blockIdx.x * blockDim.x;

	double unit1, unit2, unit3, unit4;
	if (threadIdx.y == 0)
	{
		sumWi[threadIdx.x] = 0;
		value_fitness[ind] = 0;
	}
	//shift data for calWi;
	unit1 = pop[ind + size * threadIdx.y] - d_shift[threadIdx.y];
	local[threadIdx.x + blockDim.x * threadIdx.y] = unit1 * unit1;
	__syncthreads();
	device_parallelsum(local, &result[threadIdx.x], blockDim.y);
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
	device_f13(value_fitness, rotpop, size * numFunc, ind, result, local, subbias, wi2, NULL, -1, dim);

	__syncthreads();
	unit1 = pop[ind + size * threadIdx.y] - d_shift[threadIdx.y + MAX_DIM];
	local[threadIdx.x + blockDim.x * threadIdx.y] = unit1 * unit1;
	__syncthreads();
	device_parallelsum(local, &result[threadIdx.x], blockDim.y);
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
	device_f11(value_fitness, rotpop + size * 1, size * numFunc, ind, result, local, subbias, wi2, NULL, -1, dim);


	__syncthreads();
	unit1 = pop[ind + size * threadIdx.y] - d_shift[threadIdx.y + 2 * MAX_DIM];
	local[threadIdx.x + blockDim.x * threadIdx.y] = unit1 * unit1;
	__syncthreads();
	device_parallelsum(local, &result[threadIdx.x], blockDim.y);
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
	device_f9(value_fitness, rotpop + 2 * size, size * numFunc, ind, result, local, subbias, wi2, NULL, -1, dim);

	__syncthreads();
	unit1 = pop[ind + size * threadIdx.y] - d_shift[threadIdx.y + 3 * MAX_DIM];
	local[threadIdx.x + blockDim.x * threadIdx.y] = unit1 * unit1;
	__syncthreads();
	device_parallelsum(local, &result[threadIdx.x], blockDim.y);
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
	device_f14(value_fitness, rotpop + 3 * size, size * numFunc, ind, result, local, subbias, wi2, NULL, -1, dim);

	__syncthreads();
	unit1 = pop[ind + size * threadIdx.y] - d_shift[threadIdx.y + 4 * MAX_DIM];
	local[threadIdx.x + blockDim.x * threadIdx.y] = unit1 * unit1;
	__syncthreads();
	device_parallelsum(local, &result[threadIdx.x], blockDim.y);
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
	device_f1(value_fitness, rotpop + 4 * size, size * numFunc, ind, result, local, subbias, wi2, NULL, -1, dim);
	if (threadIdx.y == 0)
		value_fitness[ind] = value_fitness[ind] / sumWi[threadIdx.x] + bias;
}

__global__ void global_f29(double* value_fitness, double * pop, double * rotpop, double * d_shift, size_t size, int *d_shuffle, int dim)
{
	double subbias;
	double bias = 2900;
	int numFunc = 3;
	double * result = shared;
	double * wi = result + blockDim.x;
	double * wi2 = wi + blockDim.x;
	double * sumWi = wi2 + blockDim.x;
	double * tmpValue = sumWi + blockDim.x;
	double * local = tmpValue + blockDim.x;
	int ind = threadIdx.x + blockIdx.x * blockDim.x;

	double unit1, unit2, unit3, unit4;
	if (threadIdx.y == 0)
	{
		sumWi[threadIdx.x] = 0;
		value_fitness[ind] = 0;
		tmpValue[threadIdx.x] = 0;
	}
	//shift data for calWi;
	unit1 = pop[ind + size * threadIdx.y] - d_shift[threadIdx.y];
	local[threadIdx.x + blockDim.x * threadIdx.y] = unit1 * unit1;
	__syncthreads();
	device_parallelsum(local, &result[threadIdx.x], blockDim.y);
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
	device_hybirdFunc1(value_fitness, rotpop, tmpValue, size * numFunc, ind, result, local, subbias, wi2, d_shuffle, dim);

	__syncthreads();
	unit1 = pop[ind + size * threadIdx.y] - d_shift[threadIdx.y + MAX_DIM];
	local[threadIdx.x + blockDim.x * threadIdx.y] = unit1 * unit1;
	__syncthreads();
	device_parallelsum(local, &result[threadIdx.x], blockDim.y);
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
	device_hybirdFunc2(value_fitness, rotpop + size * 1, tmpValue, size * numFunc, ind, result, local, subbias, wi2, d_shuffle + dim, dim);


	__syncthreads();
	unit1 = pop[ind + size * threadIdx.y] - d_shift[threadIdx.y + 2 * MAX_DIM];
	local[threadIdx.x + blockDim.x * threadIdx.y] = unit1 * unit1;
	__syncthreads();
	device_parallelsum(local, &result[threadIdx.x], blockDim.y);
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
	device_hybirdFunc3(value_fitness, rotpop + size * 2, tmpValue, size * numFunc, ind, result, local, subbias, wi2, d_shuffle + dim * 2, dim);

	__syncthreads();

	if (threadIdx.y == 0)
		value_fitness[ind] = tmpValue[threadIdx.x] / sumWi[threadIdx.x] + bias;
}

__global__ void global_f30(double* value_fitness, double * pop, double * rotpop, double * d_shift, size_t size, int *d_shuffle, int dim)
{
	double subbias;
	double bias = 3000;
	int numFunc = 3;
	double * result = shared;
	double * wi = result + blockDim.x;
	double * wi2 = wi + blockDim.x;
	double * sumWi = wi2 + blockDim.x;
	double * tmpValue = sumWi + blockDim.x;
	double * local = tmpValue + blockDim.x;
	int ind = threadIdx.x + blockIdx.x * blockDim.x;

	double unit1, unit2, unit3, unit4;
	if (threadIdx.y == 0)
	{
		sumWi[threadIdx.x] = 0;
		value_fitness[ind] = 0;
		tmpValue[threadIdx.x] = 0;
	}
	//shift data for calWi;
	unit1 = pop[ind + size * threadIdx.y] - d_shift[threadIdx.y];
	local[threadIdx.x + blockDim.x * threadIdx.y] = unit1 * unit1;
	__syncthreads();
	device_parallelsum(local, &result[threadIdx.x], blockDim.y);
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
	device_hybirdFunc4(value_fitness, rotpop, tmpValue, size * numFunc, ind, result, local, subbias, wi2, d_shuffle, dim);

	__syncthreads();
	unit1 = pop[ind + size * threadIdx.y] - d_shift[threadIdx.y + MAX_DIM];
	local[threadIdx.x + blockDim.x * threadIdx.y] = unit1 * unit1;
	__syncthreads();
	device_parallelsum(local, &result[threadIdx.x], blockDim.y);
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
	device_hybirdFunc5(value_fitness, rotpop + size * 1, tmpValue, size * numFunc, ind, result, local, subbias, wi2, d_shuffle + dim, dim);


	__syncthreads();
	unit1 = pop[ind + size * threadIdx.y] - d_shift[threadIdx.y + 2 * MAX_DIM];
	local[threadIdx.x + blockDim.x * threadIdx.y] = unit1 * unit1;
	__syncthreads();
	device_parallelsum(local, &result[threadIdx.x], blockDim.y);
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
	device_hybirdFunc6(value_fitness, rotpop + size * 2, tmpValue, size * numFunc, ind, result, local, subbias, wi2, d_shuffle + dim * 2, dim);

	__syncthreads();

	if (threadIdx.y == 0)
		value_fitness[ind] = tmpValue[threadIdx.x] / sumWi[threadIdx.x] + bias;
}

extern "C" void API_rotation(double * d_pop_rotated, double * d_pop_original, double * d_M, double * d_shift, double shift, double rate_weighted, dim3 blocks, dim3 threads, natural size_pop, natural dim, natural num_comp_func)
{
	global_matrixMultiply_comp_function << <blocks, threads >> >(d_M, d_pop_original, d_pop_rotated, dim, dim, dim, size_pop, dim, size_pop, d_shift, shift, rate_weighted, num_comp_func);
	CHECK_CUDA_ERROR();
}

extern "C" void API_evaluateFitness(double * d_fval, double * d_pop_original, double * d_pop_rotated, int * d_shuffle, double * d_shift, dim3 blocks, dim3 threads, natural ID_func_, natural size_pop, natural dim)
{
	switch (ID_func_)
	{
	case(1) :
		global_f1 << <blocks, threads, 48000 >> >(d_fval, d_pop_rotated, size_pop, d_shuffle, dim);
		break;
	case(2) :
		global_f2 << <blocks, threads, 48000 >> >(d_fval, d_pop_rotated, size_pop, d_shuffle, dim);
		break;
	case(3) :
		global_f3 << <blocks, threads, 48000 >> >(d_fval, d_pop_rotated, size_pop, d_shuffle, dim);
		break;
	case(4) :
		global_f4 << <blocks, threads, 48000 >> >(d_fval, d_pop_rotated, size_pop, d_shuffle, dim);
		break;
	case(5) :
		global_f5 << <blocks, threads, 48000 >> >(d_fval, d_pop_rotated, size_pop, d_shuffle, dim);
		break;
	case(6) :
		global_f6 << <blocks, threads, 48000 >> >(d_fval, d_pop_rotated, size_pop, d_shuffle, dim);
		break;
	case(7) :
		global_f7 << <blocks, threads, 48000 >> >(d_fval, d_pop_rotated, size_pop, d_shuffle, dim);
		break;
	case(8) :
		global_f8 << <blocks, threads, 48000 >> >(d_fval, d_pop_rotated, size_pop, d_shuffle, dim);
		break;
	case(9) :
		global_f9 << <blocks, threads, 48000 >> >(d_fval, d_pop_rotated, size_pop, d_shuffle, dim);
		break;
	case(10) :
		global_f10 << <blocks, threads, 48000 >> >(d_fval, d_pop_rotated, size_pop, d_shuffle, dim);
		break;
	case(11) :
		global_f11 << <blocks, threads, 48000 >> >(d_fval, d_pop_rotated, size_pop, d_shuffle, dim);
		break;
	case(12) :
		global_f12 << <blocks, threads, 48000 >> >(d_fval, d_pop_rotated, size_pop, d_shuffle, dim);
		break;
	case(13) :
		global_f13 << <blocks, threads, 48000 >> >(d_fval, d_pop_rotated, size_pop, d_shuffle, dim);
		break;
	case(14) :
		global_f14 << <blocks, threads, 48000 >> >(d_fval, d_pop_rotated, size_pop, d_shuffle, dim);
		break;
	case(15) :
		global_f15 << <blocks, threads, 48000 >> >(d_fval, d_pop_rotated, size_pop, d_shuffle, dim);
		break;
	case(16) :
		global_f16 << <blocks, threads, 48000 >> >(d_fval, d_pop_rotated, size_pop, d_shuffle, dim);
		break;
	case(17) :
		global_f17 << <blocks, threads, 48000 >> >(d_fval, d_pop_rotated, size_pop, d_shuffle, dim);
		break;
	case(18) :
		global_f18 << <blocks, threads, 48000 >> >(d_fval, d_pop_rotated, size_pop, d_shuffle, dim);
		break;
	case(19) :
		global_f19 << <blocks, threads, 48000 >> >(d_fval, d_pop_rotated, size_pop, d_shuffle, dim);
		break;
	case(20) :
		global_f20 << <blocks, threads, 48000 >> >(d_fval, d_pop_rotated, size_pop, d_shuffle, dim);
		break;
	case(21) :
		global_f21 << <blocks, threads, 48000 >> >(d_fval, d_pop_rotated, size_pop, d_shuffle, dim);
		break;
	case(22) :
		global_f22 << <blocks, threads, 48000 >> >(d_fval, d_pop_rotated, size_pop, d_shuffle, dim);
		break;
	case(23) :
		global_f23 << <blocks, threads, 48000 >> >(d_fval, d_pop_original, d_pop_rotated, d_shift, size_pop, dim);
		break;
	case(24) :
		global_f24 << <blocks, threads, 48000 >> >(d_fval, d_pop_original, d_pop_rotated, d_shift, size_pop, dim);
		break;
	case(25) :
		global_f25 << <blocks, threads, 48000 >> >(d_fval, d_pop_original, d_pop_rotated, d_shift, size_pop, dim);
		break;
	case(26) :
		global_f26 << <blocks, threads, 48000 >> >(d_fval, d_pop_original, d_pop_rotated, d_shift, size_pop, dim);
		break;
	case(27) :
		global_f27 << <blocks, threads, 48000 >> >(d_fval, d_pop_original, d_pop_rotated, d_shift, size_pop, dim);
		break;
	case(28) :
		global_f28 << <blocks, threads, 48000 >> >(d_fval, d_pop_original, d_pop_rotated, d_shift, size_pop, dim);
		break;
	case(29) :
		global_f29 << <blocks, threads, 48000 >> >(d_fval, d_pop_original, d_pop_rotated, d_shift, size_pop, d_shuffle, dim);
		break;
	case(30) :
		global_f30 << <blocks, threads, 48000 >> >(d_fval, d_pop_original, d_pop_rotated, d_shift, size_pop, d_shuffle, dim);
		break;
	default:
		break;
	}
	CHECK_CUDA_ERROR();
};