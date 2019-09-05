#include "device_launch_parameters.h"
#include "../include/config.h"

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

__device__ void device_CalPairwiseDistance(double * sh_distance, double * d_pop, double * d_centre, int num_centre, \
	int size_pop, double * sh_result, double * sh_local)
{
	int var = threadIdx.x + blockIdx.x * blockDim.x;
	for (int i = 0; i < num_centre; i++)
	{
		sh_local[threadIdx.x + blockDim.x * threadIdx.y] = (d_pop[var + threadIdx.y * size_pop] - d_centre[i + threadIdx.y * num_centre]) *\
		(d_pop[var + threadIdx.y * size_pop] - d_centre[i + threadIdx.y * num_centre]);
		__syncthreads();
		device_parallelsum(sh_local, &sh_result[threadIdx.x], blockDim.y);
		if (threadIdx.y == 0)
			sh_distance[threadIdx.x + i * size_pop] = sh_result[threadIdx.x];
		__syncthreads();
	}
}

__device__ bool device_LablePopWithCentreID(bool * lable_change, double * sh_distance, int * d_lable_cluster, int * d_num_mem_cluster, int * d_num_mem_accu_cluster, int * d_mem_cluster, \
	int num_centre, int size_pop, double * sh_local)
{
	int var = threadIdx.x + blockIdx.x * blockDim.x;
	int ID_cluster;
	double min_value = sh_distance[threadIdx.x];
	bool label_change = 0;
	ID_cluster = 0;
	if (threadIdx.y == 0)
	{
		for (int i = 1; i < num_centre; i++)
		{
			if (min_value > sh_distance[threadIdx.x + i * num_centre])
			{
				min_value = sh_distance[threadIdx.x + i * num_centre];
				ID_cluster = i;
			}
		}
		if (d_lable_cluster[var] != ID_cluster)
			*lable_change = 1;
		d_lable_cluster[var] = ID_cluster;
		d_num_mem_cluster[ID_cluster]++;
		__syncthreads();

	}
	d_num_mem_accu_cluster[0] = 0;
	for (int i = 1; i < num_centre; i++)
	{
		d_num_mem_accu_cluster[i] = d_num_mem_accu_cluster[i - 1] + d_num_mem_cluster[ID_cluster];
	}
	if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0)
	{
		int count = 0;
		for (int i = 1; i < num_centre; i++)
			sh_local[i] = 0;
		for (int i = 0; i < size_pop; i++)
		{
			int ID_cluster = d_lable_cluster[i];
			int index_sort = d_num_mem_accu_cluster[ID_cluster] + sh_local[ID_cluster];
			d_mem_cluster[index_sort] = d_lable_cluster[i];
			sh_local[ID_cluster]++;
		}
	}

	return label_change;
}
__device__ void device_updateCentre(double * d_centre, double * d_pop, int * d_mem_cluster, int * d_num_mem_cluster, int * d_num_mem_accu_cluster, int num_centre, int size_pop)
{
	int ID_sort_pop = 0;
	if (blockDim.x < num_centre)
	{
		d_centre[blockDim.x + threadIdx.y * num_centre] = 0;
		for (int i = 0; i < d_mem_cluster[blockDim.x]; i++)
		{
			ID_sort_pop = d_mem_cluster[i + d_num_mem_accu_cluster[blockDim.x]];

			d_centre[blockDim.x + threadIdx.y * num_centre] += d_pop[ID_sort_pop + threadIdx.y * size_pop];
		}
		__syncthreads();
		d_centre[blockDim.x + threadIdx.y * num_centre] = d_centre[blockDim.x + threadIdx.y * num_centre] / d_mem_cluster[blockDim.x];
	}
}

__device__ void device_initiCentre(double * d_centre, double * d_pop, int num_cluster, int size_pop)
{
	int var = threadIdx.x + blockIdx.x * blockDim.x;
	if (var < num_cluster)
	{
		d_centre[var + threadIdx.y * num_cluster] = d_pop[var + threadIdx.y * size_pop];
	}
	__syncthreads();

}

extern __shared__ double shared[];
__global__ void global_KMeans(double * d_centre, double * pairwise_distance, int * d_lable_cluster, int * d_num_mem_cluster, int * d_num_mem_accu_cluster, int * d_mem_cluster, \
	double * d_pop, int num_cluster, int size_pop, int maxIlteration)
{
	double * sh_result = shared;
	double * sh_local1 = sh_result + blockDim.x;
//	double * sh_distance = sh_local1 + blockDim.x * blockDim.y;
//	double * sh_local2 = sh_distance + num_cluster * blockDim.x;
	double * sh_local2 = sh_local1 + blockDim.x * blockDim.y;

	int i = 0;
	device_initiCentre(d_centre, d_pop, num_cluster, size_pop);
	bool lable_change = 1;
	while (i < maxIlteration || lable_change == 0)
	{
		device_CalPairwiseDistance(pairwise_distance, d_pop, d_centre, num_cluster, size_pop, sh_result, sh_local1);
		lable_change = 0;
		device_LablePopWithCentreID(&lable_change, pairwise_distance, d_lable_cluster, d_num_mem_cluster, d_num_mem_accu_cluster, \
			d_mem_cluster, num_cluster, size_pop, sh_local2);
		device_updateCentre(d_centre, d_pop, d_mem_cluster, d_num_mem_cluster, d_num_mem_accu_cluster, num_cluster, size_pop);
	}

}




extern "C" void API_global_KMeans(dim3 blocks, dim3 threads, double * d_centre, double * pairwise_distance, int * d_lable_cluster, int * d_num_mem_cluster, int * d_num_mem_accu_cluster, int * d_mem_cluster, \
	double * d_pop, int num_centre, int size_pop, int num_ilteration)
{
	global_KMeans << <blocks, threads, 48000 >> >(d_centre, pairwise_distance, d_lable_cluster, d_num_mem_cluster, d_num_mem_accu_cluster, d_mem_cluster, \
		d_pop, num_centre, size_pop, num_ilteration);
}