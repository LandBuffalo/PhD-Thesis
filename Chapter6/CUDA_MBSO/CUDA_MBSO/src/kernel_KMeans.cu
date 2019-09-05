#include "device_launch_parameters.h"
#include "../include/config.h"
#include <math.h> 
int kkk = 0;


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
static __device__ __forceinline__  void device_parallelsum(int * vector, int* result, int lengthSum)
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
		int sum = vector[threadIdx.x];
		for (int i = 1; i < olds; i++) {
			//~ if (blockIdx.x == 0 && threadIdx.x ==0) printf("T %d I %d OLDS %d V %f\n", threadIdx.y, i, olds, vector[threadIdx.x + blockDim.x * i]);
			sum += vector[threadIdx.x + blockDim.x * i];
		}
		*result = sum;
	}

	__syncthreads();

};

static __device__ __forceinline__  void device_parallelmin(double * vector, double * index_vector, int * index_min, int lengthSum)
{
	//blockDim.y is the dimension of problem
	int olds = lengthSum;
	//maybe can be improved--------------------------------------//
	// if the olds can be divided by 2, use paralle sum
	for (int s = lengthSum / 2; olds == s * 2; s >>= 1) {
		olds = s;
		//~ if (blockIdx.x == 0 && threadIdx.x ==0 ) printf("T %d S %d OLDS %d\n", threadIdx.y, s, olds);
		//sum the two elements(index and index + s)
		if (threadIdx.x < s) 
			if (vector[threadIdx.x] > vector[threadIdx.x + s])
			{
				vector[threadIdx.x] = vector[threadIdx.x + s];
				index_vector[threadIdx.x] = index_vector[threadIdx.x + s];
			}
			else if (vector[threadIdx.x] == vector[threadIdx.x + s])
			{
				if (index_vector[threadIdx.x] > index_vector[threadIdx.x + s])
					index_vector[threadIdx.x] = index_vector[threadIdx.x + s];
			}
		__syncthreads();

	}
	// if the olds can  not be divided by 2, use sequentially sum from threadIdx.y = 0
	if (threadIdx.x == 0)
	{
		double min_value = vector[0];
		for (int i = 1; i < olds; i++) 
		{
			if (vector[i] < min_value)
			{
				min_value = vector[i];
				index_vector[0] = index_vector[i];
			}
		}
		*index_min = __double2int_rd(index_vector[0]);
	}

	__syncthreads();

};

__global__ void global_initiCentre(double * d_centre, 
#ifdef HOST_RAND
	double * d_rand_sequence_unif, 
#endif
#ifdef DEVICE_RAND
	curandState *	d_rand_states,
#endif
int num_cluster, double minbound, double maxbound)
{
	int var = threadIdx.x + blockIdx.x * blockDim.x;
	int var_random = blockDim.x * blockDim.y * (blockIdx.x + blockIdx.y * gridDim.x) + threadIdx.x + threadIdx.y * blockDim.x;

	if (var < num_cluster)
	{
#ifdef HOST_RAND
		d_centre[var + threadIdx.y * num_cluster] = minbound + (maxbound - minbound) * d_rand_sequence_unif[var + threadIdx.y * num_cluster];
#endif
#ifdef DEVICE_RAND
		curandState localState = d_rand_states[var_random];
		d_centre[var + threadIdx.y * num_cluster] = minbound + curand_uniform_double(&localState) * (maxbound - minbound);
		d_rand_states[var_random] = localState;
#endif
	}
	__syncthreads();
}



extern "C" void API_global_initiCentre(dim3 blocks, dim3 threads, double * d_centre, 
#ifdef HOST_RAND
	double * d_rand_sequence_unif,
#endif
#ifdef DEVICE_RAND
	curandState *	d_rand_states,
#endif
	int num_cluster, int dim, double minbound, double maxbound)
{
#ifdef HOST_RAND
	global_initiCentre << <blocks, threads >> >(d_centre, d_rand_sequence_unif, num_cluster, minbound, maxbound);
#endif
#ifdef DEVICE_RAND
	global_initiCentre << <blocks, threads >> >(d_centre, d_rand_states, num_cluster, minbound, maxbound);

#endif
}


extern __shared__ double shared[];
__global__ void global_CalDistAndlabelPop(int * d_label_pop, double * d_pop, double * d_centre, int num_cluster, int size_pop, double * distance_pop_cluster)
{
	double * sh_local = shared;
//	double * sh_centre = sh_local + blockDim.y * blockDim.x;
	

	double mindistance = 0;
	double distance = 0;
	int ID_nearest_cluster = 0;
	double value = 0;
	int var = threadIdx.x + blockIdx.x * blockDim.x;

	sh_local[threadIdx.x + blockDim.x * threadIdx.y] = (d_pop[var + threadIdx.y * size_pop] - d_centre[threadIdx.y * num_cluster]) *\
		(d_pop[var + threadIdx.y * size_pop] - d_centre[threadIdx.y * num_cluster]);
	__syncthreads();
	device_parallelsum(sh_local, &mindistance, blockDim.y);

	__syncthreads();

	ID_nearest_cluster = 0;
	for (int i = 1; i < num_cluster; i++)
	{
		sh_local[threadIdx.x + blockDim.x * threadIdx.y] = (d_pop[var + threadIdx.y * size_pop] - d_centre[i + threadIdx.y * num_cluster]) *\
			(d_pop[var + threadIdx.y * size_pop] - d_centre[i + threadIdx.y * num_cluster]);
		__syncthreads();
		device_parallelsum(sh_local, &distance, blockDim.y); 
//		if (threadIdx.y == 0)
//		distance_pop_cluster[i + var * num_cluster] = distance;
		if (threadIdx.y == 0 && distance < mindistance)
		{
			mindistance = distance;
			ID_nearest_cluster = i;
		}
	}
	__syncthreads();
	if (threadIdx.y == 0)
		d_label_pop[var] = ID_nearest_cluster;
}

extern __shared__ int shared2[];
__global__ void global_CalNumMemCluster(double * d_centre, double * d_fval, int * d_label_pop, int * d_num_accum_mem_cluster, double * d_probi_mem_cluster, int * d_num_mem_cluster, int num_cluster, int size_pop, int dim, double * distance_pop_cluster)
{
	int * sh_num_mem_cluster = shared2;
	int * sum_num_mem_cluster = sh_num_mem_cluster + blockDim.y;
	sh_num_mem_cluster[threadIdx.y] = 0;
	if (threadIdx.y < size_pop && d_label_pop[threadIdx.y] == blockIdx.y)
		sh_num_mem_cluster[threadIdx.y] = 1;
	else
		sh_num_mem_cluster[threadIdx.y] = 0;
	__syncthreads();
	device_parallelsum(sh_num_mem_cluster, &d_num_mem_cluster[blockIdx.y], blockDim.y);

	
	if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0)
	{
		d_num_accum_mem_cluster[0] = 0;
		d_probi_mem_cluster[0] = d_num_mem_cluster[0] / (size_pop + 0.0);
		for (int i = 1; i < num_cluster; i++)
		{
			d_num_accum_mem_cluster[i] = d_num_accum_mem_cluster[i - 1] + d_num_mem_cluster[i - 1];
			d_probi_mem_cluster[i] = d_probi_mem_cluster[i - 1] + d_num_mem_cluster[i] / (0.0 + size_pop);
		}
	}
}

__global__ void global_FindBestIndividualInCluster(double * d_centre, double * d_pop, int * d_num_mem_cluster, double * d_fval, int * d_label_pop, int * d_ID_best_cluster, int num_cluster, int dim, int size_pop)
{
	double *sh_index_vector = shared;
	double *sh_fval = sh_index_vector + blockDim.x;

	sh_index_vector[threadIdx.x] = threadIdx.x;
	if (d_num_mem_cluster[blockIdx.y] != 0)
	{
		if (threadIdx.x < size_pop && d_label_pop[threadIdx.x] == blockIdx.y)
		{
			//		d_mem_cluster[threadIdx.x] = threadIdx.x;
			sh_fval[threadIdx.x] = d_fval[threadIdx.x];// d_total_fval_cluster[blockIdx.y + threadIdx.x * num_cluster];
		}
		else
			sh_fval[threadIdx.x] = 1e99;
		__syncthreads();

		device_parallelmin(sh_fval, sh_index_vector, &d_ID_best_cluster[blockIdx.y], blockDim.x);

		if (threadIdx.x < dim)
		{
			d_centre[blockIdx.y + threadIdx.x * num_cluster] = d_pop[d_ID_best_cluster[blockIdx.y] + threadIdx.x * size_pop];
		}
	}
	else
	{
		d_ID_best_cluster[blockIdx.y] = -1;
	}
}
__global__ void global_SetMemCluster(int * d_mem_cluster, int * d_label_pop, int *d_num_accum_mem_cluster,  int size_pop)
{
	int count[MAX_CLUSTER] = { 0 };
	d_mem_cluster[threadIdx.x] = threadIdx.x;
	int ID_cluster = 0;
	
	for (int i = 0; i < size_pop; i++)
	{
		ID_cluster = d_label_pop[i];
		d_mem_cluster[d_num_accum_mem_cluster[ID_cluster] + count[ID_cluster]] = i;
		count[ID_cluster]++;
	}
}
extern "C" void API_global_Clustering(dim3 blocks, dim3 threads, double * d_centre, int * d_label_pop, int * d_num_accum_mem_cluster, double *  d_probi_mem_cluster, int * d_mem_cluster,int * d_num_mem_cluster,
	double * d_pop, double * d_fval, int num_cluster, int size_pop, int dim, double * distance_pop_cluster, int *d_ID_best_cluster, double * d_total_fval_cluster, cudaStream_t	stream)
{
	/* Random Generation 2 */

//	global_initiCentre << <blocks, threads>> >(d_centre, d_label_pop, d_pop, num_cluster, size_pop);
	//int *d_label_change = NULL;
	//HANDLE_CUDA_ERROR(cudaMalloc(&d_label_change, sizeof(int)));
	//HANDLE_CUDA_ERROR(cudaMemset(d_label_change, 1, sizeof(int)));


//	int* h_label_pop = new int[size_pop];
//	double* h_centre = new double[num_cluster * dim];
//	int * h_num_mem_cluster = new int[num_cluster];
//	double* h_fval = new double[size_pop];
//	double * h_pop = new double[size_pop * dim];
//	int * h_ID_best_cluster = new int[num_cluster];

//	double * h_centre = new double[num_cluster * dim];
//	double * h_probi_mem_cluster = new double[num_cluster];
//	double * tmpp;
//	HANDLE_CUDA_ERROR(cudaMalloc(&tmpp, sizeof(double) * num_cluster * 50));
//	double * h_tmpp = new double[50 * num_cluster];
//	double * tmp = new double[num_cluster];
//	int * h_num_accum_mem_cluster = new int[num_cluster];

//	FILE * file_record;
//	FILE *file_record;


	global_CalDistAndlabelPop << <blocks, threads, 32000 >> >(d_label_pop, d_pop, d_centre, num_cluster, size_pop, distance_pop_cluster);
	CHECK_CUDA_ERROR();

		//cudaMemcpy(h_centre, d_centre, num_cluster * dim * sizeof(double), cudaMemcpyDeviceToHost);
	//file_record = fopen("centre.txt", "w");
	//for (int k = 0; k < num_cluster; k++)
	//{
	//	for (int j = 0; j < dim; j++)
	//		fprintf(file_record, "%.5f\t", h_centre[k + j * num_cluster]);
	//	fprintf(file_record, "\n");
	//}
	//fclose(file_record);
	//cudaMemcpy(h_label_pop, d_label_pop, size_pop * sizeof(int), cudaMemcpyDeviceToHost);
	//file_record = fopen("label_pop.txt", "w");
	//for (int k = 0; k < size_pop; k++)
	//{
	//	fprintf(file_record, "%d\t", h_label_pop[k]);
	//}
	//fclose(file_record);

	global_CalNumMemCluster << <dim3(1, num_cluster, 1), dim3(1, size_pop, 1), 48000 >> >(d_centre, d_fval, d_label_pop, d_num_accum_mem_cluster, d_probi_mem_cluster, d_num_mem_cluster, num_cluster, size_pop, dim, distance_pop_cluster);
	CHECK_CUDA_ERROR();

	//cudaMemcpy(h_label_pop, d_label_pop, size_pop * sizeof(int), cudaMemcpyDeviceToHost);
	//cudaMemcpy(h_fval, d_fval, size_pop * sizeof(double), cudaMemcpyDeviceToHost);
	//cudaMemcpy(h_pop, d_pop, size_pop * dim * sizeof(double), cudaMemcpyDeviceToHost);
	//cudaMemcpy(h_num_mem_cluster, d_num_mem_cluster, num_cluster * sizeof(int), cudaMemcpyDeviceToHost);
	//cudaMemcpy(h_centre, d_centre, num_cluster * dim * sizeof(double), cudaMemcpyDeviceToHost);
	//double * fval_best_cluster = new double[num_cluster];
	//for (int k = 0; k < num_cluster; k++)
	//{
	//	h_ID_best_cluster[k] = -1;
	//	fval_best_cluster[k] = 1e99;
	//}
	////find the best individual in each cluster
	//for (int i = 0; i < size_pop; i++)
	//{
	//	if (fval_best_cluster[h_label_pop[i]] > h_fval[i])
	//	{
	//		fval_best_cluster[h_label_pop[i]] = h_fval[i];
	//		h_ID_best_cluster[h_label_pop[i]] = i;
	//	}
	//}
	//delete[] fval_best_cluster;
	////
	////calculate the centre of cluster
	//for (int k = 0; k < num_cluster; k++)
	//{
	//	if (h_num_mem_cluster[k] != 0)
	//		for (int j = 0; j < dim; j++)
	//			h_centre[k + j * num_cluster] = h_pop[h_ID_best_cluster[k] + j * size_pop];
	//}
	//cudaMemcpy(d_ID_best_cluster, h_ID_best_cluster, num_cluster * sizeof(int), cudaMemcpyHostToDevice);
	//cudaMemcpy(d_centre, h_centre, num_cluster * dim * sizeof(double), cudaMemcpyHostToDevice);
	//delete h_label_pop;
	//delete h_centre;
	//delete h_num_mem_cluster;
	//delete h_fval;
	//delete h_pop;
	//delete h_ID_best_cluster;



	//	double * h_centre = new double[num_cluster * dim];
	//	double * h_probi_mem_cluster = new double[num_cluster];
	//	double * tmpp;
	//	HANDLE_CUDA_ERROR(cudaMalloc(&tmpp, sizeof(double) * num_cluster * 50));
	//	double * h_tmpp = new double[50 * num_cluster];
	//	double * tmp = new double[num_cluster];
	//	int * h_num_accum_mem_cluster = new int[num_cluster];
	//	FILE * file_record;
	int threadX = 0;
	double result = frexp(size_pop, &threadX);
	threadX = pow(2.0, threadX+0.0);
	global_FindBestIndividualInCluster << <dim3(1, num_cluster, 1), dim3(threadX, 1, 1), 48000 >> >(d_centre, d_pop, d_num_mem_cluster, d_fval, d_label_pop, d_ID_best_cluster, num_cluster, dim, size_pop);
	CHECK_CUDA_ERROR();

	//cudaMemcpy(h_ID_best_cluster, d_ID_best_cluster, num_cluster * sizeof(int), cudaMemcpyDeviceToHost);
	//file_record = fopen("ID_best_cluster.txt", "w");
	//for (int k = 0; k < num_cluster; k++)
	//{
	//	fprintf(file_record, "%d\t", h_ID_best_cluster[k]);
	//}
	//fclose(file_record);


	global_SetMemCluster << <dim3(1, 1, 1), dim3(1, 1, 1), 0 >> >(d_mem_cluster, d_label_pop, d_num_accum_mem_cluster, size_pop);
	CHECK_CUDA_ERROR();

	//cudaMemcpy(h_ID_best_cluster, d_ID_best_cluster, num_cluster * sizeof(int), cudaMemcpyDeviceToHost);
	//file_record = fopen("num_mem_cluster.txt", "w");
	//for (int k = 0; k < num_cluster; k++)
	//{
	//	fprintf(file_record, "%d\t", h_ID_best_cluster[k]);
	//}
	//fclose(file_record);
//	global_FindBestPopInCluster << <dim3(1, num_cluster, 1), dim3(size_pop,1 , 1) >> >(d_centre, d_fval, d_total_fval_cluster, d_label_pop, size_pop);
//	for (int i = 0; i < 100; i++)
	//CHECK_CUDA_ERROR();


	//for (int i = 0; i < num_cluster; i ++)
	//	status = cublasIdamin(cnpHandle, size_pop, d_total_fval_cluster + i * size_pop, 1, d_ID_best_cluster + i);
	//cudaMemcpy(h_ID_best_cluster, d_ID_best_cluster, num_cluster * sizeof(int), cudaMemcpyDeviceToHost);
	//file_record = fopen("best_index.txt", "w");
	//for (int j = 0; j < num_cluster; j++)
	//	fprintf(file_record, "%d\t", h_ID_best_cluster[j]);
	//fclose(file_record);
	//CHECK_CUDA_ERROR();

//	global_SetCentre << < dim3(1, num_cluster, 1), dim3(1, dim, 1) >> >(d_centre, d_pop, d_ID_best_cluster, d_num_mem_cluster, num_cluster, size_pop);
//	CHECK_CUDA_ERROR();

//
//	int * num_accum_mem_cluster = new int[num_cluster]; 
//	int * num_mem_cluster = new int[num_cluster];
//	cudaMemcpy(num_mem_cluster, d_num_mem_cluster, num_cluster * sizeof(int), cudaMemcpyDeviceToHost);
//	int * mem_cluster = new int[size_pop];
//	double * probi_mem_cluster = new double[num_cluster];
//	int * index_tmp = new int[num_cluster];
//	int * label_pop = new int[size_pop];
//	cudaMemcpy(label_pop, d_label_pop, size_pop * sizeof(int), cudaMemcpyDeviceToHost);
//	int * ID_best_cluster = new int[num_cluster];
//
//	double * fval = new double[size_pop];
//	cudaMemcpy(fval, d_fval, size_pop * sizeof(double), cudaMemcpyDeviceToHost);
//
//	double * centre = new double[num_cluster * dim];
//	cudaMemcpy(centre, d_centre, dim * num_cluster * sizeof(double), cudaMemcpyDeviceToHost);
//
//	double * pop = new double[dim * size_pop];
//	cudaMemcpy(pop, d_pop, size_pop * dim * sizeof(double), cudaMemcpyDeviceToHost);
//	double *distance = new double[size_pop];
//	num_accum_mem_cluster[0] = 0;
//	index_tmp[0] = 0;
//
//	for (int j = 0; j < num_cluster; j++)
//	{
//		if (num_mem_cluster[j] == 0)
//		{
//		
//			distance[0] = 0;
//			for (int k = 0; k < dim; k++)
//				distance[0] += (pop[0 + k * size_pop] - centre[j + k * num_cluster]) * (pop[0 + k * size_pop] - centre[j + k * num_cluster]);
//			double min_distance = 1e99;
//			int ID_min_pop = 0;
//			for (int i = 0; i < size_pop; i++)
//			{
//				distance[i] = 0;
//				for (int k = 0; k < dim; k++)
//					distance[i] += (pop[i + k * size_pop] - centre[j + k * num_cluster]) * (pop[i + k * size_pop] - centre[j + k * num_cluster]);
//				if (min_distance > distance[i] && num_mem_cluster[label_pop[i]] > 1)
//				{
//					min_distance = distance[i];
//					ID_min_pop = i;
//				}
//			}
//
//			num_mem_cluster[label_pop[ID_min_pop]]--;
//			label_pop[ID_min_pop] = j;
//			num_mem_cluster[j] = 1;
//		}
//	}
//	probi_mem_cluster[0] = num_mem_cluster[0] / (size_pop + 0.0);
//	for (int i = 1; i < num_cluster; i++)
//	{
//		index_tmp[i] = 0;
//		num_accum_mem_cluster[i] = num_accum_mem_cluster[i - 1] + num_mem_cluster[i - 1];
//		probi_mem_cluster[i] = num_mem_cluster[i] / (size_pop + 0.0) + probi_mem_cluster[i - 1];
//	}
//	for (int i = 0; i < size_pop; i++)
//	{
//		int ID_cluster = label_pop[i];
//		mem_cluster[num_accum_mem_cluster[ID_cluster] + index_tmp[ID_cluster]] = i;
//		index_tmp[ID_cluster]++;
//	}
//

//	cudaMemcpy(d_num_accum_mem_cluster, num_accum_mem_cluster, num_cluster * sizeof(int), cudaMemcpyHostToDevice);
//	cudaMemcpy(d_mem_cluster, mem_cluster, size_pop * sizeof(int), cudaMemcpyHostToDevice);
//	cudaMemcpy(d_probi_mem_cluster, probi_mem_cluster, num_cluster * sizeof(double), cudaMemcpyHostToDevice);
//	cudaMemcpy(d_num_mem_cluster, num_mem_cluster, num_cluster * sizeof(int), cudaMemcpyHostToDevice);
//
////	cudaMemcpy(centre, d_centre, num_cluster * dim * sizeof(double), cudaMemcpyDeviceToHost);
//	delete[] distance;
//	delete[] num_accum_mem_cluster;
//	delete[] num_mem_cluster;
//	delete[] mem_cluster;
//	delete[] probi_mem_cluster;
//	delete[] index_tmp;
//	delete[] label_pop;
//	delete[] ID_best_cluster;
//	delete[] fval;
//	delete[] centre;
//	delete[] pop;

	//cudaMemcpy(h_centre, d_centre, num_cluster * dim*sizeof(double), cudaMemcpyDeviceToHost);
	//file_record = fopen("centre.txt", "w");
	//for (int k = 0; k < num_cluster; k++)
	//{
	//	for (int j = 0; j < dim; j++)
	//		fprintf(file_record, "%.5f\t", h_centre[k + j * num_cluster]);
	//	fprintf(file_record, "\n");
	//}
	//fclose(file_record);
//	global_CalMemVariable << <1, 1 >> >(d_mem_cluster, d_probi_mem_cluster, d_num_mem_accu_cluster, d_label_pop, d_num_mem_cluster, num_cluster, size_pop, dim);
//	cudaMemcpy(h_probi_mem_cluster, d_probi_mem_cluster, num_cluster * sizeof(double), cudaMemcpyDeviceToHost);

//	cudaMemcpy(h_num_mem_cluster, d_num_mem_cluster, num_cluster * sizeof(int), cudaMemcpyDeviceToHost);
}
