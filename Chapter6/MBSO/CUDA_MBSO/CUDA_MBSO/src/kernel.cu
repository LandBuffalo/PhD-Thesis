#include "device_launch_parameters.h"
#include "../include/config.h"

__device__ double device_selectOneCluster(double * d_pop_candidate, double *d_pop_original, double *d_centre, int * d_num_mem_cluster, int * d_num_accum_mem_cluster, double *d_probi_mem_cluster, \
	int * d_mem_cluster, int num_cluster,
#ifdef HOST_RAND
	double *d_rand_unif, 
#endif
#ifdef DEVICE_RAND
	curandState * local_states,
#endif
	natural size_pop, double p_use_centre_one_cluster, int index_pop, int pointer_rand)
{
	double random = 0;
	double value = 0;
	int index_cluster = 0;
#ifdef HOST_RAND
	random = d_rand_unif[pointer_rand + 1];
#endif
#ifdef DEVICE_RAND
	random = curand_uniform_double(local_states);
#endif
	for (int i = 0; i < num_cluster; i++)
	{
		if (random < d_probi_mem_cluster[i])
		{
			index_cluster = i;
			break;
		}
	}
	__syncthreads();
#ifdef HOST_RAND
	random = d_rand_unif[pointer_rand + 2];
#endif
#ifdef DEVICE_RAND
	random = curand_uniform_double(local_states);
#endif
	if (random < p_use_centre_one_cluster)
		value = d_centre[index_cluster + threadIdx.y * num_cluster];
	else
	{
		int index_individual_sort = 0;
		int index_individual = 0;
#ifdef HOST_RAND
		random = d_rand_unif[pointer_rand + 3];
#endif
#ifdef DEVICE_RAND
		random = curand_uniform_double(local_states);
#endif
		while (d_num_mem_cluster[index_cluster] == 0)
		{
			index_cluster++;
			if (index_cluster >= num_cluster)
				index_cluster = 0;
		}
		index_individual_sort = d_num_accum_mem_cluster[index_cluster] + floor(random * d_num_mem_cluster[index_cluster]);
		index_individual = d_mem_cluster[index_individual_sort];
		value = d_pop_original[index_individual + threadIdx.y * size_pop];
	}
	__syncthreads();

	return value;
}

__device__ double device_selectTwoCluster(double * d_pop_candidate, double *d_pop_original, double *d_centre, int * d_num_mem_cluster, int * d_num_accum_mem_cluster, double *d_probi_mem_cluster, \
	int * d_mem_cluster, int num_cluster, 
#ifdef HOST_RAND
	double *d_rand_unif,
#endif
#ifdef DEVICE_RAND
	curandState * local_states,
#endif
	 natural size_pop, double p_use_centre_two_cluster, int index_pop, int pointer_rand)
{
	double random = 0;
	double value = 0;
	int index_cluster1, index_cluster2, index_individual_sort1, index_individual_sort2, index_individual1, index_individual2;
#ifdef HOST_RAND
	random = d_rand_unif[pointer_rand + 1];
#endif
#ifdef DEVICE_RAND
	random = curand_uniform_double(local_states);
#endif
	index_cluster1 = floor(random * num_cluster);
	while (d_num_mem_cluster[index_cluster1] == 0)
	{
		index_cluster1++;
		if (index_cluster1 >= num_cluster)
			index_cluster1 = 0;
	}
#ifdef HOST_RAND
	random = d_rand_unif[pointer_rand + 2];
#endif
#ifdef DEVICE_RAND
	random = curand_uniform_double(local_states);
#endif
	index_cluster2 = floor(random * num_cluster);
	while (d_num_mem_cluster[index_cluster2] == 0)
	{
		index_cluster2++;
		if (index_cluster2 >= num_cluster)
			index_cluster2 = 0;
	}

#ifdef HOST_RAND
	random = d_rand_unif[pointer_rand + 3];
#endif
#ifdef DEVICE_RAND
	random = curand_uniform_double(local_states);
#endif
	index_individual_sort1 = d_num_accum_mem_cluster[index_cluster1] + floor(random * d_num_mem_cluster[index_cluster1]);
	index_individual1 = d_mem_cluster[index_individual_sort1];
#ifdef HOST_RAND
	random = d_rand_unif[pointer_rand + 4];
#endif
#ifdef DEVICE_RAND
	random = curand_uniform_double(local_states);
#endif
	index_individual_sort2 = d_num_accum_mem_cluster[index_cluster2] + floor(random * d_num_mem_cluster[index_cluster2]);
	index_individual2 = d_mem_cluster[index_individual_sort2];
#ifdef HOST_RAND
	random = d_rand_unif[pointer_rand + 5];
#endif
#ifdef DEVICE_RAND
	random = curand_uniform_double(local_states);
#endif
#ifdef HOST_RAND
	double tmp_random = d_rand_unif[pointer_rand + 6];
#endif
#ifdef DEVICE_RAND
	double tmp_random = curand_uniform_double(local_states);
#endif

	if (tmp_random < p_use_centre_two_cluster)
	{
		value = random * d_centre[index_cluster1 + threadIdx.y * num_cluster] + \
			(1 - random) * d_centre[index_cluster2 + threadIdx.y * num_cluster];
	}
	else
	{
		value = random * d_pop_original[index_individual1 + threadIdx.y * size_pop] + \
			(1 - random) * d_pop_original[index_individual2 + threadIdx.y * size_pop];
	}

	__syncthreads();
	return value;
}

extern __shared__ double sharedbounds[];
__global__ void global_applystrategy(double * d_pop_candidate, double *d_pop_original, int * ID_best_cluster, double *d_centre, int * d_num_mem_cluster, int * d_num_accum_mem_cluster, double *d_probi_mem_cluster, \
	int * d_mem_cluster, int num_cluster, 
#ifdef HOST_RAND
	double *d_rand_unif, 
#endif	
#ifdef DEVICE_RAND
	curandState * d_rand_states,
#endif
	natural size_pop, natural dim, double maxbound, double minbound, double p_replace_centre_rand, double p_select_cluster_one_or_two, double p_use_centre_one_cluster, double p_use_centre_two_cluster, double pr)
{
	double random = 0;
	double value = 0;
	int flag_ID_best = 0;
	int index_pop = threadIdx.x + blockIdx.x * blockDim.x;
	int pointer_rand = index_pop * (10 + dim) + 2 + dim;
#ifdef DEVICE_RAND
	int var_random = blockDim.x * blockDim.y * (blockIdx.x + blockIdx.y * gridDim.x) + threadIdx.x + threadIdx.y * blockDim.x;
	curandState local_states = d_rand_states[var_random];
#endif
	if (threadIdx.x == 0 && blockIdx.x == 0)
	{ 
		int indx_cluster = 0;
#ifdef HOST_RAND
		random = d_rand_unif[0];
#endif
#ifdef DEVICE_RAND
		double random = curand_uniform_double(&local_states);
#endif
		if (random < p_replace_centre_rand)
		{
#ifdef HOST_RAND
			indx_cluster = floor(d_rand_unif[1] * num_cluster);
			d_centre[indx_cluster + threadIdx.y * num_cluster] = minbound + (maxbound - minbound) * d_rand_unif[threadIdx.y + 2];
#endif
#ifdef DEVICE_RAND
			indx_cluster = floor(curand_uniform_double(&local_states) * num_cluster);
			d_centre[indx_cluster + threadIdx.y * num_cluster] = minbound + (maxbound - minbound) * curand_uniform_double(&local_states);
#endif
		}
		__syncthreads();
	}
#ifdef HOST_RAND
	random = d_rand_unif[pointer_rand];
#endif
#ifdef DEVICE_RAND
	random = curand_uniform_double(&local_states);
#endif
	__syncthreads();
	for (int k = 0; k < num_cluster; k++)
	{
		if (index_pop == ID_best_cluster[k])
		{
			flag_ID_best = 1;
			break;
		}
	}
	if (flag_ID_best == 0)
	{
		if (random < p_select_cluster_one_or_two)
#ifdef HOST_RAND
			value = device_selectOneCluster(d_pop_candidate, d_pop_original, d_centre, d_num_mem_cluster, d_num_accum_mem_cluster, d_probi_mem_cluster, \
			d_mem_cluster, num_cluster, d_rand_unif, size_pop, p_use_centre_one_cluster, index_pop, pointer_rand);
		else

			value = device_selectTwoCluster(d_pop_candidate, d_pop_original, d_centre, d_num_mem_cluster, d_num_accum_mem_cluster, d_probi_mem_cluster, \
			d_mem_cluster, num_cluster, d_rand_unif, size_pop, p_use_centre_two_cluster, index_pop, pointer_rand);
#endif
#ifdef DEVICE_RAND
			value = device_selectOneCluster(d_pop_candidate, d_pop_original, d_centre, d_num_mem_cluster, d_num_accum_mem_cluster, d_probi_mem_cluster, \
			d_mem_cluster, num_cluster, &local_states, size_pop, p_use_centre_one_cluster, index_pop, pointer_rand);
		else
			//value = device_selectOneCluster(d_pop_candidate, d_pop_original, d_centre, d_num_mem_cluster, d_num_accum_mem_cluster, d_probi_mem_cluster, \
			d_mem_cluster, num_cluster, d_rand_unif, size_pop, p_use_centre_one_cluster, index_pop, pointer_rand);
			value = device_selectTwoCluster(d_pop_candidate, d_pop_original, d_centre, d_num_mem_cluster, d_num_accum_mem_cluster, d_probi_mem_cluster, \
			d_mem_cluster, num_cluster, &local_states, size_pop, p_use_centre_two_cluster, index_pop, pointer_rand);
#endif
		__syncthreads();
#ifdef HOST_RAND
		random = d_rand_unif[pointer_rand + 7];
#endif
#ifdef DEVICE_RAND
		random = curand_uniform_double(&local_states);
#endif
		if (random < pr)
		{			
#ifdef HOST_RAND
			value = minbound + (maxbound - minbound) * d_rand_unif[pointer_rand + 10 + threadIdx.y];
#endif
#ifdef DEVICE_RAND
			value = minbound + (maxbound - minbound) * curand_uniform_double(&local_states);
#endif
		}
		else
		{
#ifdef HOST_RAND
			int ID_individual1 = floor(d_rand_unif[pointer_rand + 8] * size_pop);
			int ID_individual2 = floor(d_rand_unif[pointer_rand + 9] * size_pop);
			value = value + d_rand_unif[pointer_rand + 10 + threadIdx.y] * \
				(d_pop_original[ID_individual1 + threadIdx.y * size_pop] - d_pop_original[ID_individual2 + threadIdx.y * size_pop]);
#endif
#ifdef DEVICE_RAND
			int ID_individual1 = floor(curand_uniform_double(&local_states) * size_pop);
			int ID_individual2 = floor(curand_uniform_double(&local_states) * size_pop);
			value = value + curand_uniform_double(&local_states) * \
				(d_pop_original[ID_individual1 + threadIdx.y * size_pop] - d_pop_original[ID_individual2 + threadIdx.y * size_pop]);
#endif

		}
		
		while ((value < minbound) || (value > maxbound)) {
			if (value < minbound) {
				//~ value = MINBOUND + CURAND_UNIFORM_REAL(&localState)*(MINBOUND - value);
				value = minbound + minbound - value;
			}
			if (value > maxbound) {
				//~ value = MAXBOUND - CURAND_UNIFORM_REAL(&localState)*(value - MAXBOUND);
				value = maxbound - (value - maxbound);
			}
		}
	}
	else
	{
		value = d_pop_original[index_pop + threadIdx.y * size_pop];
	}
	d_pop_candidate[index_pop + threadIdx.y * size_pop] = value;
#ifdef DEVICE_RAND
	d_rand_states[var_random] = local_states;
#endif
}

__global__ void global_initiPop(double *d_pop, natural size_pop, natural dim, double minbound, double maxbound, 
#ifdef HOST_RAND
	const double *d_rand_sequence) 
#endif
#ifdef DEVICE_RAND
	curandState * d_rand_states)
#endif
{
	//var is the index(out of popsize) of individual, a column of block is a whole individual variable, and popsize of column in total(blockDim.x * gridDim.x)   
	int var = threadIdx.x + blockDim.x * blockIdx.x;
#ifdef HOST_RAND
	d_pop[var + threadIdx.y * size_pop] = minbound + d_rand_sequence[var + threadIdx.y * size_pop] * (maxbound - minbound);
#endif
#ifdef DEVICE_RAND
	int var_random = blockDim.x * blockDim.y * (blockIdx.x + blockIdx.y * gridDim.x) + threadIdx.x + threadIdx.y * blockDim.x;
	curandState localState = d_rand_states[var_random];
	d_pop[var + threadIdx.y * size_pop] = minbound + curand_uniform_double(&localState) * (maxbound - minbound);
	d_rand_states[var_random] = localState;
#endif
}

__global__ void global_generateNewPop(double *pop_new, double *fval_new, double *pop_candidate, double *pop_old, double *fval_candidate, double *fval_old, size_t size_pop)
{
	natural ind = threadIdx.x + blockIdx.x * blockDim.x;
	if ((fval_candidate[ind] - fval_old[ind]) >= 0)
	{
		if (threadIdx.y == 0)
			fval_new[ind] = fval_old[ind];
		pop_new[ind + threadIdx.y * size_pop] = pop_old[ind + threadIdx.y * size_pop];
	}
	else
	{
		if (threadIdx.y == 0)
			fval_new[ind] = fval_candidate[ind];
		pop_new[ind + threadIdx.y * size_pop] = pop_candidate[ind + threadIdx.y * size_pop];
	}
	__syncthreads();
}
#ifdef HOST_RAND
extern "C"
void API_initiPop(dim3 blocks, dim3 threads, double * d_pop, natural size_pop, natural dim, double minbound, double maxbound, const double *d_rand_sequence)
{
	global_initiPop << < blocks, threads >> >(d_pop, size_pop, dim, minbound, maxbound, d_rand_sequence);
	CHECK_CUDA_ERROR();

}
#endif
#ifdef DEVICE_RAND
extern "C"
void API_initiPop(dim3 blocks, dim3 threads, double * d_pop, natural size_pop, natural dim, double minbound, double maxbound, curandState * d_rand_states)
{
	global_initiPop << < blocks, threads >> >(d_pop, size_pop, dim, minbound, maxbound, d_rand_states);
	CHECK_CUDA_ERROR();
}
#endif
extern "C"
void API_generateNewPop(dim3 blocks, dim3 threads, double *pop_new, double *fval_new, double *pop_candidate, double *pop_old, double *fval_candidate, double *fval_old, size_t size_pop)
{
	global_generateNewPop << <blocks, threads >> >(pop_new, fval_new, pop_candidate, pop_old, fval_candidate, fval_old, size_pop);
}

extern "C"
void API_applystrategy(dim3 blocks, dim3 threads, double * d_pop_candidate, double *d_pop_original, int * ID_best_cluster, double *d_centre, int * d_num_mem_cluster, int * d_num_accum_mem_cluster, double *d_probi_mem_cluster, \
int * d_mem_cluster, int num_cluster,
#ifdef HOST_RAND
	double *d_rand_unif,
#endif
#ifdef DEVICE_RAND
	curandState * d_rand_states,
#endif
natural size_pop, natural dim, double maxbound, double minbound, double p_replace_centre_rand, double p_select_cluster_one_or_two, double p_use_centre_one_cluster, double p_use_centre_two_cluster, double pr)
{
#ifdef HOST_RAND
	global_applystrategy << <blocks, threads, 48000 >> >(d_pop_candidate, d_pop_original, ID_best_cluster, d_centre, d_num_mem_cluster, d_num_accum_mem_cluster, d_probi_mem_cluster, \
		d_mem_cluster, num_cluster, d_rand_unif, size_pop, dim, maxbound, minbound, p_replace_centre_rand, p_select_cluster_one_or_two, p_use_centre_one_cluster, p_use_centre_two_cluster, pr);
#endif
#ifdef DEVICE_RAND
	global_applystrategy << <blocks, threads, 48000 >> >(d_pop_candidate, d_pop_original, ID_best_cluster, d_centre, d_num_mem_cluster, d_num_accum_mem_cluster, d_probi_mem_cluster, \
		d_mem_cluster, num_cluster, d_rand_states, size_pop, dim, maxbound, minbound, p_replace_centre_rand, p_select_cluster_one_or_two, p_use_centre_one_cluster, p_use_centre_two_cluster, pr);

#endif
	CHECK_CUDA_ERROR();

}

