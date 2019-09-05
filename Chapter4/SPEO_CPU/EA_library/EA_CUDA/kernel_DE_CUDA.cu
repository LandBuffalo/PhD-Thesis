#include "EA_CUDA.h"
#define CUDA_ERROR_CHECK

#define CudaSafeCall( err ) __cudaSafeCall( err, __FILE__, __LINE__ )
#define CudaCheckError()    __cudaCheckError( __FILE__, __LINE__ )
#define MAX_R_SIZE 5

inline void __cudaSafeCall( cudaError err, const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
	if ( cudaSuccess != err )
	{
		fprintf( stderr, "cudaSafeCall() failed at %s:%i : %s\n",
		         file, line, cudaGetErrorString( err ) );
		exit( -1 );
	}
#endif

	return;
}


inline void __cudaCheckError( const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
	cudaError err = cudaGetLastError();
	if ( cudaSuccess != err )
	{
		fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
		         file, line, cudaGetErrorString( err ) );
		exit( -1 );
	}

	// More careful checking. However, this will affect performance.
	// Comment away if needed.
	err = cudaDeviceSynchronize();
	if( cudaSuccess != err )
	{
		fprintf( stderr, "cudaCheckError() with sync failed at %s:%i : %s\n",
		         file, line, cudaGetErrorString( err ) );
		exit( -1 );
	}
#endif

	return;
}

int CalConfigurationsDE(dim3 & block_dim, dim3 & grid_dim, int island_size)
{
	if (MAX_THREAD / WARP_SIZE < island_size)
	{
		block_dim = dim3(WARP_SIZE, MAX_THREAD / WARP_SIZE, 1);
		grid_dim = dim3(island_size / block_dim.y, 1, 1);
	}
	else
	{
		block_dim = dim3(WARP_SIZE, island_size, 1);
		grid_dim = dim3(1, 1, 1);
	}
	return 0;
}

inline __device__ __host__ int NextPow2(int x)
{
	int y = 1;
	while (y < x) y <<= 1;
	if (y < WARP_SIZE)
		y = WARP_SIZE;
	return y;
}

static __device__ __forceinline__ int device_ParallelBest(real * vector, int * index)
{
	__syncwarp();	
	if(threadIdx.x < WARP_SIZE / 2)
		if(vector[threadIdx.x] > vector[threadIdx.x + WARP_SIZE / 2])
		{
			vector[threadIdx.x] =  vector[threadIdx.x + WARP_SIZE / 2];
			index[threadIdx.x] = index[threadIdx.x + WARP_SIZE / 2];
		}
	__syncwarp();	
	if(threadIdx.x < WARP_SIZE / 4)
		if(vector[threadIdx.x] > vector[threadIdx.x + WARP_SIZE / 4])
		{
			vector[threadIdx.x] =  vector[threadIdx.x + WARP_SIZE / 4];
			index[threadIdx.x] = index[threadIdx.x + WARP_SIZE / 4];
		}
	__syncwarp();	
	if(threadIdx.x < WARP_SIZE / 8)
		if(vector[threadIdx.x] > vector[threadIdx.x + WARP_SIZE / 8])
		{
			vector[threadIdx.x] =  vector[threadIdx.x + WARP_SIZE / 8];
			index[threadIdx.x] = index[threadIdx.x + WARP_SIZE / 8];
		}
	__syncwarp();		
	if(threadIdx.x < WARP_SIZE / 16)
		if(vector[threadIdx.x] > vector[threadIdx.x + WARP_SIZE / 16])
		{
			vector[threadIdx.x] =  vector[threadIdx.x + WARP_SIZE / 16];
			index[threadIdx.x] = index[threadIdx.x + WARP_SIZE / 16];
		}
	__syncwarp();	
	if(threadIdx.x < WARP_SIZE / 32)
		if(vector[threadIdx.x] > vector[threadIdx.x + WARP_SIZE / 32])
		{
			vector[threadIdx.x] =  vector[threadIdx.x + WARP_SIZE / 32];
			index[threadIdx.x] = index[threadIdx.x + WARP_SIZE / 32];
		}

	return 0;

}


__global__ void global_DE_FindBestIndividualInIsland(int * d_best_individual_ID, GPU_Population d_population, DEInfo DE_info, ProblemInfo problem_info)
{
	int island_ID = blockIdx.x;
	int group_size = DE_info.group_size;
	int next_pow2_pop = NextPow2(group_size);
	int loop_times_pop = next_pow2_pop / WARP_SIZE;

	__shared__ real sh_fitness[WARP_SIZE];
	__shared__ int sh_index[WARP_SIZE];

	int indivudal_in_group_ID = threadIdx.x;

	if (indivudal_in_group_ID < DE_info.group_size)
	{
		sh_fitness[threadIdx.x] = d_population.fitness_value[indivudal_in_group_ID];
		sh_index[threadIdx.x] = indivudal_in_group_ID;
	}
	for (int i = 1; i < loop_times_pop; i++)
	{
		indivudal_in_group_ID = threadIdx.x + i * WARP_SIZE;

		if (indivudal_in_group_ID < DE_info.group_size)
		{
			if (sh_fitness[threadIdx.x] > d_population.fitness_value[indivudal_in_group_ID])
			{
				sh_fitness[threadIdx.x] = d_population.fitness_value[indivudal_in_group_ID];
				sh_index[threadIdx.x] = indivudal_in_group_ID;
			}
		}
	}
	device_ParallelBest(sh_fitness, sh_index);
	if(threadIdx.x == 0)
		d_best_individual_ID[island_ID] = sh_index[0];

}

__device__ __forceinline__ real device_CheckBound(real to_check_elements, real min_bound, real max_bound)
{
    while ((to_check_elements < min_bound) || (to_check_elements > max_bound))
	{
	    if (to_check_elements < min_bound)
	        to_check_elements = min_bound + (min_bound - to_check_elements);
	    if (to_check_elements > max_bound)
	        to_check_elements = max_bound - (to_check_elements - max_bound);
	}
	return to_check_elements;
}

__global__ void global_DE_GenerateNewPopulation(GPU_Population d_candidate, GPU_Population d_population, int * d_best_individual_ID, curandState *d_rand_states, DEInfo DE_info, ProblemInfo problem_info)
{
	int dim = problem_info.dim;
	int individual_ID = threadIdx.y + blockIdx.x * blockDim.y;
	int group_size = DE_info.group_size;
	int island_ID = individual_ID / group_size;

	int next_pow2_dim = NextPow2(dim);
	int loop_times = next_pow2_dim / WARP_SIZE;
	int best_individual_ID = d_best_individual_ID[island_ID] + group_size * island_ID;
	int local_r[MAX_R_SIZE] = {0};
	int random_state_ID = individual_ID * WARP_SIZE;
	curandState local_state = d_rand_states[random_state_ID];

	for(int i = 0; i < MAX_R_SIZE; i++)
	{
#ifdef DOUBLE_PRECISION
		local_r[i] = (int)(curand_uniform_double(&local_state) * group_size);
#endif
#ifdef SINGLE_PRECISION
		local_r[i] = (int)(curand_uniform(&local_state) * group_size);
#endif
		if(local_r[i] == group_size)
			local_r[i]--;
		for(int j = 0; j < i; j++)
		{
			while(local_r[i] == local_r[j])
			{
#ifdef DOUBLE_PRECISION
				local_r[i] = (int)(curand_uniform_double(&local_state) * group_size);
#endif
#ifdef SINGLE_PRECISION
				local_r[i] = (int)(curand_uniform(&local_state) * group_size);
#endif
			}
		}
		if(local_r[i] == group_size)
			local_r[i]--;
		local_r[i] += group_size * island_ID;
	}

#ifdef DOUBLE_PRECISION
	int local_j = (int) (curand_uniform_double(&local_state) * dim);
#endif
#ifdef SINGLE_PRECISION
	int local_j = (int) (curand_uniform (&local_state) * dim);
#endif
	if(threadIdx.x == 0)
    	d_rand_states[random_state_ID] = local_state;

	random_state_ID = threadIdx.x + individual_ID * WARP_SIZE;
	local_state = d_rand_states[random_state_ID];
	real F = DE_info.F;
	real CR = DE_info.CR;
	int strategy_ID = DE_info.strategy_ID;
	real local_candidate_elements = 0;
	for(int i = 0; i < loop_times; i++)
	{
		int element_ID = threadIdx.x + i * WARP_SIZE;
#ifdef DOUBLE_PRECISION
	real rand_CR = curand_uniform_double(&local_state);
#endif
#ifdef SINGLE_PRECISION
	real rand_CR = curand_uniform (&local_state);
#endif
        if(element_ID < dim)
        {
            if ((element_ID == local_j) || (rand_CR < CR))
            {
				if(strategy_ID == 0)
            		local_candidate_elements = d_population.elements[element_ID + individual_ID * next_pow2_dim] + F * (d_population.elements[element_ID + local_r[0] * next_pow2_dim] - d_population.elements[element_ID + local_r[1] * next_pow2_dim]);
                if(strategy_ID == 1)
            		local_candidate_elements = d_population.elements[element_ID + individual_ID * next_pow2_dim] + F * (d_population.elements[element_ID + local_r[0] * next_pow2_dim] - d_population.elements[element_ID + local_r[1] * next_pow2_dim]) + F * (d_population.elements[element_ID + local_r[2] * next_pow2_dim] - d_population.elements[element_ID + local_r[3] * next_pow2_dim]);
				if(strategy_ID == 2)
                    local_candidate_elements = d_population.elements[element_ID + individual_ID * next_pow2_dim] + F * (d_population.elements[element_ID +  best_individual_ID * next_pow2_dim] - d_population.elements[element_ID + individual_ID * next_pow2_dim]) + F * (d_population.elements[element_ID + local_r[0] * next_pow2_dim] - d_population.elements[element_ID + local_r[1] * next_pow2_dim]);
				if(strategy_ID == 3)
                    local_candidate_elements = d_population.elements[element_ID + individual_ID * next_pow2_dim] + F * (d_population.elements[element_ID + best_individual_ID  * next_pow2_dim] - d_population.elements[element_ID + individual_ID * next_pow2_dim]) + F * (d_population.elements[element_ID + local_r[0] * next_pow2_dim] - d_population.elements[element_ID + local_r[1] * next_pow2_dim]) + F * (d_population.elements[element_ID + local_r[1] * next_pow2_dim] - d_population.elements[element_ID + local_r[2] * next_pow2_dim]);
				if(strategy_ID == 4)
                    local_candidate_elements = d_population.elements[element_ID + local_r[0] * next_pow2_dim] + F * (d_population.elements[element_ID + local_r[1] * next_pow2_dim] - d_population.elements[element_ID + local_r[2] * next_pow2_dim]);
				if(strategy_ID == 5)
                    local_candidate_elements = d_population.elements[element_ID + local_r[0] * next_pow2_dim] + F * (d_population.elements[element_ID + local_r[1] * next_pow2_dim] - d_population.elements[element_ID + local_r[2] * next_pow2_dim]) + F * (d_population.elements[element_ID + local_r[3] * next_pow2_dim] - d_population.elements[element_ID + local_r[4] * next_pow2_dim]);
				if(strategy_ID == 6)
                    local_candidate_elements = d_population.elements[element_ID + best_individual_ID * next_pow2_dim] + F * (d_population.elements[element_ID + local_r[0] * next_pow2_dim] - d_population.elements[element_ID + local_r[1] * next_pow2_dim]);
				if(strategy_ID == 7)
                    local_candidate_elements = d_population.elements[element_ID + best_individual_ID * next_pow2_dim] + F * (d_population.elements[element_ID + local_r[0] * next_pow2_dim] - d_population.elements[element_ID + local_r[1] * next_pow2_dim]) +\
				F * (d_population.elements[element_ID + local_r[2] * next_pow2_dim] - d_population.elements[element_ID + local_r[3] * next_pow2_dim]);
                if(strategy_ID == 8)
            		local_candidate_elements = d_population.elements[element_ID + individual_ID * next_pow2_dim] + F * (d_population.elements[element_ID + local_r[0] * next_pow2_dim] - d_population.elements[element_ID + individual_ID * next_pow2_dim]) + F * (d_population.elements[element_ID + local_r[1] * next_pow2_dim] - d_population.elements[element_ID + local_r[2] * next_pow2_dim]);
            }
            else
            {
                local_candidate_elements = d_population.elements[element_ID + individual_ID * next_pow2_dim];
            }

            local_candidate_elements = device_CheckBound(local_candidate_elements, problem_info.min_bound, problem_info.max_bound);
            d_candidate.elements[element_ID + individual_ID * next_pow2_dim] = local_candidate_elements;
        }
	}
    d_rand_states[random_state_ID] = local_state;
}

extern "C"
int API_DE_GenerateNewPopulation(GPU_Population d_candidate, GPU_Population d_population, int * d_best_individual_ID, curandState *d_rand_states, DEInfo DE_info, ProblemInfo problem_info)
{
	dim3 block_dim, grid_dim;

	global_DE_FindBestIndividualInIsland<< <dim3(DE_info.group_num, 1, 1), dim3(WARP_SIZE, 1, 1) >> >(d_best_individual_ID, d_population, DE_info, problem_info);
	CudaCheckError();
	CalConfigurationsDE(block_dim, grid_dim, DE_info.group_num * DE_info.group_size);
	global_DE_GenerateNewPopulation << <grid_dim, block_dim >> >(d_candidate, d_population, d_best_individual_ID, d_rand_states, DE_info, problem_info);
	CudaCheckError();

	return 0;
}


__global__ void global_DE_SelectSurvival(GPU_Population d_population, GPU_Population d_candidate, ProblemInfo problem_info)
{
	int dim = problem_info.dim;
	int individual_ID = threadIdx.y + blockIdx.x * blockDim.y;
	int next_pow2_dim = NextPow2(dim);
	int loop_times = next_pow2_dim / WARP_SIZE;

    if(d_candidate.fitness_value[individual_ID] < d_population.fitness_value[individual_ID])
	{

		for(int i = 0; i < loop_times; i++)
		{
			int element_ID = threadIdx.x + i * WARP_SIZE;
	        if(element_ID < dim)
				d_population.elements[element_ID + individual_ID * next_pow2_dim] = d_candidate.elements[element_ID + individual_ID * next_pow2_dim];

		}
		if (threadIdx.x == 0)
		{
			d_population.fitness_value[individual_ID] = d_candidate.fitness_value[individual_ID];
		}
	}
}

extern "C"
int API_DE_SelectSurvival(GPU_Population d_population, GPU_Population d_candidate, DEInfo DE_info, ProblemInfo problem_info)
{
	dim3 block_dim, grid_dim;
	CalConfigurationsDE(block_dim, grid_dim, DE_info.group_num * DE_info.group_size);
	global_DE_SelectSurvival << <grid_dim, block_dim >> >(d_population, d_candidate, problem_info);
	CudaCheckError();

	return 0;
}
/*
static __device__ __forceinline__ real device_ParallelSum(real * vector)
{
	if(threadIdx.x < WARP_SIZE / 2)
		vector[threadIdx.x] += vector[threadIdx.x + WARP_SIZE / 2];
	if(threadIdx.x < WARP_SIZE / 4)
		vector[threadIdx.x] += vector[threadIdx.x + WARP_SIZE / 4];
	if(threadIdx.x < WARP_SIZE / 8)
		vector[threadIdx.x] += vector[threadIdx.x + WARP_SIZE / 8];
	if(threadIdx.x < WARP_SIZE / 16)
		vector[threadIdx.x] += vector[threadIdx.x + WARP_SIZE / 16];
	if(threadIdx.x < WARP_SIZE / 32)
		vector[threadIdx.x] += vector[threadIdx.x + WARP_SIZE / 32];
	return vector[0];

};
static __device__ __forceinline__ real device_ParallelBest(real * vector)
{
	if(threadIdx.x < WARP_SIZE / 2)
		vector[threadIdx.x] = vector[threadIdx.x] > vector[threadIdx.x + WARP_SIZE / 2] ? vector[threadIdx.x + WARP_SIZE / 2] : vector[threadIdx.x]
	if(threadIdx.x < WARP_SIZE / 4)
		vector[threadIdx.x] = vector[threadIdx.x] > vector[threadIdx.x + WARP_SIZE / 4] ? vector[threadIdx.x + WARP_SIZE / 4] : vector[threadIdx.x]
	if(threadIdx.x < WARP_SIZE / 8)
		vector[threadIdx.x] = vector[threadIdx.x] > vector[threadIdx.x + WARP_SIZE / 8] ? vector[threadIdx.x + WARP_SIZE / 8] : vector[threadIdx.x]
	if(threadIdx.x < WARP_SIZE / 16)
		vector[threadIdx.x] = vector[threadIdx.x] > vector[threadIdx.x + WARP_SIZE / 16] ? vector[threadIdx.x + WARP_SIZE / 16] : vector[threadIdx.x]
	if(threadIdx.x < WARP_SIZE / 32)
		vector[threadIdx.x] = vector[threadIdx.x] > vector[threadIdx.x + WARP_SIZE / 32] ? vector[threadIdx.x + WARP_SIZE / 32] : vector[threadIdx.x]
	return vector[0];

};
extern __shared__ real sh_mem[];
__global__ void global_CalulateMatrixDistance(real * d_distance, real * d_elements, int pop_size, int dim)
{
	int next_pow2_dim = NextPow2(problem_info.dim);
	int loop_times_dim = next_pow2_dim / WARP_SIZE;
	int loop_times_individual = pop_size / blockDim.y;
	int from_individual_ID = blockIdx.x;
	int to_individual_ID = 0;

    real sh_elements = sh_mem;
    real sh_sumed_value = sh_mem + WARP_SIZE;

    for(int i = 0; i < loop_times_individual; i++)
    {
    	to_individual_ID = threadIdx.y + i * blockDim.y;
    	sh_sumed_value[threadIdx.x + threadIdx.y * WARP_SIZE] = 0;
    	for(int j = 0; j < loop_times_dim; j++)
    	{
    		int element_ID = threadIdx.x + j * WARP_SIZE;
    		if(threadIdx.y == 0)
				sh_elements[threadIdx.x] = d_elements[element_ID + from_individual_ID * next_pow2_dim];
			real tmp1 = sh_elements[threadIdx.x];
			real tmp2 = d_elements[element_ID + to_individual_ID * next_pow2_dim];
			tmp1 = (tmp1 - tmp2) * (tmp1 - tmp2)
			sh_sumed_value[threadIdx.x + threadIdx.y * WARP_SIZE] += tmp1;
    	}
    	d_distance[to_individual_ID + from_individual_ID * pop_size] = device_ParallelSum(sh_sumed_value + threadIdx.y * WARP_SIZE);
    }
}

__global__ void global_FindNearest(int *d_nearest_index, real * d_distance, int pop_size)
{
	int next_pow2_dim = NextPow2(problem_info.dim);
	int loop_times_individual = pop_size / WARP_SIZE;
	int individual_ID = threadIdx.y + blockIdx.x * blockDim.y;
	int local_individual_ID = threadIdx.x;

    real sh_distance = sh_mem;

	sh_distance[local_individual_ID] = d_distance[local_individual_ID + individual_ID * pop_size];
	for(int i = 1; i < loop_times_individual; i++)
	{
		local_individual_ID = threadIdx.x + i * WARP_SIZE;
		if(sh_distance[local_individual_ID]  > d_distance[local_individual_ID + individual_ID * pop_size])
			sh_distance[threadIdx.x + individual_ID * WARP_SIZE] = d_distance[local_individual_ID + individual_ID * pop_size];
	}
	nearest_index[individual_ID] = device_ParallelBest(sh_distance + threadIdx.y * WARP_SIZE);
}


int FindNearestIndividualIndex(int *nearest_index, real * d_elements, real * d_distance, int pop_size, int dim)
{
	dim3 block1(WARP_SIZE, WARP_SIZE / 2 , 1);
	dim3 grid1(pop_size, 1, 1);
	global_CalulateMatrixDistance<<<grid1, block1, WARP_SIZE * (block1.y + 1) * sizeof(real)>>>(d_distance, d_elements, pop_size, dim);

	int tmp_block_y = MAX_THREAD / 2 < pop_size ? MAX_THREAD / 2 : pop_size;
	dim3 block2(WARP_SIZE, tmp_block_y, 1);
	dim3 grid2(1, pop_size / tmp_block_y, 1);
	global_FindNeares<<<grid2, block2, WARP_SIZE * tmp_block_y * sizeof(real)>>>(d_nearest_index, d_elements, pop_size);
	return 0;
}
*/
