#include "../include/Population.h"

Population::Population(natural size_pop, natural dim)
{
	size_pop_ = size_pop;
	dim_ = dim;

	d_pop_ = NULL;
	d_fval_ = NULL;
	d_pitch_pop_ = 0;

	HANDLE_CUDA_ERROR(cudaMalloc(&d_pop_, size_pop_ * sizeof(double) * dim_));
	HANDLE_CUDA_ERROR(cudaMemset(d_pop_, 0, size_pop_ * sizeof(double) * dim_));
	HANDLE_CUDA_ERROR(cudaMalloc(&d_fval_, size_pop_ * sizeof(double)));
	HANDLE_CUDA_ERROR(cudaMemset(d_fval_, 0, size_pop_ * sizeof(double)));

	h_pop_ = NULL;
	h_fval_ = NULL;
	h_pop_best_ = NULL;
	h_pop_ = new double[dim_ * size_pop_ * sizeof(double)];
	h_pop_best_ = new double[dim_ * sizeof(double)];
	h_fval_ = new double[size_pop_ * sizeof(double)];
}

Population :: ~Population()
{
	delete[] h_pop_;
	delete[] h_fval_;
	delete[] h_pop_best_;
	HANDLE_CUDA_ERROR(cudaFree(d_pop_));
	HANDLE_CUDA_ERROR(cudaFree(d_fval_));
}
#ifdef HOST_RAND
error Population::InitiPop(dim3 blocks, dim3 threads, double minbound, double maxbound, const double *d_rand_sequence)
{
	API_initiPop(blocks, threads, d_pop_, size_pop_, dim_, minbound, maxbound, d_rand_sequence);
	CHECK_CUDA_ERROR();
	return SUCCESS;
}
#endif
#ifdef DEVICE_RAND
error Population::InitiPop(dim3 blocks, dim3 threads, double minbound, double maxbound, curandState * d_rand_states)
{
	API_initiPop(blocks, threads, d_pop_, size_pop_, dim_, minbound, maxbound, d_rand_states);
	CHECK_CUDA_ERROR();
	return SUCCESS;
}
#endif

void Population::FindIndivualBest()
{
	int index_best = 0;
	LoadFromDev();

	h_fval_best_ = h_fval_[index_best];
	for (int i = 1; i < size_pop_; i++) 
		if (h_fval_best_ > h_fval_[i])
		{
			index_best = i;
			h_fval_best_ = h_fval_[i];
		}

	for (int i = 0; i < dim_; i++)
		h_pop_best_[i] = h_pop_[index_best + size_pop_ * i];
}

double	Population::h_fval_best()
{
	return h_fval_best_;
}

double*	Population::d_pop()
{
	return d_pop_;
}

double*	Population::d_fval()
{
	return d_fval_;
}

void Population::LoadFromDev()
{
	HANDLE_CUDA_ERROR(cudaThreadSynchronize());
	HANDLE_CUDA_ERROR(cudaMemcpy(h_pop_, d_pop_, sizeof(double) * size_pop_  * dim_, cudaMemcpyDeviceToHost));
	HANDLE_CUDA_ERROR(cudaMemcpy(h_fval_, d_fval_, sizeof(double) * size_pop_, cudaMemcpyDeviceToHost));
}
#ifdef DEBUG
void Population::Check()
{
	LoadFromDev();
	FILE * file_record;
	file_record = fopen("population.txt", "w");

	for (int i = 0; i < size_pop_; i++)
	{
		for (int j = 0; j < dim_; j++)
			fprintf(file_record, "%.10f\t", h_pop_[i + j * size_pop_]);
		fprintf(file_record, "\n");
	}
	fclose(file_record);

	file_record = fopen("fitness.txt", "w");

	for (int i = 0; i < size_pop_; i++)
	{
		fprintf(file_record, "%.10f\t", h_fval_[i]);
	}
	fclose(file_record);
}
void Population::LoadToDev()
{
	HANDLE_CUDA_ERROR(cudaMemcpy2D(d_pop_, d_pitch_pop_, h_pop_, size_pop_ * sizeof(double), size_pop_ * sizeof(double), dim_, cudaMemcpyHostToDevice));
	HANDLE_CUDA_ERROR(cudaMemcpy(d_fval_, h_fval_, size_pop_ * sizeof(double), cudaMemcpyHostToDevice));

}
#endif