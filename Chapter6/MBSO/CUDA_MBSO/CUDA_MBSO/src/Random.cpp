#include "../include/Random.h"

#ifdef HOST_RAND
Random::Random(int length_rand_sequence_unif, natural size_pop, natural dim, natural run, int ID_device)
#endif
#ifdef DEVICE_RAND
Random::Random(natural size_pop, natural dim, natural run, int ID_device)
#endif
{
	size_pop_ = size_pop;
	dim_ = dim;
	run_ = run;
	kernel_configuration_ = new KernelConfiguration(size_pop_, dim_, ID_device);

#ifdef DEVICE_RAND
	d_rand_states_ = NULL;
	kernel_configuration_->CalKernelConfiguration();
#endif
#ifdef HOST_RAND
	length_rand_sequence_unif_ = length_rand_sequence_unif;
	pointer_start_unif_ = 0;
	d_rand_sequence_unif_ = NULL;
	pointer_start_unif_ = dim_ * size_pop_;
	HANDLE_CUDA_ERROR(cudaMalloc(&d_rand_sequence_unif_, sizeof(double) * length_rand_sequence_unif_));
#endif
#ifdef DEBUG
#ifdef HOST_RAND
	h_rand_sequence_unif_ = NULL;
	h_rand_sequence_norm_ = NULL;
	h_rand_sequence_unif_ = new double[sizeof(double) * length_rand_sequence_unif_];
#endif
#endif
#ifdef IMPORT_RAND
#ifndef DEBUG
	h_rand_sequence_ = NULL;
	h_rand_sequence_ = new double[sizeof(double) * length_rand_sequence_];
#endif
#endif
}

Random::~Random()
{
#ifdef DEBUG
	delete[] h_rand_sequence_unif_;
	delete[] h_rand_sequence_norm_;
#endif
#ifdef DEVICE_RAND.
	HANDLE_CUDA_ERROR(cudaFree(d_rand_states_));
#endif
#ifdef IMPORT_RAND
#ifndef DEBUG
	delete[] h_rand_sequence_unif_;
	delete[] h_rand_sequence_norm_; 
#endif
#endif
#ifdef HOST_RAND
	HANDLE_CUDA_ERROR(cudaFree(d_rand_sequence_unif_));
	kernel_configuration_->~KernelConfiguration();
#endif
}

natural	Random::run()
{
	return run_;
}

void Random::set_seed(int seed)
{
	seed_ = seed + run_ * size_pop_;
}

#ifdef DEVICE_RAND
error Random::initRandom()
{
	size_t size_rand = kernel_configuration_->blocks_.x * kernel_configuration_->blocks_.y * \
		kernel_configuration_->threads_.x * kernel_configuration_->threads_.y;
	HANDLE_CUDA_ERROR(cudaMalloc((void **)&d_rand_states_, size_rand * sizeof(curandState)));
	CHECK_CUDA_ERROR();
	API_setupRandomState (kernel_configuration_->blocks_, kernel_configuration_->threads_, d_rand_states_, seed_);
	CHECK_CUDA_ERROR();
	return SUCCESS;
}
curandState * Random::d_rand_states()
{
	return d_rand_states_;
}

#endif

#ifdef HOST_RAND
error Random::LoopAllocation()
{
	return SUCCESS;
}
error Random::Generate_rand_sequence()
{
#ifdef IMPORT_RAND
	RandHostToDevice();
#endif
#ifndef IMPORT_RAND
	//	HANDLE_CUDA_ERROR(cudaMalloc((void **)&d_randNum, sizeof(real) * sizeRand));
	curandCreateGenerator(&gen_, CURAND_RNG_PSEUDO_XORWOW);
	curandSetPseudoRandomGeneratorSeed(gen_, seed_);

	curandGenerateUniformDouble(gen_, d_rand_sequence_unif_, length_rand_sequence_unif_);
#endif
	return SUCCESS;
}
double * Random::d_rand_sequence_unif()
{
	return d_rand_sequence_unif_;
}
#endif
#ifdef IMPORT_RAND
void Random::RandFileToHost(char * name_file_unif, char * name_file_norm)
{
	FILE *file = fopen(name_file_unif, "r");
	for (int i = 0; i < length_rand_sequence_unif_; i++)
		fscanf(file, "%lf", &h_rand_sequence_unif_[i]);
	fclose(file);

	file = fopen(name_file_norm, "r");
	for (int i = 0; i < length_rand_sequence_norm_; i++)
		fscanf(file, "%lf", &h_rand_sequence_norm_[i]);
}
void Random::RandHostToDevice()
{
	HANDLE_CUDA_ERROR(cudaMemcpy(d_rand_sequence_unif_, h_rand_sequence_unif_, length_rand_sequence_unif_ * sizeof(double), cudaMemcpyHostToDevice));
	HANDLE_CUDA_ERROR(cudaMemcpy(d_rand_sequence_norm_, h_rand_sequence_norm_, length_rand_sequence_norm_ * sizeof(double), cudaMemcpyHostToDevice));
}
#endif

#ifdef DEBUG
void Random::Check(int debug_level)
{
	switch (debug_level)
	{
	case(1) :
#ifdef HOST_RAND
		HANDLE_CUDA_ERROR(cudaMemcpy(h_rand_sequence_unif_, d_rand_sequence_unif_, sizeof(double) * length_rand_sequence_unif_, cudaMemcpyDeviceToHost));
#endif
		break;
	default:
		break;
	}
}
#endif
