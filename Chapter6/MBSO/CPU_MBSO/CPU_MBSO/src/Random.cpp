#include "../include/Random.h"
#include "../include/config.h"

#ifdef HOSTCURAND
Random::Random(int length_rand_sequence_unif, int length_rand_sequence_norm, natural size_pop, natural dim, natural run)
{
	length_rand_sequence_unif_ = length_rand_sequence_unif;
	length_rand_sequence_norm_ = length_rand_sequence_norm;
	pointer_start_unif_ = 0;
	pointer_start_norm_ = 0;
	size_pop_ = size_pop;
	dim_ = dim;
	run_ = run;

	h_rand_sequence_unif_ = NULL;
	h_rand_sequence_norm_ = NULL;
	pointer_start_unif_ = dim_ * size_pop_;

	h_rand_sequence_unif_ = new double[length_rand_sequence_unif_];
	h_rand_sequence_norm_ = new double[length_rand_sequence_norm_];

#ifdef IMPORT_RAND
#ifndef DEBUG
	h_rand_sequence_ = NULL;
	h_rand_sequence_ = new double[sizeof(double) * length_rand_sequence_];
#endif
#endif
}

Random::~Random()
{
#ifdef IMPORT_RAND
#endif
	delete[] h_rand_sequence_unif_;
	delete[] h_rand_sequence_norm_;
}

error Random::LoopAllocationRandSequence()
{
	return SUCCESS;
}
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

#endif
error Random::Generate_rand_sequence()
{
#ifdef IMPORT_RAND
//	RandHostToDevice();
#endif
#ifndef IMPORT_RAND
	//	HANDLE_CUDA_ERROR(cudaMalloc((void **)&d_randNum, sizeof(real) * sizeRand));
	curandCreateGeneratorHost(&gen_, CURAND_RNG_PSEUDO_XORWOW);
	curandSetPseudoRandomGeneratorSeed(gen_, seed_);

	curandGenerateUniformDouble(gen_, h_rand_sequence_unif_, length_rand_sequence_unif_);
#endif
	return SUCCESS;
}

natural	Random::run()
{
	return run_;
}

double * Random::h_rand_sequence_unif()
{
	return h_rand_sequence_unif_;
}
double * Random::h_rand_sequence_norm()
{
	return h_rand_sequence_norm_;
}
void Random::set_seed(int seed)
{
	seed_ = seed + run_ * size_pop_;
}

#endif