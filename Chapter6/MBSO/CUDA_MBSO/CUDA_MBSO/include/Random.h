#ifndef __RANDOM_H__
#define __RANDOM_H__

#include <curand.h>
#include "../include/config.h"
#include "../include/error.h"
#include "../include/KernelConfiguration.h"

extern "C"
void API_setupRandomState(dim3 blocks, dim3 threads, curandState * states, natural seed);

#ifdef DEVICE_RAND
typedef curandStateXORWOW curandState;
#endif
class Random
{
private:

	int						size_pop_;
	int						dim_;
	int						seed_;
	natural					run_;
	KernelConfiguration *	kernel_configuration_;

#ifdef HOST_RAND
	curandGenerator_t		gen_;
	double *				d_rand_sequence_unif_;
	natural					length_rand_sequence_unif_;
	natural					pointer_start_unif_;
#endif

#ifdef DEVICE_RAND
	curandState *			d_rand_states_;
#endif
#ifdef DEBUG
	double *				h_rand_sequence_unif_;
	double *				h_rand_sequence_norm_;
#endif
#ifdef IMPORT_RAND
#ifndef DEBUG
	double *				h_rand_sequence_unif_;
	double *				h_rand_sequence_norm_;
#endif
	void					RandHostToDevice();
#endif
public:
	natural					run();
	void					set_seed(int seed);
							~Random();
#ifdef DEVICE_RAND
							Random(natural size_pop, natural dim, natural run, int ID_device);
	error					initRandom();
	curandState *			d_rand_states();
#endif
							
#ifdef IMPORT_RAND
	void					RandFileToHost(char * name_file_unif, char * name_file_norm);
#endif

#ifdef HOST_RAND
							Random(int length_rand_sequence, natural size_pop, natural dim, natural run, int ID_device);
	double *				d_rand_sequence_unif();
	error					LoopAllocation();
	error					Generate_rand_sequence();
#endif
#ifdef DEBUG
	void					Check(int debug_level);
#endif
};
#endif