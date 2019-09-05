#ifndef __RANDOM_H__
#define __RANDOM_H__

#include "curand.h"
#include "../include/config.h"
#include "../include/error.h"



class Random
{
private:
	natural					length_rand_sequence_unif_;				//length of uniform random number sequence
	natural					length_rand_sequence_norm_;				//length of normal random number sequence
	natural					pointer_start_unif_;					//pointer to uniform random number sequence
	natural					pointer_start_norm_;					//pointer to normal random number sequence

	double *				h_rand_sequence_unif_;					//uniform random number sequence
	double *				h_rand_sequence_norm_;					//normal random number sequence

	int						size_pop_;								//size of population
	int						dim_;									//dimension of problem
	int						seed_;									//seed of instance
	natural					run_;									//ID of runs

	curandGenerator_t		gen_;									//cuda curand RNG

#ifdef IMPORT_RAND
#ifndef DEBUG
	double *				h_rand_sequence_unif_;
	double *				h_rand_sequence_norm_;
#endif
#endif
public:
							Random(int length_rand_sequence, int length_rand_sequence_norm,natural size_pop, natural dim, natural run);
							~Random();

	error					Generate_rand_sequence();				//generate random number sequence

	error					LoopAllocationRandSequence();			//allocate the random number sequence to specific stragety
#ifdef IMPORT_RAND
	void					RandFileToHost(char * name_file_unif, char * name_file_norm);	// load random number sequence from file
#endif

	natural					run();									//return run

	void					set_seed(int seed);						//set seed

	double *				h_rand_sequence_unif();					//return uniform random number sequence
	double *				h_rand_sequence_norm();					//return normal random number sequence
#ifdef DEBUG
	void					Check(int debug_level);
#endif
};
#endif