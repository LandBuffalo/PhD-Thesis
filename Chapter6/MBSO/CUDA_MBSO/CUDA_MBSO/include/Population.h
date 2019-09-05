#ifndef __POPULATION_HH__
#define __POPULATION_HH__

#include "../include/error.h"
#include "../include/config.h"

#ifdef HOST_RAND
extern "C"
void API_initiPop(dim3 blocks, dim3 threads, double * d_pop, natural size_pop, natural dim, double minbound, double maxbound, const double *d_rand_sequence);
#endif
#ifdef DEVICE_RAND
extern "C"
void API_initiPop(dim3 blocks, dim3 threads, double * d_pop, natural size_pop, natural dim, double minbound, double maxbound, curandState * d_rand_states);
#endif
class Population
{
private:
	double * 		d_pop_;		//d_pop is a vector(dim*poprsize) of the population in device
	double *		d_fval_;		//d_fval is a array of the fitness values in device
	double * 		h_pop_;		//h_pop is a vector(dim*popsize) of the population in host
	double *		h_fval_;		//h_fval is a array of the fitness values in host

	double * 		h_pop_best_;		
	double 			h_fval_best_;		

	size_t			d_pitch_pop_;	//d_pitch is the length of d_pop in memory in bytes

	natural			size_pop_;
	natural			dim_;

	void			LoadFromDev();

#ifdef DEBUG
	void			LoadToDev();
#endif

public:
					Population(natural size_pop, natural dim);
					~Population();

	double*			d_pop();
	double*			d_fval();
	double			h_fval_best();

	void			FindIndivualBest();
#ifdef HOST_RAND	
	error			InitiPop(dim3 blocks, dim3 threads, double minbound, double maxbound, const double *d_rand_sequence);
#endif
#ifdef DEVICE_RAND	
	error			InitiPop(dim3 blocks, dim3 threads, double minbound, double maxbound, curandState *	d_rand_states);
#endif

#ifdef DEBUG
	void			Check();
#endif
};
#endif