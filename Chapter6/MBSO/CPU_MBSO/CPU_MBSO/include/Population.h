#ifndef __POPULATION_HH__
#define __POPULATION_HH__

#include "../include/error.h"
#include "../include/config.h"

class Population
{
private:

	double * 		pop_;				//coordinate of population
	double *		fval_;				//fitness value of population

	double * 		pop_best_;			//the best individual one in population	
	double 			fval_best_;			//the best fitness value in population

	natural			size_pop_;			//the size of population
	natural			dim_;				//the dimension of problem

public:
					Population(natural size_pop, natural dim);
					~Population();

	double*			pop();				//return pop
	double*			fval();				//return fval
	double			fval_best();		//return fval_best

	void			FindIndivualBest();	//find the best individual among population
#ifdef HOSTCURAND	
	error			InitiPop(double minbound, double maxbound, const double *h_rand_sequence);	// initilize the pop randomly based on host API random number sequence
#endif

#ifdef DEBUG
	void			Check();
#endif
};
#endif