#ifndef __CEC2014_H__
#define __CEC2014_H__

#include "../include/config.h"

#define MAX_NUM_COMP_FUNC 5

class  CEC2014
{
private:
	int						max_feval_base_;					//the base of max function evaluation times, in CEC2014, it is set as 10000
	int						max_feval_;							//the max function evaluation times, which is DIM * 10000

	natural					ID_func_;							//ID of test function ID in (0, 29)
	natural					size_pop_;							//size of population
	natural					dim_;								//dimension of problem

	double					minbound_;							//minimum bound of searching space
	double					maxbound_;							//maximum bound of searching space


	double *				fval_;								//array of fitness value
	double *				pop_original_;						//population without rotation and shift

	void					set_max_feval();													//set the max function evaluation times

public:
							CEC2014(natural ID_func, natural size_pop, natural dim);			//construction function of CEC2014, it sets member variables 
							~CEC2014();

	void					set_pop_original_and_fval(double * pop_original, double * fval);	//load population and fitness value from ones to be evaluated

	natural					ID_func();															//return the ID of thest function
	int						max_feval();														//return the max function evaluation times
	double					minbound();															//return the minimum bound of searching space
	double					maxbound();															//return the maximum bound of searching space

	void					EvaluateFitness(double * fval, double * pop);						//fitness evalating the giving population
};

#endif