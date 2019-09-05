#include "../include/Population.h"

Population::Population(natural size_pop, natural dim)
{
	size_pop_ = size_pop;
	dim_ = dim;

	pop_ = NULL;
	fval_ = NULL;

	pop_ = new double[sizeof(double) * size_pop_ * dim_];
	fval_ = new double[sizeof(double) * size_pop_];

	pop_best_ = new double[dim_ * sizeof(double)];
}

Population :: ~Population()
{
	delete[] pop_;
	delete[] fval_;
	delete[] pop_best_;
}

error Population::InitiPop(double minbound, double maxbound, const double *h_rand_sequence)
{
	for (int i = 0; i < size_pop_; i++)
		for (int j = 0; j < dim_; j++)
			pop_[i + j * size_pop_] = minbound + (maxbound - minbound) * h_rand_sequence[i + j * size_pop_];
	return SUCCESS;
}

void Population::FindIndivualBest()
{
	int index_best = 0;

	fval_best_ = fval_[index_best];
	for (int i = 1; i < size_pop_; i++) 
		if (fval_best_ > fval_[i])
		{
			index_best = i;
			fval_best_ = fval_[i];
		}

	for (int i = 0; i < dim_; i++)
		pop_best_[i] = pop_[index_best + size_pop_ * i];
}

double	Population::fval_best()
{
	return fval_best_;
}

double*	Population::pop()
{
	return pop_;
}

double*	Population::fval()
{
	return fval_;
}
#ifdef DEBUG
void Population::Check()
{
	FILE * file_record;
	file_record = fopen("population.txt", "w");

	for (int i = 0; i < size_pop_; i++)
	{
		for (int j = 0; j < dim_; j++)
			fprintf(file_record, "%.10f\t", pop_[i + j * size_pop_]);
		fprintf(file_record, "\n");
	}
	fclose(file_record);

	file_record = fopen("fitness.txt", "w");

	for (int i = 0; i < size_pop_; i++)
	{
		fprintf(file_record, "%.10f\t", fval_[i]);
	}
	fclose(file_record);

}
#endif