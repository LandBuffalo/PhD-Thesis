#include "EA_CPU.h"
EA_CPU::EA_CPU()
{

}

EA_CPU::~EA_CPU()
{

}

int EA_CPU::Initialize(IslandInfo island_info, ProblemInfo problem_info, EAInfo EA_info)
{
    problem_info_ = problem_info;   
    island_info_ = island_info;
    cec2014_.Initialize(problem_info_.function_ID, problem_info_.dim);

    return 0;
}

int EA_CPU::InitializePopulation(Population & population)
{
    for(int i = 0; i < island_info_.island_size; i++)
    {
        Individual tmp_individual;

        for (int j = 0; j < problem_info_.dim; j++)
            tmp_individual.elements.push_back(random_.RandRealUnif(problem_info_.min_bound, problem_info_.max_bound));

        tmp_individual.fitness_value = cec2014_.EvaluateFitness(tmp_individual.elements);
        population.push_back(tmp_individual);
    }
    return 0;
}

int EA_CPU::Uninitialize()
{
    cec2014_.Uninitialize();
    return 0;
}
real EA_CPU::CheckBound(real to_check_elements, real min_bound, real max_bound)
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

Individual EA_CPU::FindBestIndividual(Population & population)
{
    int best_individual_ind = 0;
    double best_individual_fitness_value = population[0].fitness_value;
    for(int i = 1; i < island_info_.island_size; i++)
    {
        if(population[i].fitness_value < best_individual_fitness_value)
        {
            best_individual_ind = i;
            best_individual_fitness_value = population[i].fitness_value;
        }
    }
	return population[best_individual_ind];
}

