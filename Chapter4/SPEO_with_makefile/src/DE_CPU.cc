#include "EA_CPU.h"

DE_CPU::DE_CPU(NodeInfo node_info)
{
    node_info_ = node_info;
}

DE_CPU::~DE_CPU()
{
    EA_CPU::Uninitialize();
}

string DE_CPU::GetParameters(DEInfo DE_info)
{

    string str;
    ostringstream temp1, temp2;
    string parameters = "CR/F=";
    double CR = DE_info.CR;
    temp1<<CR;
    str=temp1.str();
    parameters.append(str);

    parameters.append("/");
    double F = DE_info.F;
    temp2<<F;
    str=temp2.str();
    parameters.append(str);

    if(DE_info.strategy_ID == 0)
        parameters.append("_current/1/bin");
    else if(DE_info.strategy_ID == 1)
        parameters.append("_current/2/bin");
    else if(DE_info.strategy_ID == 2)
        parameters.append("_current-to-best/1/bin");
    else if(DE_info.strategy_ID == 3)
        parameters.append("_current-to-best/2/bin");
    else if(DE_info.strategy_ID == 4)
        parameters.append("_rand/1/bin");
    else if(DE_info.strategy_ID == 5)
        parameters.append("_rand/2/bin");
    else if(DE_info.strategy_ID == 6)
        parameters.append("_best/1/bin");
    else if(DE_info.strategy_ID == 7)
        parameters.append("_best/2/bin");
    else if(DE_info.strategy_ID == 8)
        parameters.append("_current_to_rand/1/bin");
    return parameters;
}

int DE_CPU::Initialize(IslandInfo island_info, ProblemInfo problem_info, DEInfo DE_info)
{
	EA_CPU::Initialize(island_info, problem_info, DE_info);
    DE_info_ = DE_info;
	return 0;
}
int DE_CPU::ConfigureEA(DEInfo DE_info)
{
    DE_info_ = DE_info;
    return 0;
}


int DE_CPU::InitializePopulation(Population & population)
{
    EA_CPU::InitializePopulation(population);

    return 0;
}

int DE_CPU::Uninitialize()
{
    return 0;
}

int DE_CPU::Reproduce(Population & population)
{
    Individual best_individual = FindBestIndividual(population);

    double F = DE_info_.F;
    double CR = DE_info_.CR;
    for (int i = 0; i < island_info_.island_size; i++)
    {
        Individual tmp_individual = population[i];
        vector<int> r = random_.Permutate(island_info_.island_size, 5);

        for (int j = 0; j < problem_info_.dim; j++)
        {
            switch (DE_info_.strategy_ID)
            {
                case 0:
                    tmp_individual.elements[j] = population[i].elements[j] + F * (population[r[0]].elements[j] - population[r[1]].elements[j]);
                    break;
                case 1:
                    tmp_individual.elements[j] = population[i].elements[j] + F * (population[r[0]].elements[j] - population[r[1]].elements[j]) + \
                    + F * (population[r[2]].elements[j] - population[r[3]].elements[j]);
                    break;
                case 2:
                    tmp_individual.elements[j] = population[i].elements[j] + F * (best_individual.elements[j] - population[i].elements[j]) + \
                    + F * (population[r[0]].elements[j] - population[r[1]].elements[j]);
                    break;
                case 3:
                    tmp_individual.elements[j] = population[i].elements[j] + F * (best_individual.elements[j] - population[i].elements[j]) + \
                    + F * (population[r[0]].elements[j] - population[r[1]].elements[j]) + F * (population[r[2]].elements[j] - population[r[3]].elements[j]);
                    break;
                case 4:
                    tmp_individual.elements[j] = population[r[0]].elements[j] + F * (population[r[1]].elements[j] - population[r[2]].elements[j]);
                    break;
                case 5:
                    tmp_individual.elements[j] = population[r[0]].elements[j] + F * (population[r[1]].elements[j] - population[r[2]].elements[j]) + \
                    + F * (population[r[3]].elements[j] - population[r[4]].elements[j]);
                    break;
                case 6:
                    tmp_individual.elements[j] = best_individual.elements[j] + F * (population[r[0]].elements[j] - population[r[1]].elements[j]);
                    break;
                case 7:
                    tmp_individual.elements[j] = best_individual.elements[j] + F * (population[r[0]].elements[j] - population[r[1]].elements[j]) + \
                    + F * (population[r[2]].elements[j] - population[r[3]].elements[j]);
                    break;
                case 8:
                    tmp_individual.elements[j] = population[i].elements[j] + F * (population[r[0]].elements[j] - population[i].elements[j]) + \
                    + F * (population[r[1]].elements[j] - population[r[2]].elements[j]) + F * (population[r[3]].elements[j] - population[r[4]].elements[j]);
                    break;
                default:
                    break;
            }
            if (random_.RandRealUnif(0, 1) > CR && j != random_.RandIntUnif(0, problem_info_.dim - 1))
                tmp_individual.elements[j] = population[i].elements[j];
            tmp_individual.elements[j] = CheckBound(tmp_individual.elements[j], problem_info_.min_bound, problem_info_.max_bound);
        }
        tmp_individual.fitness_value = cec2014_.EvaluateFitness(tmp_individual.elements);

        if (tmp_individual.fitness_value < population[i].fitness_value)
        {
            population[i].fitness_value = tmp_individual.fitness_value;
            population[i].elements = tmp_individual.elements;
        }
    }
    return 0;
}


int DE_CPU::Run(Population & population, DEInfo EA_info)
{
    Reproduce(population);
    //SelectSurvival(population);
    return 0;
}
