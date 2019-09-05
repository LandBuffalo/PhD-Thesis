#include "EA_CUDA.h"
extern "C"
int API_DE_GenerateNewPopulation(GPU_Population d_candidate, GPU_Population d_population, int * d_best_individual_ID, curandState *d_rand_states, DEInfo DE_info, ProblemInfo problem_info);
extern "C"
int API_DE_SelectSurvival(GPU_Population d_population, GPU_Population d_candidate, DEInfo DE_info, ProblemInfo problem_info);

DE_CUDA::DE_CUDA(NodeInfo node_info)
{
	node_info_ = node_info;
    cudaSetDevice(node_info_.GPU_ID % 2);
}

DE_CUDA::~DE_CUDA()
{

}

string DE_CUDA::GetParameters(DEInfo DE_info)
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

int DE_CUDA::Initialize(IslandInfo island_info, ProblemInfo problem_info, DEInfo DE_info)
{
	EA_CUDA::Initialize(island_info, problem_info, DE_info);

    cudaMalloc(&d_best_individual_ID_,  DE_info.group_num * sizeof(int));
    cudaMemset(d_best_individual_ID_, 0,  DE_info.group_num * sizeof(int));
    return 0;
}

int DE_CUDA::Uninitialize()
{
    cudaFree(d_best_individual_ID_);

	EA_CUDA::Uninitialize();

    return 0;
}

int DE_CUDA::ConfigureEA(DEInfo DE_info)
{
    cudaFree(d_best_individual_ID_);
    cudaMalloc(&d_best_individual_ID_,  DE_info.group_num * sizeof(int));
    cudaMemset(d_best_individual_ID_, 0,  DE_info.group_num * sizeof(int));

	return 0;
}

int DE_CUDA::Run(Population & population, DEInfo DE_info)
{
	API_DE_GenerateNewPopulation(d_candidate_, d_population_, d_best_individual_ID_, d_rand_states_, DE_info, problem_info_);
	cec2014_cuda_.EvaluateFitness(d_candidate_.fitness_value, d_candidate_.elements);
	API_DE_SelectSurvival(d_population_, d_candidate_, DE_info, problem_info_);

    return 0;
}
