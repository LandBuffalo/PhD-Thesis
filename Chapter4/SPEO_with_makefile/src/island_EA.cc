#include "island_EA.h"
IslandEA::IslandEA(const NodeInfo node_info):migrate_(node_info),record_(node_info)
{
    node_info_ = node_info;
}

IslandEA::~IslandEA()
{

}


int IslandEA::Initialize(IslandInfo island_info, ProblemInfo problem_info, EAInfo EA_info)
{
    problem_info_ = problem_info;
    island_info_ = island_info;
    EA_info_ = EA_info;
#ifdef GPU_EA
    EA_ = new DE_CUDA(node_info_);
#else
    EA_ = new DE_CPU(node_info_);
#endif
    EA_->Initialize(island_info_, problem_info_, EA_info_);
    EA_info_.EA_parameters = EA_->GetParameters(EA_info_);
    migrate_.Initialize(island_info_, problem_info_, EA_info_);

    EA_->InitializePopulation(population_);
    record_.Initialize(island_info_, problem_info_, EA_info_);
    record_.CheckAndCreatRecordFile();

    return 0;
}

int IslandEA::Uninitialize()
{
    population_.clear();
    EA_->Uninitialize();
    migrate_.Uninitialize();
    record_.Uninitialize();
    delete EA_;

    return 0;
}

int IslandEA::Execute()
{
    long int generation = 1;
    real current_FEs = island_info_.island_size;
    real start_time = MPI_Wtime();
    real computing_time = 0;
    real comm_time = 0;
    vector<real> record_criterion;
    real total_FEs = problem_info_.max_base_FEs * problem_info_.dim / island_info_.island_num;
#ifndef COMPUTING_TIME
    record_criterion.push_back(total_FEs);
    while(current_FEs <= total_FEs)
#else
    real total_computing_time = problem_info_.computing_time / 120.0;
    record_criterion.push_back(total_computing_time / 100.0);
    record_criterion.push_back(total_computing_time / 10.0);
    record_criterion.push_back(total_computing_time / 2.0);
    record_criterion.push_back(total_computing_time);
    while(computing_time <= problem_info_.computing_time / (120 + 0.0))
#endif
    {
        EA_->Run(population_, EA_info_);
        comm_time += migrate_.ElasticAsyncMigrate(population_, EA_, generation);
#ifdef GPU_EA
        migrate_.Regroup(population_, EA_, &EA_info_, generation);
#endif
        generation++;
        current_FEs += island_info_.island_size;
        computing_time = (real) (MPI_Wtime() - start_time);
        if(record_.RecordFlag(current_FEs, computing_time, record_criterion) == 1)
            record_.RecordResult(current_FEs, computing_time, comm_time, EA_->FindBestIndividual(population_).fitness_value);
    }
    migrate_.Finish();
    record_.GatherResults(0);
    return 0;
}

