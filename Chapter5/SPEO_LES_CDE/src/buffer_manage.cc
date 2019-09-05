#include "buffer_manage.h"

BufferManage::BufferManage()
{

}
BufferManage::~BufferManage()
{

}

int BufferManage::Initialize(IslandInfo island_info, ProblemInfo problem_info)
{
    island_info_ = island_info;
    problem_info_ = problem_info;
    ELM tmp_ELM;
    for (int i = 0; i < island_info_.model_num; i++)
    {
        tmp_ELM.Initialize(island_info_, problem_info_);
        ELM_.push_back(tmp_ELM);
    }
    return 0;
}

int BufferManage::Uninitialize()
{
    ELM_.clear();
    return 0;
}
int BufferManage::RecvData(Population & immigrations, Population &population)
{
    for (int i = 0; i < immigrations.size(); i++)
    {
        vector<real_float> distance = CalDistance(immigrations[i], population);
        for(int j = 0; j < island_info_.k_nearest; j++)
        {
            int nearest_index = 0;
            real_float min_distance = distance[0];

            for(int n = 1; n < population.size(); n++)
            {
                if(min_distance > distance[n])
                {
                    min_distance = distance[n];
                    nearest_index = n;
                }
            }
            ELM_[nearest_index].RecvData(immigrations[i]);
            distance[nearest_index] = 1e20;
        }
    }
    return 0;
}

int BufferManage::MajorityVote(Population & candidates, Population &population)
{
    for (int i = 0; i < ELM_.size(); i++)
    {
        ELM_[i].Train();
    }

    Population tmp_candidates;
    Population wait_candidates;

    for(int i = 0; i < candidates.size(); i++)
    {
        vector<real_float> predict_value = Predict(candidates[i], population);
        int success_rate = 0;
        real_float sum_fitness_value = 0;
        for (int j = 0; j < predict_value.size(); j++)
        {
            sum_fitness_value += predict_value[i];
            if (predict_value[i] < population[i].fitness_value)
                success_rate++;
        }
        if(success_rate > predict_value.size() / 2)
        {
            tmp_candidates.push_back(candidates[i]);
            if(tmp_candidates.size() == population.size())
                break;

        }
        else
        {
            candidates[i].fitness_value = sum_fitness_value / (predict_value.size() + 0.0);
            wait_candidates.push_back(candidates[i]);
        }
    }

    while(tmp_candidates.size() < population.size())
    {
        int best_index = 0;
        real_float best_fitness_value = wait_candidates[0].fitness_value;

        for (int i = 1; i < wait_candidates.size(); i++)
        {
            if (best_fitness_value > wait_candidates[i].fitness_value)
            {
                best_fitness_value = wait_candidates[i].fitness_value;
                best_index = i;
            }
        }
        wait_candidates[best_index].fitness_value = 1e20;
        tmp_candidates.push_back(wait_candidates[best_index]);
    }
    candidates = tmp_candidates;

    return 0;
}

vector<real_float> BufferManage::Predict(Individual & candidates_individual, Population &population)
{
    vector<real_float> predict_value;

    vector<real_float> distance = CalDistance(candidates_individual, population);
    for(int j = 0; j < island_info_.k_nearest; j++)
    {
        int nearest_index = 0;
        real_float min_distance = distance[0];

        for(int n = 1; n < population.size(); n++)
        {
            if(min_distance > distance[n])
            {
                min_distance = distance[n];
                nearest_index = n;
            }
        }
        distance[nearest_index] = 1e20;

        predict_value.push_back(ELM_[nearest_index].Predict(candidates_individual));
    }
    return predict_value;
}



vector<real_float> BufferManage::CalDistance(Individual & individual1, Population & population)
{
    vector<real_float> distance;
    int dim = individual1.elements.size();
    for (int i = 0; i < population.size(); i++)
    {
        distance.push_back(0);
        for(int j = 0; j < problem_info_.dim; j++)
            distance[i] += (individual1.elements[j] - population[i].elements[j]) * (individual1.elements[j] - population[i].elements[j]);
    }

    return distance;
}


