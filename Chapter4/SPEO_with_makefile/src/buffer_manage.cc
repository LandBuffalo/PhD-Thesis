#include "buffer_manage.h"

BufferManage::BufferManage()
{

}
BufferManage::~BufferManage()
{

}
int BufferManage::Initialize(IslandInfo island_info)
{
    island_info_ = island_info;
}
int BufferManage::Uninitialize()
{
    recv_buffer_.clear();
}
Population BufferManage::SelectFromBuffer(int emigration_num)
{
    Population emigration_export;

//    for(int i = 0; i < emigration_num && recv_buffer_.size() > 0; i++)
//    {
        
//        int best_ID = 0;
        /*
        real best_fitness_value = recv_buffer_[0].fitness_value;
        for(int j = 1; j < recv_buffer_.size(); j++)
        {
            if(best_fitness_value > recv_buffer_[i].fitness_value)
            {
                best_fitness_value = recv_buffer_[i].fitness_value;
                best_ID = j;
            }
        }
        */
//        emigration_export.push_back(recv_buffer_[best_ID]);
//        recv_buffer_.erase(recv_buffer_.begin() + best_ID);
        //recv_buffer_.push_back(emigration_export[i]);
//    }

    emigration_export = recv_buffer_;
    recv_buffer_.clear();
    return emigration_export;
}

real BufferManage::CalDiversity()
{
    real sum = 0;
    if(recv_buffer_.size() > 0)
    {
        for(int i = 0; i < recv_buffer_.size(); i++)
            sum += recv_buffer_[i].fitness_value;
        real mean = sum / (recv_buffer_.size() + 0.0);
        sum = 0;
        for(int i = 0; i < recv_buffer_.size(); i++)
            sum += (recv_buffer_[i].fitness_value - mean) * (recv_buffer_[i].fitness_value - mean);
        return sqrt(sum / (recv_buffer_.size() + 0.0));
    }
    else
    {
        return 0;
    }
}


DiversityPreserving::DiversityPreserving()
{

}

DiversityPreserving::~DiversityPreserving()
{

}
Population DiversityPreserving::SelectFromBuffer(int emigration_num)
{
    Population emigration_export;
    for(int i = 0; i < emigration_num && recv_buffer_.size() > 0; i++)
    {
        
        int best_ID = 0;
        /*
        real best_fitness_value = recv_buffer_[0].fitness_value;
        for(int j = 1; j < recv_buffer_.size(); j++)
        {
            if(best_fitness_value > recv_buffer_[i].fitness_value)
            {
                best_fitness_value = recv_buffer_[i].fitness_value;
                best_ID = j;
            }
        }
        */
        //emigration_export.push_back(recv_buffer_[best_ID]);
        //recv_buffer_.erase(recv_buffer_.begin() + best_ID);
        //recv_buffer_.push_back(emigration_export[i]);
    }
    emigration_export = recv_buffer_;
//    recv_buffer_.clear();
    return emigration_export;
}

int DiversityPreserving::FindNearestIndividual(Individual &individual, Population &recv_buffer)
{

    int min_distance = CalDistance(individual, recv_buffer[0]);
    int nearest_index = 0;
    for(int i = 1; i < recv_buffer.size(); i++)
    {
        real distances = CalDistance(individual, recv_buffer[i]);
        if(min_distance > distances)
        {
            min_distance = distances;
            nearest_index = i;
        }
    }

    return nearest_index;
}
real DiversityPreserving::CalDistance(Individual &individual1, Individual &individual2)
{
    real distance_sum = 0;
    int dim = individual1.elements.size();
    for(int i = 0; i < dim; i++)
        distance_sum += (individual1.elements[i] - individual2.elements[i]) * (individual1.elements[i] - individual2.elements[i]);
    //distance_sum = abs(individual1.fitness_value - individual2.fitness_value);

    return distance_sum;
}

int DiversityPreserving::UpdateBuffer(Individual &individual)
{

        if(recv_buffer_.size() < island_info_.buffer_capacity * island_info_.island_size)
        {
            recv_buffer_.push_back(individual);
        }
        else
        {
            int nearest_individual_ID = FindNearestIndividual(individual, recv_buffer_);
            if(recv_buffer_[nearest_individual_ID].fitness_value > individual.fitness_value)
                recv_buffer_[nearest_individual_ID] = individual;
        }
    
    return 0;
}


BestPreserving::BestPreserving()
{

}

BestPreserving::~BestPreserving()
{

}
int BestPreserving::UpdateBuffer(Individual &individual)
{
    recv_buffer_.push_back(individual);
    while(recv_buffer_.size() > island_info_.buffer_capacity * island_info_.island_size)
    {

        int worst_ID = 0;
        real worst_fitness_value = 0;
        for(int i = 1; i < recv_buffer_.size(); i++)
        {
            if(worst_fitness_value < recv_buffer_[i].fitness_value)
            {
                worst_fitness_value = recv_buffer_[i].fitness_value;
                worst_ID = i;
            }
        }
        recv_buffer_.erase(recv_buffer_.begin() + worst_ID);
    }
    return 0;
}


RandomReplaced::RandomReplaced()
{

}

RandomReplaced::~RandomReplaced()
{

}
int RandomReplaced::UpdateBuffer(Individual &individual)
{
    recv_buffer_.push_back(individual);
    while(recv_buffer_.size() > island_info_.buffer_capacity * island_info_.island_size)
        recv_buffer_.erase(recv_buffer_.begin() + rand() % recv_buffer_.size());
    return 0;
}

FirstReplaced::FirstReplaced()
{

}

FirstReplaced::~FirstReplaced()
{

}
int FirstReplaced::UpdateBuffer(Individual &individual)
{
    recv_buffer_.push_back(individual);
    while(recv_buffer_.size() > island_info_.buffer_capacity * island_info_.island_size)
        recv_buffer_.erase(recv_buffer_.begin());
    return 0;
}
