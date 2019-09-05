#include "elm.h"

ELM::ELM()
{

}

ELM::~ELM()
{

}
int ELM::Initialize(IslandInfo island_info, ProblemInfo problem_info)
{
    island_info_ = island_info;
    problem_info_ = problem_info;
    flag_initialized_ = 0;
    model_.L = island_info_.island_size;

}

int ELM::RecvData(Individual & incoming_individual_data)
{
    train_data_pool_.push_back(incoming_individual_data);
    return 0;
}

int ELM::Train()
{

    if(flag_initialized_ == 0)
    {
        InitialiseELM();
        flag_initialized_ = 1;
    }
    else
    {
        OnlineTrain();
    }

    return 0;
}
real_float ELM::Predict(Individual & test_data)
{
    Population tmp_test_data(1, test_data);
    MatrixXd h = CalHValue(tmp_test_data);
    MatrixXd output = h * model_.out_weight;

    return output(0,0);
}
int ELM::InitialiseELM()
{
    Population train_data = train_data_pool_;
    train_data_pool_.clear();


    MatrixXd tmp_input_weight(model_.L, problem_info_.dim);
    MatrixXd tmp_bias(1, model_.L);
    model_.input_weight = tmp_input_weight;
    model_.bias = tmp_bias;

    for (int i = 0; i < model_.L; i++)
    {
        for (int j = 0; j < problem_info_.dim; j++)
            model_.input_weight(i,j) = random_.RandRealUnif(-1, 1);
        model_.bias(0,i) = random_.RandRealUnif(0, 1);
    }


    MatrixXd  T(train_data.size(), 1);
    for (int i = 0; i < train_data.size(); i++)
        T(i, 0) = train_data[i].fitness_value;

    MatrixXd  h = CalHValue(train_data);
    model_.M = (h.transpose() * h).inverse();
    model_.out_weight = model_.M * h.transpose() * T;

    return 0;
}
int ELM::OnlineTrain()
{
    while(train_data_pool_.size() > island_info_.buffer_capacity)
    {
        Population train_data;
        for (int i = 0; i < island_info_.buffer_capacity; i++)
        {
            train_data.push_back(train_data_pool_[0]);
            train_data_pool_.erase(train_data_pool_.begin());
        }
        MatrixXd  h = CalHValue(train_data);
        MatrixXd  T(train_data.size(), 1);
        for (int i = 0; i < train_data.size(); i++)
            T(i,0) = train_data[i].fitness_value;

        model_.M = model_.M - model_.M * h.transpose() * (MatrixXd::Identity(train_data.size(), train_data.size()) + h * model_.M * h.transpose()).inverse() * h * model_.M;
        model_.out_weight = model_.out_weight + model_.M * h.transpose() * (T - h * model_.out_weight);
    }
    return 0;
}




MatrixXd ELM::CalHValue(Population & train_data)
{
    MatrixXd  h(train_data.size(), model_.L);

    for (int i = 0; i < train_data.size(); i++)
    {
        for (int j = 0; j < model_.L; j++)
        {
            real_float sum = 0;
            for (int k = 0; k < problem_info_.dim; k++)
                sum += model_.input_weight(j,k) * train_data[i].elements[k];
            h(i,j) = ActiveFunction(sum + model_.bias(0, j));
        }
    }
    return h;
}



real_float ELM::ActiveFunction(real_float input)
{
    real_float output = 1.0 / (1.0 + pow(E_2, -input));
    return output;
}
