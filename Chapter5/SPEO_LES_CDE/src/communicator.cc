#include "communicator.h"
Communicator::Communicator(NodeInfo node_info):comm_(node_info)
{
    node_info_ = node_info;
}

Communicator::~Communicator()
{

}

int Communicator::Initialize(IslandInfo island_info, ProblemInfo problem_info)
{
    island_info_ = island_info;
    problem_info_ = problem_info;
    buffer_manage_ = new BufferManage();
 
    buffer_manage_->Initialize(island_info_, problem_info);
    comm_.Initialize(problem_info_.dim);

    return 0;
}
int Communicator::Uninitialize()
{
    buffer_manage_->Uninitialize();
    delete buffer_manage_;
    message_queue_.clear();
    comm_.Uninitialize();
    return 0;
}

vector<int> Communicator::SelectDestination()
{
    vector<int> destinations;
    int current_island_ID = node_info_.node_ID / COMP_CORE_RATIO;
    if(island_info_.migration_topology == "dynamic")
    {
        int tmp = random_.RandIntUnif(0, island_info_.island_num - 1);
        while(tmp == current_island_ID)
            tmp = random_.RandIntUnif(0, island_info_.island_num - 1);
        destinations.push_back(tmp);
    }
    if(island_info_.migration_topology == "EDT")
    {
        vector<int> tmp = random_.Permutate(island_info_.island_num, ceil(island_info_.connection_rate * island_info_.island_num));
        for(int i = 0; i < tmp.size(); i++)
        {
            if(tmp[i] != current_island_ID)
                destinations.push_back(tmp[i]);
        }
    }
    if(island_info_.migration_topology == "chain")
        if(current_island_ID < island_info_.island_num - 1)
            destinations.push_back(current_island_ID+ 1);
    if(island_info_.migration_topology == "ring")
    {
        destinations.push_back((current_island_ID + 1) % island_info_.island_num);
        destinations.push_back((current_island_ID - 1 + island_info_.island_num) % island_info_.island_num);
    }
    if(island_info_.migration_topology == "lattice")
    {
        destinations.push_back((current_island_ID + 2) % island_info_.island_num);
        destinations.push_back((current_island_ID + 1) % island_info_.island_num);
        destinations.push_back((current_island_ID - 1 + island_info_.island_num) % island_info_.island_num);
        destinations.push_back((current_island_ID - 2 + island_info_.island_num) % island_info_.island_num);
    }

    return destinations;
}

int Communicator::Execute()
{
    int base_tag = 1000 * problem_info_.function_ID + 10 * problem_info_.run_ID;

    int flag_finish = 0;

    int flag_send_finish = 0;
    Population immigrations_pool;

    while(flag_finish != island_info_.island_num || message_queue_.size() > 0)
    {
    
        Message msg;
        msg.flag = 0;
        if(comm_.CheckRecv(msg, node_info_.node_ID - 1, base_tag + GPU_LAUNCHER_TO_COMM) == 1)
        {
            comm_.RecvData(msg);
            if(flag_send_finish == 0)
            {
                Population send_data; //= buffer_manage_->SelectFromBuffer((int) ceil(island_info_.migration_rate * island_info_.island_size));

                if(send_data.size() > 0)
                {
                    vector<int> receiver(1, node_info_.node_ID - 1);
                    comm_.GenerateMsg(message_queue_, send_data, receiver, base_tag + COMM_TO_GPU_LAUNCHER, 0);
                }

                vector<int> receiver = SelectDestination();
                for (int i = 0; i < receiver.size(); i++)
                    receiver[i] = COMP_CORE_RATIO * receiver[i] + 1;
                comm_.GenerateMsg(message_queue_, msg.data, receiver, base_tag + COMM_TO_COMM, 1);
            }
        }

        Message msg_from_COMM;
        msg_from_COMM.flag = 0;
        if(comm_.CheckRecv(msg_from_COMM, -1, base_tag + COMM_TO_COMM) == 1)
        {
            comm_.RecvData(msg_from_COMM);
            if(flag_send_finish == 0)
            {
                for (int i = 0; i < msg_from_COMM.data.size(); i++)
                {
                    //buffer_manage_->UpdateBuffer(msg_from_COMM.data[i]);
                    Message tmp_msg_from_COMM;
                    tmp_msg_from_COMM.flag = -1;
                    if (comm_.CheckRecv(tmp_msg_from_COMM, -1, -1) == 1)
                        break;
                }
            }
        }

        Message msg_flag;
        msg_flag.flag = -1;
        if(comm_.CheckRecv(msg_flag, -1, base_tag + FINISH) == 1)
        {
            comm_.RecvData(msg_flag);
            flag_finish++;
            if(msg_flag.sender == node_info_.node_ID - 1)
                flag_send_finish = 1;
        }

        if(flag_send_finish == 1)
            flag_send_finish = SendFlagFinish();

        if(message_queue_.size() > 0 && comm_.SendData(message_queue_[0]))
            message_queue_.erase(message_queue_.begin());
    }
    return 0;
}

int Communicator::SendFlagFinish()
{
    Message message;
    int base_tag = 1000 * problem_info_.function_ID + 10 * problem_info_.run_ID;

    message.flag = 1;
    message.tag = base_tag + FINISH;
    message_queue_.clear();
    //comm_.Cancel();
    if(comm_.CheckSend() == 1)
    {
        for (int i = 0; i < island_info_.island_num; i++)
        {
            if(node_info_.node_ID != COMP_CORE_RATIO * i + 1)
            {
                message.receiver = COMP_CORE_RATIO * i + 1;
                message_queue_.push_back(message);
            }
            else
            {
                message.receiver = node_info_.node_ID - 1;
                message_queue_.push_back(message);   
            }
        }
        return -1;
    }

    return 1;
}
/*
int Communicator::MergeImmigrations(Population &incoming_population, Population &immigrations_pool)
{
    if(immigrations_pool.size() == 0)
    {
        immigrations_pool = incoming_population;
        incoming_population.clear();
    }

    while(incoming_population.size() > 0)
    {
        int best_ID = 0;
        real_float min_distance = 1e20;
        for(int i = 0; i < immigrations_pool.size(); i++)
        {
            real_float tmp_distance_sum = 0;
            for(int d = 0; d < problem_info_.dim; d++)
                tmp_distance_sum += (incoming_population[0].elements[d] - immigrations_pool[i].elements[d]) * (incoming_population[0].elements[d] - immigrations_pool[i].elements[d]);
            if(min_distance > tmp_distance_sum)
            {
                min_distance = tmp_distance_sum;
                best_ID = i;
            }
        }
        if(incoming_population[0].fitness_value < immigrations_pool[best_ID].fitness_value)
            immigrations_pool[best_ID].fitness_value = incoming_population[0].fitness_value;
        incoming_population.erase(incoming_population.begin());
    }

    return 0;
}*/
