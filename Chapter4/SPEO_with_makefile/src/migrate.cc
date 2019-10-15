#include "migrate.h"

Migrate::Migrate(NodeInfo node_info):comm_(node_info)
{
    node_info_ = node_info;
}

Migrate::~Migrate()
{

}

int Migrate::Initialize(IslandInfo island_info, ProblemInfo problem_info, EAInfo EA_info)
{
    island_info_ = island_info;
    problem_info_ = problem_info;
    EA_info_ = EA_info;
    if(island_info_.buffer_manage == "diversity")
        buffer_manage_ = new DiversityPreserving();
    if(island_info_.buffer_manage == "first")
        buffer_manage_ = new FirstReplaced();
    if(island_info_.buffer_manage == "best")
        buffer_manage_ = new BestPreserving();
    if(island_info_.buffer_manage == "random")
        buffer_manage_ = new RandomReplaced();
    buffer_manage_->Initialize(island_info_);
    comm_.Initialize(problem_info_.dim);
    migration_counter_ = 0;
    return 0;
}
int Migrate::Uninitialize()
{
    buffer_manage_->Uninitialize();
    delete buffer_manage_;
    message_queue_.clear();
    comm_.Uninitialize();

    return 0;
}

int Migrate::Finish()
{
    int base_tag = 1000 * problem_info_.function_ID + 10 * problem_info_.run_ID;
#ifdef DUAL_CONTROL
    Message message;
    message.flag = 1;
    message.tag = base_tag + FINISH;
    message.receiver = node_info_.node_ID + 1;
    int flag_send = 0;
    int flag_recv = 0;

    while(flag_send * flag_recv == 0)
    {
        if(flag_send == 0 && comm_.SendData(message) == 1)
            flag_send = 1;

        Message msg_flag;
        if(comm_.CheckRecv(msg_flag, node_info_.node_ID + 1, base_tag + FINISH) == 1)
        {
            msg_flag.flag = -1;
            comm_.RecvData(msg_flag);
            flag_recv = 1;
        }
        Message msg_from_COMM;
        if(comm_.CheckRecv(msg_from_COMM, node_info_.node_ID + 1, base_tag + COMM_TO_GPU_LAUNCHER) == 1)
            comm_.RecvData(msg_from_COMM);
    }
#else
    message_queue_.clear();

    Message message;
    message.flag = 1;
    message.tag = base_tag + FINISH;
    for(int i = 0; i < node_info_.node_num; i++)
    {
        if(node_info_.node_ID !=  i)
        {
            message.receiver = i;
            message.sender = node_info_.node_ID;
            message_queue_.push_back(message);
        }
    }

    int finish = 1;
    while(finish < node_info_.node_num || message_queue_.size() > 0 )
    { 
        Message msg_data;
        msg_data.flag = 0;
        if(comm_.CheckRecv(msg_data, -1, base_tag + MIGRATIONS) == 1)
            comm_.RecvData(msg_data);
        
        Message msg_flag;
        msg_flag.flag = -1;
        if(comm_.CheckRecv(msg_flag, -1, base_tag + FINISH) == 1)
        {
            comm_.RecvData(msg_flag);
            finish++;
        }

        if(message_queue_.size() > 0 && comm_.SendData(message_queue_[0]))
            message_queue_.erase(message_queue_.begin());    
    }
#endif
    return 0;
}

int Migrate::MigrationCriteria(int generation)
{
    if(generation % island_info_.interval == 0)
        return 1;
    else
        return 0;
}
vector<int> Migrate::BestOrWorst(Population &population, int select_num, string flag)
{
    Population tmp_population(population);
    vector<int> selected_individuals;

    for(int i = 0; i < select_num; i++)
    {
        int index = 0;
        real fitness_value = tmp_population[0].fitness_value;
        for(int j = 1; j < population.size(); j++)
        {
            int tmp_flag1 = ((flag == "best") && (fitness_value > tmp_population[i].fitness_value));
            int tmp_flag2 = ((flag == "worst") && (fitness_value < tmp_population[i].fitness_value));

            if(tmp_flag1 || tmp_flag2)
            {
                fitness_value = tmp_population[i].fitness_value;
                index = j;
            }
        }
        selected_individuals.push_back(index);
        tmp_population.erase(tmp_population.begin() + index);
    }

    return selected_individuals;
}


real Migrate::ElasticAsyncMigrate(Population &population, EA * EA, long int generation)
{
    real communication_time = comm_.Time();
    if (MigrationCriteria(generation) == 1)
    {
#ifdef GPU_EA
        EA->TransferDataToCPU(population);
#endif
#ifndef DUAL_CONTROL
        Population immigrations = buffer_manage_->SelectFromBuffer((int) ceil(island_info_.migration_rate * island_info_.island_size));
        PrepareEmigrations(population); 
        InsertIntoIsland(population, immigrations);
    #ifdef GPU_EA
        EA->TransferDataFromCPU(population);
    #endif
#endif
    }
    RecvImmigrations(population, EA);

    while(message_queue_.size() > 0 && comm_.SendData(message_queue_[0]))
        message_queue_.erase(message_queue_.begin());

    return comm_.Time() - communication_time;
}

int Migrate::PrepareEmigrations(Population &population)
{
#ifdef DUAL_CONTROL
    vector<int> receiver(1, node_info_.node_ID + 1);
    int tag = 1000 * problem_info_.function_ID + 10 * problem_info_.run_ID + GPU_LAUNCHER_TO_COMM;
#else
    vector<int> receiver = SelectDestination();
    int tag = 1000 * problem_info_.function_ID + 10 * problem_info_.run_ID + MIGRATIONS;
#endif
    Population emigrations;
    //vector<int> best_individuals_ID = BestOrWorst(population, (int) ceil(island_info_.migration_rate * island_info_.island_size), "best");
    //emigrations.push_back(population[best_individuals_ID[i]]);

    for(int i = 0; i < (int) ceil(island_info_.migration_rate * island_info_.island_size); i++)
    {
        vector<int> rand_index = random_.Permutate(population.size(), 2);
        if(population[rand_index[0]].fitness_value < population[rand_index[1]].fitness_value)
            emigrations.push_back(population[rand_index[0]]);
        else
            emigrations.push_back(population[rand_index[1]]);
    }
    message_queue_.clear();
    comm_.GenerateMsg(message_queue_, emigrations, receiver, tag, 1);

return 0;
}

int Migrate::RecvImmigrations(Population &population, EA * EA)
{
    Message recv_msg;
    recv_msg.flag = 0;

#ifdef DUAL_CONTROL
    int sender = node_info_.node_ID + 1;
    int tag =  1000 * problem_info_.function_ID + 10 * problem_info_.run_ID + COMM_TO_GPU_LAUNCHER;
    while(comm_.CheckRecv(recv_msg, sender, tag) == 1)
    {
        comm_.RecvData(recv_msg);

    #ifdef GPU_EA
        EA->TransferDataToCPU(population);
        InsertIntoIsland(population, recv_msg.data);
        EA->TransferDataFromCPU(population);
    #else
        InsertIntoIsland(population, recv_msg.data);
    #endif
    }
#else
    int sender = -1;
    int tag =  1000 * problem_info_.function_ID + 10 * problem_info_.run_ID + MIGRATIONS;
    while(comm_.CheckRecv(recv_msg, sender, tag) == 1)
    {
        comm_.RecvData(recv_msg);
        for(int i = 0; i < recv_msg.data.size(); i++)
        	buffer_manage_->UpdateBuffer(recv_msg.data[i]);
    }
#endif

    return 0;
}


int Migrate::InsertIntoIsland(Population &population, Population &immigrations)
{
    for (int i = 0; i < immigrations.size(); i++)
    {
        int worst_individual = BestOrWorst(population, 1, "worst")[0];
        if(population[worst_individual].fitness_value > immigrations[i].fitness_value)
            population[worst_individual] = immigrations[i];
        //int rand_index = random_.RandIntUnif(0, population.size() - 1);
        //if(population[rand_index].fitness_value > immigrations[i].fitness_value)
        //population[rand_index] = immigrations[i];
    }
    migration_counter_++;

    return 0;
}


//--------------------------------------------------------------------------------------------
#ifdef GPU_EA
int Migrate::Regroup(Population & population, EA * EA, EAInfo *EA_info, long int generation)
{   
    if(island_info_.regroup_option == "regroup" && MigrationCriteria(generation) == 1)
    {
        int max_index = 0, pop_size = island_info_.island_size / MIN_GROUP_SIZE;
        while (pop_size > 1)
        {
            pop_size = pop_size / 2;
            max_index++;
        }
        EA_info_.group_size = MIN_GROUP_SIZE << random_.RandIntUnif(0, max_index);
        EA_info_.group_num = island_info_.island_size / EA_info_.group_size;
        EA->ConfigureEA(EA_info_);
        *EA_info = EA_info_;
    }
    return 0;
}

#endif

//--------------------------------------------------------------------------------------------
#ifndef DUAL_CONTROL
vector<int> Migrate::SelectDestination()
{
    vector<int> destinations;
    if(island_info_.migration_topology == "dynamic")
    {
        int tmp = random_.RandIntUnif(0, node_info_.node_num - 1);
        while(tmp == node_info_.node_ID)
            tmp = random_.RandIntUnif(0, node_info_.node_num - 1);
        destinations.push_back(tmp);
    }
    if(island_info_.migration_topology == "EDT")
    {
        vector<int> tmp = random_.Permutate(island_info_.island_num, (int) ceil(island_info_.connection_rate * island_info_.island_num));
        for(int i = 0; i < tmp.size(); i++)   
        {
            if(tmp[i] != node_info_.node_ID)
                destinations.push_back(tmp[i]);
        }    
    }
    if(island_info_.migration_topology == "chain")
        if(node_info_.node_ID < node_info_.node_num - 1)
            destinations.push_back(node_info_.node_ID + 1);
    if(island_info_.migration_topology == "ring")
    {
        destinations.push_back((node_info_.node_ID + 1) % island_info_.island_num);
        destinations.push_back((node_info_.node_ID - 1 + island_info_.island_num) % island_info_.island_num);
    }
    if(island_info_.migration_topology == "lattice")
    {
        destinations.push_back((node_info_.node_ID + 2) % island_info_.island_num);
        destinations.push_back((node_info_.node_ID + 1) % island_info_.island_num);
        destinations.push_back((node_info_.node_ID - 1 + island_info_.island_num) % island_info_.island_num);
        destinations.push_back((node_info_.node_ID - 2 + island_info_.island_num) % island_info_.island_num);
    }

    return destinations;
}
#endif







