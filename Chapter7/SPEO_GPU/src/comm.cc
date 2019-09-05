#include "comm.h"

Comm::Comm(NodeInfo node_info)
{
    node_info_ = node_info;
        flag_ready_to_send_ = 1;

}

Comm::~Comm()
{
}

int Comm::Initialize(int dim)
{
    dim_ = dim;
    send_msg_to_other_EA_ = new real[dim + 1];

    return 0;
}

int Comm::Uninitialize()
{
    delete []send_msg_to_other_EA_;

    return 0;
}

int Comm::CheckRecv(Message & message, int sender, int tag)
{
    int flag_incoming_msg = 0;
    MPI_Status mpi_status;
    if(sender == -1 && tag == -1)
        MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag_incoming_msg, &mpi_status);
    if(sender == -1 && tag != -1)
        MPI_Iprobe(MPI_ANY_SOURCE, tag, MPI_COMM_WORLD, &flag_incoming_msg, &mpi_status);
    if(sender != -1 && tag == -1)
        MPI_Iprobe(sender, MPI_ANY_TAG, MPI_COMM_WORLD, &flag_incoming_msg, &mpi_status);
    if(sender != -1 && tag != -1)
        MPI_Iprobe(sender, tag, MPI_COMM_WORLD, &flag_incoming_msg, &mpi_status);
    if(flag_incoming_msg == 1)
    {
    	message.sender = mpi_status.MPI_SOURCE;
    	message.receiver = node_info_.node_ID;
    	message.tag = mpi_status.MPI_TAG;
        if(message.flag != -1)
        {
            message.flag = 0;
#ifdef DOUBLE_PRECISION
            MPI_Get_count(&mpi_status, MPI_DOUBLE, &message.msg_length);
#endif
#ifdef SINGLE_PRECISION
            MPI_Get_count(&mpi_status, MPI_FLOAT, &message.msg_length);
#endif   
        }

    }
    return flag_incoming_msg;
}

int Comm::RecvData(Message &message)
{
    MPI_Status mpi_status;

    if(message.flag == 0)
    {
        real * msg_recv = new real[message.msg_length];

#ifdef DOUBLE_PRECISION
	    MPI_Recv(msg_recv, message.msg_length, MPI_DOUBLE, message.sender, message.tag, MPI_COMM_WORLD, &mpi_status);
#endif
#ifdef SINGLE_PRECISION
	    MPI_Recv(msg_recv, message.msg_length, MPI_FLOAT, message.sender, message.tag, MPI_COMM_WORLD, &mpi_status);
#endif 
	    DeserialMsgToIndividual(message.data, msg_recv, message.msg_length / (dim_ + 1));
	    delete [] msg_recv;
    }
    else if(message.flag == -1)
    {
	    MPI_Recv(&message.flag, 1, MPI_INT, message.sender, message.tag, MPI_COMM_WORLD, &mpi_status);
    }

    return 0;
}

int Comm::CheckSend()
{
    MPI_Status mpi_status;

    if(flag_ready_to_send_ == 0)
        MPI_Test(&mpi_request_, &flag_ready_to_send_, &mpi_status);
    
    return flag_ready_to_send_;
}

real Comm::Time()
{
    return MPI_Wtime();
}


int Comm::SendData(Message &message)
{
    if(message.flag == 0)
    {
        if(CheckSend() == 1)
        {
            delete []send_msg_to_other_EA_;
            int msg_length = message.data.size() * (dim_ + 1);
            send_msg_to_other_EA_ = new real[msg_length];
            SerialIndividualToMsg(send_msg_to_other_EA_, message.data);
#ifdef DOUBLE_PRECISION
            MPI_Isend(send_msg_to_other_EA_, msg_length, MPI_DOUBLE, message.receiver, message.tag, MPI_COMM_WORLD, &mpi_request_);
#endif
#ifdef SINGLE_PRECISION
            MPI_Isend(send_msg_to_other_EA_, msg_length, MPI_FLOAT, message.receiver, message.tag, MPI_COMM_WORLD, &mpi_request_);
#endif
            flag_ready_to_send_ = 0;
            return 1;
        }
        return  0;
    }
    else
    {
        if(CheckSend() == 1)
        {
            MPI_Isend(&message.flag, 1, MPI_INT, message.receiver, message.tag, MPI_COMM_WORLD, &mpi_request_);
            flag_ready_to_send_ = 0;
            return  1;
        }
        return  0;
    }
}
int Comm::Cancel()
{
    if(CheckSend() == 0)
    {
        int flag_cancel = 0;
        MPI_Status mpi_stutus;
        while(flag_cancel == 0)
        {
            MPI_Cancel(&mpi_request_);
            MPI_Test_cancelled(&mpi_stutus, &flag_cancel);
        }
        flag_ready_to_send_ = 1;
    }
    return 0;
}

int Comm::DeserialMsgToIndividual(Population &individual, real *msg, int length)
{
    int count = 0;
    individual.clear();
    for (int i = 0; i < length; i++)
    {
        Individual local_individual;
        for(int j = 0; j < dim_; j++)
        {
            local_individual.elements.push_back(msg[count]);
            count++;
        }
        local_individual.fitness_value = msg[count];
        count++;
        individual.push_back(local_individual);
    }
    return 0;
}


int Comm::SerialIndividualToMsg(real *msg, Population &individual)
{
    int count = 0;
    for (int i = 0; i < individual.size(); i++)
    {
        for (int j = 0; j < dim_; j++)
        {
            msg[count] = individual[i].elements[j];
            count++;
        }
        msg[count] = individual[i].fitness_value;
        count++;
    }
    return 0;
}

int Comm::GenerateMsg(vector<Message> & message_queue, Population msg_data, vector<int> destinations, int tag, int pos)
{
    for(int i = 0; i < destinations.size(); i++)
    {
        int flag_merge = 0;
        /*for(int j = 0; j < message_queue.size(); j++)
        {
            if(message_queue[j].receiver == destinations[i])
            {
            	//message_queue[j].data = msg_data;
                message_queue[j].data.insert(message_queue[j].data.end(), msg_data.begin(), msg_data.end());
                flag_merge = 1;
                break;
            }
        }*/
        if(flag_merge == 0)
        {
            Message message;
            message.data = msg_data;
            message.sender = node_info_.node_ID;
            message.receiver = destinations[i];
            message.tag = tag;
            message.flag = 0;
            if(pos == 0)
                message_queue.insert(message_queue.begin(), message);
            else
                message_queue.push_back(message);
        }
    }
    return 0;
}

int Comm::Finish(int base_tag)
{

    return 0;
}