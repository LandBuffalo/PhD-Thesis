#ifndef __COMM_H__
#define __COMM_H__
#include <mpi.h>
#include "config.h"

class Comm
{
private:
    MPI_Request             mpi_request_;
    NodeInfo                node_info_;
    int                     dim_;
    int                     flag_ready_to_send_;
    real_float *                  send_msg_to_other_EA_;
    Population              Best(Population &population, int select_num);
    int                     SerialIndividualToMsg(real_float *msg_of_node_EA, Population &individual);
    int                     DeserialMsgToIndividual(Population &individual, real_float *msg_of_node_EA, int length);
    int                     FindNearestIndividual(Individual &individual, Population &population);

public:
                            Comm(const NodeInfo node_info);
                            ~Comm();
    int                     Initialize(int dim);
    int                     Uninitialize();
    int                     GenerateMsg(vector<Message> & message_queue, Population msg_data, vector<int> destinations, int tag, int pos);
    int                     CheckRecv(Message & message, int sender, int tag);
    int                     SendData(Message &message);
    int                     RecvData(Message &message);
    int                     Cancel();
    int                     CheckSend();
    int                     Finish(int base_tag);
    real_float                    Time();

};
#endif