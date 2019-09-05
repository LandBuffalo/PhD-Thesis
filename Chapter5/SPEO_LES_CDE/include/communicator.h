#ifndef __COMMUNICATOR_HH__
#define __COMMUNICATOR_HH__
#include "random.h"
#include "config.h"
#include "buffer_manage.h"
#include "comm.h"

class Communicator
{
private:
    Random                  random_;
    IslandInfo              island_info_;
    ProblemInfo             problem_info_;
    NodeInfo                node_info_;
    BufferManage *          buffer_manage_;
    Comm                    comm_;
    vector<Message>         message_queue_;       
    vector<int>             SelectDestination();
    int                     SendFlagFinish();
    //int                     MergeImmigrations(Population &incoming_population, Population &existing_population);

public:
                            Communicator(const NodeInfo node_info);
                            ~Communicator();

    int                     Initialize(IslandInfo island_info, ProblemInfo problem_info);
    int                     Uninitialize();
    int                     Execute();
};
#endif
