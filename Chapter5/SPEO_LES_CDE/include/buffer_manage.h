#ifndef __BUFFER_MANAGE__
#define __BUFFER_MANAGE__

#include "random.h"
#include <algorithm>
#include "elm.h"
#include "config.h"

class BufferManage
{
protected:
    vector<ELM>         ELM_;
    Random              random_;
    IslandInfo          island_info_;
    ProblemInfo         problem_info_;
    vector<real_float>  CalDistance(Individual & individual1, Population & population);

public:
                        BufferManage();
                        ~BufferManage();
    int                 Initialize(IslandInfo island_info, ProblemInfo problem_info);
    int                 Uninitialize();             
    int                 RecvData(Population & immigrations, Population &population);
    int                 Train();
    int                 MajorityVote(Population & candidates, Population &population);
    vector<real_float> Predict(Individual & candidates_individual, Population &population);
};


#endif
