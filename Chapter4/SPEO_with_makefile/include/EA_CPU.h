#ifndef __CPU_EA_LIBRARY_H__
#define __CPU_EA_LIBRARY_H__
#include <stdio.h>
#include "config.h"
#include "CEC2014.h"
#include "random.h"
#include <sstream>

class EA_CPU
{
protected:
    ProblemInfo             problem_info_;
    IslandInfo              island_info_;
    NodeInfo                node_info_;
    CEC2014                 cec2014_;
    Random                  random_;
    real                    CheckBound(real to_check_elements, real min_bound, real max_bound);

public:
                            EA_CPU();
                            ~EA_CPU();
    virtual int             InitializePopulation(Population & population);
    virtual int             Initialize(IslandInfo island_info, ProblemInfo problem_info, EAInfo EA_info);
    virtual int             Uninitialize();


    Individual              FindBestIndividual(Population & population);
    virtual string          GetParameters(DEInfo DE_info) = 0;
    virtual int             Run(Population & population, EAInfo EA_info) = 0;
    virtual int             ConfigureEA(EAInfo EA_info) = 0;
};

class DE_CPU : public EA_CPU
{
private:
    DEInfo                  DE_info_;
    int                     Reproduce(Population & population);
    int                     SelectSurvival(Population & population);
    virtual int             InitializePopulation(Population & population);
public:
                            DE_CPU(NodeInfo node_info);
                            ~DE_CPU();
    virtual int             Initialize(IslandInfo island_info, ProblemInfo problem_info, DEInfo DE_info);
    virtual int             Uninitialize();
    virtual int             Run(Population & population, DEInfo DE_info);
    virtual string          GetParameters(DEInfo DE_info);
    virtual int             ConfigureEA(DEInfo DE_info);    
};

#endif
