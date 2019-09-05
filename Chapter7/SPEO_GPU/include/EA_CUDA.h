#ifndef __CUDA_EA_LIBRARY_H__
#define __CUDA_EA_LIBRARY_H__
#include "curand.h"
#include "curand_kernel.h"
#include "cuda_runtime.h"
#include "cuda.h"
#include <stdio.h>
#include "config.h"
#include "CEC2014_CUDA.h"
#include "CEC2014.h"

#include "random.h"
#include <sstream>

#define MAX_THREAD 512
#define WARP_SIZE 32
#define MAX_POOL_SIZE 64
#define TILE_WIDTH 16

//population in GPU global memory
struct GPU_Population
{
    real *elements;
    real *fitness_value;
};


extern "C"
int API_Initialize(GPU_Population d_population, curandState *d_rand_states, EAInfo EA_info, ProblemInfo problem_info);
extern "C"
int API_CheckCUDAError();

class EA_CUDA
{
protected:
    ProblemInfo             problem_info_;
    IslandInfo              island_info_;
    NodeInfo                node_info_;
    CEC2014_CUDA            cec2014_cuda_;
    CEC2014                 cec2014_;

    Random                  random_;
    GPU_Population          d_population_;
    GPU_Population          d_candidate_;
    GPU_Population          h_population_;

    curandState *           d_rand_states_;
    int                     CalNextPow2Dim();
    real                    CheckBound(real to_check_elements, real min_bound, real max_bound);   //check bound of individuals
public:
                            EA_CUDA();
                            ~EA_CUDA();

    int                     InitializePopulation(Population & population);       //Initialize the population in GPU global memory
    virtual int             Initialize(IslandInfo island_info, ProblemInfo problem_info, EAInfo EA_info);
    virtual int             Uninitialize();

    int                     TransferDataToCPU(Population &population);          //transfer date from GPU globale memory to CPU RAM
    int                     TransferDataFromCPU(Population &population);        //transfer date from CPU RAM to GPU globale memory
    int                     VerifyCorrectness(Population &population);          //verify the correctness of GPU EA operations by compare with CPU EA operations

    virtual Individual      FindBestIndividual(Population & population);        //find the best individuals in population of GPU global memory
    Population              FindBestIndividualInIslands(Population & population);
    virtual int             Run(Population & population, EAInfo EA_info) = 0;
    virtual string          GetParameters(DEInfo DE_info) = 0;
    virtual int             ConfigureEA(EAInfo EA_info)=0;

};

class DE_CUDA : public EA_CUDA
{
private:
    int *                   d_best_individual_ID_;
    int                     Reproduce_CPU(Population & candidate, Population & population);
    int                     SelectSurvival_CPU(Population & population, Population & candidate);
public:
                            DE_CUDA(NodeInfo node_info);
                            ~DE_CUDA();
    virtual int             Initialize(IslandInfo island_info, ProblemInfo problem_info, DEInfo DE_info);
    virtual int             Uninitialize();
    virtual int             Run(Population & population, DEInfo DE_info);
    virtual string          GetParameters(DEInfo DE_info);
    virtual int             ConfigureEA(DEInfo DE_info);

};


#endif
