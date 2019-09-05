#ifndef __BUFFER_MANAGE__
#define __BUFFER_MANAGE__

#include "random.h"
#include <algorithm>
#include "comm.h"
class BufferManage
{
protected:
    Population          recv_buffer_;
    Population          tmp_recv_buffer_;

    Random              random_;
    IslandInfo          island_info_;
public:
                        BufferManage();
    int                 Initialize(IslandInfo island_info);
    int                 Uninitialize();
    virtual             ~BufferManage();
    virtual int         UpdateBuffer(Individual &individual) = 0;
    virtual Population  SelectFromBuffer(int emigration_num);

    real                CalDiversity();
};

class DiversityPreserving: public BufferManage
{
protected:
    real                CalDistance(Individual &individual1, Individual &individual2);
    int                 FindNearestIndividual(Individual &individual, Population &recv_buffer);
public:
                        DiversityPreserving();
    virtual             ~DiversityPreserving();
    virtual int         UpdateBuffer(Individual &individual);
    virtual Population  SelectFromBuffer(int emigration_num);

};

class BestPreserving : public BufferManage
{
private:
public:
                        BestPreserving();
    virtual             ~BestPreserving();
    virtual int         UpdateBuffer(Individual &individual);
};

class RandomReplaced : public BufferManage
{
private:
public:
                        RandomReplaced();
    virtual             ~RandomReplaced();
    virtual int         UpdateBuffer(Individual &individual);
};

class FirstReplaced : public BufferManage
{
private:
public:
                        FirstReplaced();
    virtual             ~FirstReplaced();
    virtual int         UpdateBuffer(Individual &individual);
};

#endif
