#ifndef __MIGRATE_HH__
#define __MIGRATE_HH__
#include "random.h"
#include "config.h"
#include "buffer_manage.h"
#include "comm.h"

#ifdef GPU_EA
	#include "EA_CUDA.h"
	typedef EA_CUDA EA;
#else
	#include "EA_CPU.h"
	typedef EA_CPU EA;
#endif

class Migrate
{
private:
    NodeInfo                node_info_;
    IslandInfo              island_info_;
    ProblemInfo             problem_info_;
    EAInfo                  EA_info_;

    Random                  random_;
    BufferManage *          buffer_manage_;
    Comm                    comm_;
    vector<Message>         message_queue_;
    int 					migration_counter_;
    vector<int>             BestOrWorst(Population &population, int select_num, string flag);
    int                     MigrationCriteria(int generation);
    int                     PrepareEmigrations(Population &population);
    int                     RecvImmigrations(Population &population, EA * EA);
    int                     InsertIntoIsland(Population & population, Population & immigrations);
#ifndef DUAL_CONTROL
    vector<int>             SelectDestination();
#endif 
public:
                            Migrate(const NodeInfo node_info);
                            ~Migrate();

    int                     Initialize(IslandInfo island_info, ProblemInfo problem_info, EAInfo EA_info);
    int                     Uninitialize();
    int                     Finish();
    real                    ElasticAsyncMigrate(Population &population, EA * EA, long int generation);
#ifdef GPU_EA
    int                     Regroup(Population & population, EA * EA, EAInfo *EA_info, long int generation);
#endif
};
#endif
