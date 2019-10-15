#ifndef __RECORD_H__
#define __RECORD_H__
#include <mpi.h>
#include "config.h"
#include <sstream>

class Record
{
private:
    NodeInfo            node_info_;
    IslandInfo          island_info_;
    ProblemInfo         problem_info_;
    EAInfo              EA_info_;
    string 				file_name_;
	vector<Result>		result_;
    vector<Result> 		MergeResults(vector< vector<Result> > result);
	int					PrintResult(vector< vector<Result> > result);
public:
						Record(const NodeInfo node_info);
						~Record();
	int                 Initialize(IslandInfo island_info, ProblemInfo problem_info, EAInfo EA_info);
	int                 Uninitialize();
	int			        CheckAndCreatRecordFile();
    int                 GatherResults(int printer);
    int                 RecordResult(real current_FEs, real computing_time, real comm_time, real FEVs);
    int 				RecordFlag(real current_FEs, real computing_time, vector<real> &record_criterion);
};

#endif
