#ifndef __ISLANDEA_H__
#define __ISLANDEA_H__
#pragma once
#include "config.h"
#include "random.h"
#include "migrate.h"
#include "record.h"

#include "EA_CPU.h"

#ifdef DUAL_CONTROL
	#include "comm.h"
#endif

class IslandEA
{
private:

	EA_CPU	*				EA_;



	Migrate					migrate_;
	Record 					record_;
	Random                  random_;
	NodeInfo				node_info_;
	EAInfo					EA_info_;
	ProblemInfo				problem_info_;
	IslandInfo				island_info_;
    Population 				population_;



public:
							IslandEA(const NodeInfo node_info);
							~IslandEA();
	int 					Initialize(IslandInfo island_info, ProblemInfo problem_info, EAInfo EA_info);
	int 					Uninitialize();
	int						Execute();
};

#endif
