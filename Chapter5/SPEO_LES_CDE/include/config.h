#ifndef __CONFIG_H__
#define __CONFIG_H__

#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <string>
#include <cmath>
#include <time.h>
#include <numeric>
#include <string>
#include <iostream>
#include <iomanip>

using namespace std;
//#define SEQUENTIAL
//#define STANDARD_CDE
//#define GPU_EA
//#define DUAL_CONTROL
//#define COMPUTING_TIME
#define GPU_PER_NODE				8
#ifdef GPU_EA
	#define MIN_GROUP_SIZE      	4
#endif

#ifdef DUAL_CONTROL
	#define COMP_CORE_RATIO 		3
#else
	#define COMP_CORE_RATIO 		1
#endif


#define GPU_LAUNCHER_TO_COMM   		1
#define COMM_TO_GPU_LAUNCHER    	2
#define COMM_TO_COMM     			3
#define MIGRATIONS  		 		1
#define FINISH     					0
#define RECORD						4

#define DOUBLE_PRECISION
#ifdef DOUBLE_PRECISION
	typedef double real_float;
#endif
#ifdef SINGLE_PRECISION
	typedef float real_float;
#endif


struct Individual
{
	vector<real_float> elements;
	real_float fitness_value;
};

typedef vector<Individual> Population;

struct Message
{
	Population data;
	int flag;
	int sender;
	int receiver;
	int msg_length;
	int tag;
};

struct ProblemInfo
{
	int dim;
	int function_ID;
	int run_ID;
	int max_base_FEs;
	int seed;
	int computing_time;
	real_float max_bound;
	real_float min_bound;
};

struct NodeInfo
{
    int task_ID;
	int node_ID;
	int node_num;

	int GPU_num;
	int GPU_ID;

};

struct IslandInfo
{
	int island_size;
	int island_num;
	int interval;
    real_float buffer_capacity;
    real_float migration_rate;
	real_float connection_rate;
    string regroup_option;
    string migration_topology;
	string buffer_manage;
	int model_num;
	int k_nearest;
};

struct DEInfo
{
    real_float CR;
    real_float F;
    int strategy_ID;
    int group_size;
	int group_num;
	int candidate_rate;
	string EA_parameters;
};
typedef DEInfo EAInfo;

struct Result
{
    real_float FEs;
    real_float computing_time;
	real_float comm_time;
	real_float FEVs;
};

#endif

