#include "record.h"
Record::Record(const NodeInfo node_info)
{
    node_info_ = node_info;
}

Record::~Record()
{

}
int Record::Initialize(IslandInfo island_info, ProblemInfo problem_info, EAInfo EA_info)
{
    island_info_ = island_info;
    problem_info_ = problem_info;
    EA_info_ = EA_info;

    return 0;
}
int Record::Uninitialize()
{
    result_.clear();
    return 0;
}
int Record::CheckAndCreatRecordFile()
{
#ifdef GPU_EA
    #ifdef DUAL_CONTROL
        file_name_ = "./Results/SPEO_GPU_DUAL";
    #else
        file_name_ = "./Results/SPEO_GPU_SINGLE";
    #endif
#else
    #ifdef DUAL_CONTROL
        file_name_ = "./Results/SPEO_CPU_DUAL";
    #else
        file_name_ = "./Results/SPEO_CPU_SINGLE";
    #endif       
#endif
#ifdef COMPUTING_TIME
    file_name_ += "_time.csv";
#else
    file_name_ += ".csv";
#endif

    ifstream exist_file;
    exist_file.open(file_name_.c_str());
    ofstream file;

    if(!exist_file)
    {
        file.open(file_name_.c_str());
        file<< "function_ID,run_ID,dim,FEVs,computing_time,comm_time,total_FEs,node_num,GPU_num,island_num,global_pop_size,island_size,interval,migration_rate,connection_rate,buffer_capacity,migration_topology,buffer_manage,regroup_option,EA_parameters"<<endl;
        file.close();
    }
    else
        exist_file.close();

    return 0;
}

vector<Result> Record::MergeResults(vector< vector<Result> > result)
{
    vector<Result> merged_result;
    int num_records = result[0].size();
    for(int i = 0; i < island_info_.island_num; i++)
    {
        if(num_records != result[i].size())
            printf("error: num_records=%d, actual_record=%d", num_records, result[i].size());
    }
    for(int i = 0; i < num_records; i++)
    {
        Result rmp_merged_result;
        rmp_merged_result.FEVs = result[0][i].FEVs;
        rmp_merged_result.FEs = result[0][i].FEs;
        rmp_merged_result.comm_time = result[0][i].comm_time;
        rmp_merged_result.computing_time = result[0][i].computing_time;

        for(int j = 1; j < island_info_.island_num; j++)
        {
            rmp_merged_result.computing_time += result[j][i].computing_time;
            rmp_merged_result.comm_time += result[j][i].comm_time;
            rmp_merged_result.FEs += result[j][i].FEs;
            if(rmp_merged_result.FEVs > result[j][i].FEVs)
                rmp_merged_result.FEVs = result[j][i].FEVs;
        }
        rmp_merged_result.computing_time = rmp_merged_result.computing_time / (island_info_.island_num + 0.0);
        rmp_merged_result.comm_time = rmp_merged_result.comm_time / (island_info_.island_num + 0.0);
        merged_result.push_back(rmp_merged_result);
    }

    return merged_result;
}

int Record::PrintResult(vector< vector<Result> > result)
{
    vector<Result> merged_result = MergeResults(result);
    printf("Fun: %d, Run: %d\n", problem_info_.function_ID, problem_info_.run_ID);

	ofstream file;
    file.open(file_name_.c_str(), ios::app);

    int global_pop_size = island_info_.island_num * island_info_.island_size;

    for(int i = 0; i < merged_result.size(); i++)
        file<<problem_info_.function_ID<<','<<problem_info_.run_ID<<','<<problem_info_.dim<<','<<fixed<<setprecision(15)<<merged_result[i].FEVs<<','<<fixed<<setprecision(2)<<merged_result[i].computing_time<<','<<fixed<<setprecision(2)<<merged_result[i].comm_time<<','<<floor(merged_result[i].FEs)<<','<<node_info_.node_num<<','<<node_info_.GPU_num<<','<<island_info_.island_num<<','<<global_pop_size<<','<<island_info_.island_size<<','<<island_info_.interval<<','<<fixed<<setprecision(4)<<island_info_.migration_rate<<','<<fixed<<setprecision(4)<<island_info_.connection_rate<<','<<island_info_.buffer_capacity<<','<<island_info_.migration_topology<<','<<island_info_.buffer_manage<<','<<island_info_.regroup_option<<','<<EA_info_.EA_parameters<<endl;

	file.close();
    return 0;
}
int Record::RecordResult(real current_FEs, real computing_time, real comm_time, real FEVs)
{
    Result result;
    result.FEVs = FEVs;
    result.FEs = current_FEs;
    result.comm_time = comm_time;
    result.computing_time = computing_time;
    result_.push_back(result);
    return 0;
}
int Record::RecordFlag(real current_FEs, real computing_time, vector<real> &record_criterion)
{
    int flag = 0;

#ifndef COMPUTING_TIME
    if(current_FEs > record_criterion[0])
#else
    if(computing_time > record_criterion[0])
#endif
    {
        flag = 1;
        record_criterion.erase(record_criterion.begin());
    }

    return flag;
}

int Record::GatherResults(int printer)
{
    if (node_info_.node_ID != printer)
    {
        int result_num = result_.size();
        real *msg = new real[4 * result_num];
        for(int i = 0; i < result_num; i++)
        {
            msg[4 * i] = result_[i].comm_time;
            msg[4 * i + 1] = result_[i].FEVs;
            msg[4 * i + 2] = result_[i].computing_time;
            msg[4 * i + 3] = result_[i].FEs;
        }

        int tag = problem_info_.function_ID * 1000 +  10 * problem_info_.run_ID + RECORD;
#ifdef DOUBLE_PRECISION
        MPI_Send(msg, 4 * result_num, MPI_DOUBLE, printer, tag, MPI_COMM_WORLD);
#endif
#ifdef SINGLE_PRECISION
        MPI_Send(msg, 4 * result_num, MPI_FLOAT, printer, tag, MPI_COMM_WORLD);
#endif
        delete [] msg;
    }
    else
    {
        MPI_Status mpi_status;
        int tag = problem_info_.function_ID * 1000 +  10 * problem_info_.run_ID + RECORD;
        vector< vector<Result> > result;
        for(int i = 0; i < island_info_.island_num; i++)
        {
            vector<Result> tmp_result;
            result.push_back(tmp_result);
        }
        result[printer] = result_;

        for(int i = 1; i < island_info_.island_num; i++)
        {
            MPI_Status mpi_status;
            int length_msg = 0;
            MPI_Probe(MPI_ANY_SOURCE, tag, MPI_COMM_WORLD, &mpi_status);

#ifdef DOUBLE_PRECISION
            MPI_Get_count(&mpi_status, MPI_DOUBLE, &length_msg);

            real *msg = new real[length_msg];
            MPI_Recv(msg, length_msg, MPI_DOUBLE, mpi_status.MPI_SOURCE, tag, MPI_COMM_WORLD, &mpi_status);
#endif
#ifdef SINGLE_PRECISION
            MPI_Get_count(&mpi_status, MPI_FLOAT, &length_msg);
            real *msg = new real[length_msg];
            MPI_Recv(msg, length_msg, MPI_FLOAT, mpi_status.MPI_SOURCE, tag, MPI_COMM_WORLD, &mpi_status);
#endif
            int island_ID = mpi_status.MPI_SOURCE / COMP_CORE_RATIO;
            for (int i = 0; i < length_msg / 4; i++)
            {
                Result tmp_result;
                tmp_result.comm_time = msg[4 * i];
                tmp_result.FEVs = msg[4 * i + 1];
                tmp_result.computing_time = msg[4 * i + 2];
                tmp_result.FEs = msg[4 * i + 3];
                result[island_ID].push_back(tmp_result);
     //           printf("%d\t%d\n", island_ID, result[island_ID].size());

            }
            delete []msg;
        }
        PrintResult(result);
    }
    return 0;
}
