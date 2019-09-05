#ifndef __CUDA_BSO_HH__
#define __CUDA_BSO_HH__

#include "../include/config.h"
#include "../include/Population.h"
#include "../include/CEC2014.h"
#include "../include/Random.h"
#include "../include/KernelConfiguration.h"
#include "../include/Cluster.h"
#include "../include/utc_time_stamp.h"

extern "C"
void API_generateNewPop(dim3 blocks, dim3 threads, double *pop_new, double *fval_new, double *pop_candidate, double *pop_old, double *fval_candidate, double *fval_old, size_t size_pop);
#ifdef HOST_RAND
extern "C"
void API_applystrategy(dim3 blocks, dim3 threads, double * d_pop_candidate, double *d_pop_original, int * ID_best_cluster, double *d_centre, int * d_num_mem_cluster, int * d_num_accum_mem_cluster, double *d_probi_mem_cluster, \
int * d_mem_cluster, int num_cluster, double *d_rand_unif, natural size_pop, natural dim, double maxbound, double minbound, double p_replace_centre_rand, double p_select_cluster_one_or_two, double p_use_centre_one_cluster, double p_use_centre_two_cluster, double pr);
#endif
#ifdef DEVICE_RAND
extern "C"
void API_applystrategy(dim3 blocks, dim3 threads, double * d_pop_candidate, double *d_pop_original, int * ID_best_cluster, double *d_centre, int * d_num_mem_cluster, int * d_num_accum_mem_cluster, double *d_probi_mem_cluster, \
int * d_mem_cluster, int num_cluster, curandState * d_rand_states, natural size_pop, natural dim, double maxbound, double minbound, double p_replace_centre_rand, double p_select_cluster_one_or_two, double p_use_centre_one_cluster, double p_use_centre_two_cluster, double pr);
#endif
class CUDA_BSO
{
private:
	natural					current_feval_;
	natural					stid_;
	natural					size_pop_;
	natural					dim_;

	double					p_replace_centre_rand_;
	double					p_select_cluster_one_or_two_;
	double					p_use_centre_one_cluster_;
	double					p_use_centre_two_cluster_;
	double					pr_;
	Cluster *				cluster_;
	Population *			population_;
	Population *			population2_;
	Population *			population_candidate_;
	Random *				random_;
	CEC2014 *				CEC2014_;
	KernelConfiguration	*	kernel_configuration_;

	curandState				* d_rand_states;
	error					EvaluateFitness(double * d_fval, double * d_pop);
	error					GenerateNewPop(natural *flag_stream);
	error					InitiPop();

	void					RecordResults(natural flag_stream, double duration_computation);
	void					DisplayResults(natural flag_stream, double duration_computation);

	void					ClusterPop(natural flag_stream);
	double					GetTime();
#ifdef HOST_RAND
	error					Applystrategy(natural flag_stream, int start_rand);
	error					LoopAllocation();
	error					Generate_rand_sequence();
#endif

#ifdef DEVICE_RAND
	error					Applystrategy(natural flag_stream);
#endif
public:
							CUDA_BSO(int ID_device, natural ID_func, natural run, natural size_pop, natural dim, int seed, \
								int num_cluster, double	p_replace_centre_rand, double p_select_cluster_one_or_two, \
								double p_use_centre_one_cluster, double p_use_centre_two_cluster, double pr);
							~CUDA_BSO();
#ifdef IMPORT_RAND
	void					RandFileToHost(char * name_file_unif, char * name_file_norm);
#endif
	error					BSO();


};
#endif