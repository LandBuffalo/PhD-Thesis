#ifndef __KMEANS__
#define __KMEANS__
#include "../include/KernelConfiguration.h"
#ifdef GPU_KMEANS
extern "C" void API_global_Clustering(dim3 blocks, dim3 threads, double * d_centre, int * d_label_pop, int * d_num_accum_mem_cluster, double *  d_probi_mem_cluster, int * d_mem_cluster, int * d_num_mem_cluster, \
	double * d_pop, double * d_fval, int num_cluster, int size_pop, int dim, double * distance_pop_cluster, int *d_ID_best_cluster, cudaStream_t	stream);
#endif
#ifdef HOST_RAND
extern "C" void API_global_initiCentre(dim3 blocks, dim3 threads, double * d_centre, double * d_rand_sequence_unif, int num_cluster, int dim, double minbound, double maxbound);
#endif
#ifdef DEVICE_RAND
extern "C" void API_global_initiCentre(dim3 blocks, dim3 threads, double * d_centre, curandState *	d_rand_states, int num_cluster, int dim, double minbound, double maxbound);
#endif

class KMeans
{
private:
	int				size_pop_;							//size of population
	int				dim_;								//dimension of problems
	int  			num_cluster_;						//the number of clusters(centres)
	int				max_ilteration_kmeans_;				//the max ilteration for kmeans

	double *		d_probi_mem_cluster_;				//the accumulate probility of each cluster based on number of members in cluster
	int *			d_ID_best_cluster_;					//of population, they are sorted by the membership of clusters)
	double *		d_centre_;							//coordinate of centres
	double *		d_distance_pop_cluster_;
	int	*			d_num_accum_mem_cluster_;			//the accumulate number of members of clusters
	int *			d_num_mem_cluster_;					//the number of members of each cluster
	int *			d_mem_cluster_;						//the array of membership of clusters(the member is the ID 

#ifdef	GPU_KMEANS
	double *		d_pop_;								//population for kmeans clustering
	double *		d_fval_;							//fitness of population for kmeans clustering
//	int	*			d_num_accum_mem_cluster_;			//the accumulate number of members of clusters
	int	*			d_label_changed_;					//the label to record whether the centre is changed from last to next generation
	int *			d_label_pop_;						//label of individual in population with cluster IDs
//	int *			d_num_mem_cluster_;					//the number of members of each cluster
//	int *			d_mem_cluster_;						//the array of membership of clusters(the member is the ID 
	KernelConfiguration	*	kernel_configuration_;
	cudaStream_t	stream;
#endif
#ifdef	CPU_KMEANS
	double *		h_pop_;								//population for kmeans clustering
	double *		h_fval_;							//fitness of population for kmeans clustering
	double *		h_centre_;							//coordinate of centres
	double *		h_probi_mem_cluster_;				//the accumulate probility of each cluster based on number of members in cluster
	int	*			h_ID_best_cluster_;


	int *			h_label_pop_;						//label of individual in population with cluster IDs
	int *			h_num_mem_cluster_;					//the number of members of each cluster
	int *			h_mem_cluster_;						//the array of membership of clusters(the member is the ID 																											//of population, they are sorted by the membership of clusters)
	int	*			h_num_accum_mem_cluster_;			//the accumulate number of members of clusters
	int	*			h_label_changed_;					//the label to record whether the centre is changed from last to next generation
#endif
	void			set_d_mem_variable(double * d_pop, double * d_fval, double * d_centre, int * d_ID_best_cluster, double * d_probi_mem_cluster, \
					int * d_mem_cluster, int * d_num_mem_cluster, int * d_num_accum_mem_cluster);
	void			set_centre();
public:
					KMeans(int num_cluster, int dim, int size_pop, int ID_device);
					~KMeans();

	double *		d_centre();							//return centre
	int *			d_ID_best_cluster();
	int  *			d_num_mem_cluster();				//return num_mem_cluster
	int  *			d_mem_cluster();					//return mem_cluster
	int	*			d_num_mem_accu_cluster();			//return num_mem_accu_cluster
	double *		d_probi_mem_cluster();				//return probi_mem_cluster																		
														//set member variables
	void			run_kmeans(double * d_pop, double * d_fval, double * d_centre_, int * d_ID_best_cluster, double * d_probi_mem_cluster, int * d_mem_cluster, \
					int * d_num_mem_cluster, int * d_num_accum_mem_cluster);
#ifdef HOST_RAND
	void			initi_d_centre(dim3 blocks, dim3 threads, double * d_centre, double * d_rand_sequence_unif, double maxbound, double minbound);	//initiliza centre randomly 
#endif
#ifdef DEVICE_RAND
	void			initi_d_centre(dim3 blocks, dim3 threads, double * d_centre, curandState * d_rand_states, double maxbound, double minbound);	//initiliza centre randomly 
#endif 
};

#endif