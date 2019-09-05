#include "../include/config.h"
#include "../include/KMeans.h"

class Cluster
{
private:
	double *		d_pop_;									//data(populatin) for clustering
	double *		d_fval_;								//fitness value of data(population) of clustering
	int				size_pop_;								//size of data(size of population)
	int				dim_;									//diemsion of data(problem)

	int				num_cluster_;							//the number of clusters(centres)
	int *			d_ID_best_cluster_;						//the ID of best individual in population of each cluster
	int *			d_mem_cluster_;							//the array of membership of clusters(the member is the ID 															//of population, they are sorted by the membership of clusters)
//	int *			d_label_pop_;							//the label of population individuals
	int *			d_num_mem_cluster_;						//the number of members of each cluster
	int *			d_num_accum_mem_cluster_;				//the accumulate number of members of clusters
	double *		d_probi_mem_cluster_;					//the accumulate probility of each cluster based on number of members in cluster
	double *		d_centre_;								//the coordinate of centre

	KMeans *		kmeans_;								//the object of KMeans to cluster data

	void			AllocateSpace();						//allocate space for the member variables
	void			set_num_cluster(int num_cluster);		//set the num_cluster


#ifdef DEBUG
	double *		h_pop_;
	double *		h_fval_;
//	int *			h_label_pop_;
	int *			h_mem_cluster_;
	int *			h_num_mem_cluster_;
	int *			h_num_accum_mem_cluster_;
	int *			h_ID_best_cluster_;						
	double *		h_probi_mem_cluster_;					//the accumulate probility of each cluster based on number of members in cluster
	double *		h_centre_;

#endif
public:
#ifdef KMEANS
					Cluster(int num_cluster, int dim, int size_pop, int ID_device);
#endif
					~Cluster();
	void			InitiCluster(int num_cluster);
	void			set_d_pop_and_fval_(double * d_pop, double *d_fval);

	int				num_cluster();
	int *			d_mem_cluster();
	int *			d_num_mem_cluster();
	int *			d_num_accum_mem_cluster();
	double *		d_probi_mem_cluster();
	double *		d_centre();
	int *			d_ID_best_cluster();
																		//set centre based on BSO centre generation, it choose the best individual in
	//each cluster

	void			ClusterPop(double * d_pop, double * d_fval);											//main program for clustering the population
#ifdef HOST_RAND
	void			initi_centre(dim3 blocks, dim3 threads, double *d_rand_sequence_unif, double maxbound, double minbound);			//initilise the centres of clusters
#endif
#ifdef DEVICE_RAND
	void			initi_centre(dim3 blocks, dim3 threads, curandState * d_rand_states, double maxbound, double minbound);			//initilise the centres of clusters
#endif
#ifdef DEBUG
	void			Check(int level);																		//for debug model
#endif
};