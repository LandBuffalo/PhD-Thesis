#include "../include/config.h"
#include "../include/KMeans.h"

class Cluster
{
private:
	double *		pop_;								//data(populatin) for clustering
	double *		fval_;								//fitness value of data(population) of clustering
	int				size_pop_;							//size of data(size of population)
	int				dim_;								//diemsion of data(problem)

	int				num_cluster_;						//the number of clusters(centres)
	int *			ID_best_cluster_;					//the ID of best individual in population of each cluster
	int *			mem_cluster_;						//the array of membership of clusters(the member is the ID 
														//of population, they are sorted by the membership of clusters)
	int *			label_pop_;							//the label of population individuals
	int *			num_mem_cluster_;					//the number of members of each cluster
	int *			num_accum_mem_cluster_;				//the accumulate number of members of clusters
	double *		probi_mem_cluster_;					//the accumulate probility of each cluster based on number of members in cluster
	double *		centre_;							//the coordinate of centre

	KMeans *		kmeans_;							//the object of KMeans to cluster data

	void			AllocateSpace();					//allocate space for the member variables
	void			set_num_cluster(int num_cluster);	//set the num_cluster

public:
#ifdef KMEANS
					Cluster(int num_cluster, int dim, int size_pop);						//construction function of cluster object
#endif
					~Cluster();
	void			InitiCluster(int num_cluster);															//initilize the obejct, it calls AllocateSpace() and set_num_cluster()
	void			set_pop_and_fval(double * pop, double *fval);											//load the population and its fitness values from user for clustering

	int				num_cluster();																			//return num_cluster
	int *			mem_cluster();																			//return mem_cluster
	int *			num_mem_cluster();																		//return num_mem_cluster
	int *			num_accum_mem_cluster();																//return num_accum_mem_cluster
	double *		probi_mem_cluster();																	//return probi_mem_cluster
	double *		centre();																				//return centre
	int *			ID_best_cluster();																		//return ID_best_cluster

	void			set_centre();																			//set centre based on BSO centre generation, it choose the best individual in
																											//each cluster

	void			ClusterPop(double * pop, double * fval, double minbound, double maxbound, double * d_rand_sequence_unif);												//main program for clustering the population
	void			initi_centre(double *h_rand_sequence_unif, double maxbound, double minbound);			//initilise the centres of clusters
#ifdef DEBUG
	void			Check(int level);																		//for debug model
#endif
};