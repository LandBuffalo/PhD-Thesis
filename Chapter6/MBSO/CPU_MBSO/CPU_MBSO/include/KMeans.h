#ifndef __KMEANS__
#define __KMEANS__

class KMeans
{
private:
	double *		pop_;								//population for kmeans clustering
	double *		fval_;								//fitness of population for kmeans clustering
	int				size_pop_;							//size of population
	int				dim_;								//dimension of problems


	double *		centre_;							//coordinate of centres
	double *		probi_mem_cluster_;					//the accumulate probility of each cluster based on number of members in cluster
	int *			label_pop_;							//label of individual in population with cluster IDs
	int *			num_mem_cluster_;					//the number of members of each cluster
	int *			mem_cluster_;						//the array of membership of clusters(the member is the ID 														
	int *			ID_best_cluster_;					//of population, they are sorted by the membership of clusters)
	int	*			num_accum_mem_cluster_;				//the accumulate number of members of clusters
	int	*			label_changed_;						//the label to record whether the centre is changed from last to next generation
	int  			num_cluster_;						//the number of clusters(centres)

	void			set_centre();
	void			set_mem_variable(double * pop, double * fval, double * centre, int * label_pop, int * ID_best_cluster, double * probi_mem_cluster, int * mem_cluster, \
					int * num_mem_cluster, int * num_accum_mem_cluster);
public:
					KMeans(int num_cluster, int dim, int size_pop);
					~KMeans();
	double  *		centre();							//return centre
	int  *			num_mem_cluster();					//return num_mem_cluster
	int  *			mem_cluster();						//return mem_cluster
	int	*			num_mem_accu_cluster();				//return num_mem_accu_cluster
	double *		probi_mem_cluster();				//return probi_mem_cluster

														//set member variables
	void			run_kmeans(double * pop, double * fval, double * centre, int * label_pop, int * ID_best_cluster, double * probi_mem_cluster, int * mem_cluster, \
					int * num_mem_cluster, int * num_accum_mem_cluster, double minbound, double maxbound, double * d_rand_sequence_unif);						//run kmeans for clusering
	void			initi_centre(double * centre, double * h_rand_sequence_unif, double maxbound, double minbound);	//initiliza centre randomly 
	void			Check(int level);
};

#endif