#include "../include/KMeans.h"
#include "../include/config.h"
//initialize the centre of KMeans by the certain num_centre individuals, it doesnot be called int this program

void host_kmeans(double * pop, double * centre, int * num_mem_cluster, int * mem_cluster, int * label_pop, int * num_mem_accum_cluster, double * probi_mem_cluster, \
	int num_cluster, int size_pop, int dim, double minbound, double maxbound, double * h_rand_sequence_unif)
{
	int* index_tmp = new int[num_cluster];
	double *distance = new double[num_cluster];
	for (int i = 0; i < num_cluster; i++)
		for (int j = 0; j < dim; j++)
			centre[i + j * num_cluster] = minbound + (maxbound - minbound) * h_rand_sequence_unif[i + j * num_cluster];
	//FILE * file_record;
	//file_record = fopen("centre.txt", "w");
	//for (int k = 0; k < num_cluster; k++)
	//{
	//	for (int j = 0; j < dim; j++)
	//		fprintf(file_record, "%.5f\t", centre[k + j * num_cluster]);
	//	fprintf(file_record, "\n");
	//}
	//fclose(file_record);
	//double *distance = new double[num_cluster];
#ifdef DEBUG
	//FILE * file_record;
#endif
	for (int i = 0; i < size_pop; i++)
	{
		for (int j = 0; j < num_cluster; j++)
		{
			distance[j] = 0;
			for (int k = 0; k < dim; k++)
				distance[j] += (pop[i + k * size_pop] - centre[j + k * num_cluster]) * (pop[i + k * size_pop] - centre[j + k * num_cluster]);
		}

#ifdef DEBUG
		//file_record = fopen("distance.txt", "a");
		//for (int k = 0; k < num_cluster; k++)
		//	fprintf(file_record, "%.5f\t", distance[k]);
		//fprintf(file_record, "\n");
		//fclose(file_record);
#endif

		double min_value = distance[0];
		int ID_nearest_cluster = 0;
		for (int j = 1; j < num_cluster; j++)
		{
			if (min_value > distance[j])
			{
				min_value = distance[j];
				ID_nearest_cluster = j;
			}
		}
		if (label_pop[i] != ID_nearest_cluster)
		{
			label_pop[i] = ID_nearest_cluster;
			//			*d_label_change = 1;
		}
	}
	delete[] distance;
	//FILE * file_record;
	//file_record = fopen("label_pop.txt", "w");
	//for (int k = 0; k < size_pop; k++)
	//{
	//		fprintf(file_record, "%d\t", label_pop[k]);
	//}
	//fclose(file_record);




	for (int j = 0; j < num_cluster; j++)
	{
		num_mem_cluster[j] = 0;
		for (int i = 0; i < size_pop; i++)
			if (label_pop[i] == j)
				num_mem_cluster[j]++;
	}

	num_mem_accum_cluster[0] = 0;
	index_tmp[0] = 0;
	probi_mem_cluster[0] = num_mem_cluster[0] / (size_pop + 0.0);
	for (int i = 1; i < num_cluster; i++)
	{
		index_tmp[i] = 0;
		num_mem_accum_cluster[i] = num_mem_accum_cluster[i - 1] + num_mem_cluster[i - 1];
		probi_mem_cluster[i] = num_mem_cluster[i] / (size_pop + 0.0) + probi_mem_cluster[i - 1];
	}
	for (int i = 0; i < size_pop; i++)
	{
		int ID_cluster = label_pop[i];
		mem_cluster[num_mem_accum_cluster[ID_cluster] + index_tmp[ID_cluster]] = i;
		index_tmp[ID_cluster]++;
	}
	delete[] index_tmp;
	//for (int j = 0; j < num_cluster; j++)
	//{
	//	if (num_mem_cluster[j] == 0)
	//	{
	//		distance[0] = 0;
	//		for (int k = 0; k < dim; k++)
	//			distance[0] += (pop[0 + k * size_pop] - centre[j + k * num_cluster]) * (pop[0 + k * size_pop] - centre[j + k * num_cluster]);
	//		double min_distance = 1e99;
	//		int ID_min_pop = 0;
	//		for (int i = 0; i < size_pop; i++)
	//		{
	//			distance[i] = 0;
	//			for (int k = 0; k < dim; k++)
	//				distance[i] += (pop[i + k * size_pop] - centre[j + k * num_cluster]) * (pop[i + k * size_pop] - centre[j + k * num_cluster]);
	//			if (min_distance > distance[i] && num_mem_cluster[label_pop[i]] > 1)
	//			{
	//				min_distance = distance[i];
	//				ID_min_pop = i;
	//			}
	//		}
	//		num_mem_cluster[label_pop[ID_min_pop]]--;
	//		label_pop[ID_min_pop] = j;
	//		num_mem_cluster[j] = 1;
	//	}
	//}

}



KMeans::KMeans(int num_cluster, int dim, int size_pop)
{
	num_cluster_ = num_cluster;
	dim_ = dim;
	size_pop_ = size_pop;

	centre_ = NULL;
	num_mem_cluster_ = NULL;
	mem_cluster_ = NULL;
	label_pop_ = NULL;
	num_accum_mem_cluster_ = NULL;
	label_changed_ = NULL;
	probi_mem_cluster_ = NULL;
	ID_best_cluster_ = NULL;

}
KMeans::~KMeans()
{
}
double * KMeans::centre()
{
	return centre_;
}
int * KMeans::num_mem_cluster()
{
	return num_mem_cluster_;
}

int * KMeans::mem_cluster()
{
	return mem_cluster_;
}

int	* KMeans::num_mem_accu_cluster()
{
	return num_accum_mem_cluster_;
}
double * KMeans::probi_mem_cluster()
{
	return probi_mem_cluster_;
}
void KMeans::set_mem_variable(double * pop, double * fval, double * centre, int * label_pop, int * ID_best_cluster, double * probi_mem_cluster, int * mem_cluster, int * num_mem_cluster, int * num_accum_mem_cluster)
{
	pop_ = pop;
	fval_ = fval;
	centre_ = centre;
	ID_best_cluster_ = ID_best_cluster;
	num_mem_cluster_ = num_mem_cluster;
	mem_cluster_ = mem_cluster;
	num_accum_mem_cluster_ = num_accum_mem_cluster;
	probi_mem_cluster_ = probi_mem_cluster;
	label_pop_ = label_pop;
}

void KMeans::set_centre()
{


	double * fval_best_cluster = new double[num_cluster_];
	for (int k = 0; k < num_cluster_; k++)
	{
		if (num_mem_cluster_[k] == 0)
		{
			ID_best_cluster_[k] = -1;
		}
		else
		{
			ID_best_cluster_[k] = *(mem_cluster_ + *(num_accum_mem_cluster_ + k));
			fval_best_cluster[k] = fval_[ID_best_cluster_[k]];
		}
	}
	//find the best individual in each cluster
	for (int i = 0; i < size_pop_; i++)
	{
		if (fval_best_cluster[label_pop_[i]] > fval_[i])
		{
			fval_best_cluster[label_pop_[i]] = fval_[i];
			ID_best_cluster_[label_pop_[i]] = i;
		}
	}
	delete[] fval_best_cluster;

	//calculate the centre of cluster
	for (int k = 0; k < num_cluster_; k++)
	{
		if (num_mem_cluster_[k] != 0)
			for (int j = 0; j < dim_; j++)
				centre_[k + j * num_cluster_] = pop_[ID_best_cluster_[k] + j * size_pop_];
	}

}
void KMeans::run_kmeans(double * pop, double * fval, double * centre, int * label_pop, int * ID_best_cluster, double * probi_mem_cluster, int * mem_cluster, \
	int * num_mem_cluster, int * num_accum_mem_cluster, double minbound, double maxbound, double * d_rand_sequence_unif)
{
	set_mem_variable(pop, fval, centre, label_pop, ID_best_cluster, probi_mem_cluster, mem_cluster, num_mem_cluster, num_accum_mem_cluster);

	host_kmeans(pop_, centre_, num_mem_cluster_, mem_cluster_, label_pop_, num_accum_mem_cluster_, probi_mem_cluster_, num_cluster_, size_pop_, dim_, minbound, maxbound, d_rand_sequence_unif);
	set_centre();
}

void KMeans::initi_centre(double * centre, double * h_rand_sequence_unif, double maxbound, double minbound)
{
	centre_ = centre;
	for (int k = 0; k < num_cluster_; k++)
		for (int j = 0; j < dim_; j++)
			centre_[k + j*num_cluster_] = minbound + (maxbound - minbound) * h_rand_sequence_unif[k + j*num_cluster_];
}

void KMeans::Check(int level)
{


	switch (level)
	{
	case(1):

	default:
		break;
	}
}