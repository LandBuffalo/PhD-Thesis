#include "../include/Cluster.h"

#ifdef KMEANS
Cluster::Cluster(int num_cluster, int dim, int size_pop)
{
	size_pop_ = size_pop;
	dim_ = dim;

	pop_ = NULL;
	fval_ = NULL;

	num_cluster_ = num_cluster;
	mem_cluster_ = NULL;
	label_pop_ = NULL;
	num_mem_cluster_ = NULL;
	num_accum_mem_cluster_ = NULL;
	probi_mem_cluster_ = NULL;
	centre_ = NULL;

	ID_best_cluster_ = NULL;

	kmeans_ = new KMeans(num_cluster, dim_, size_pop);
}

Cluster::~Cluster()
{
	kmeans_->~KMeans();

	delete[] ID_best_cluster_;
	delete[] mem_cluster_;
	delete[] label_pop_;
	delete[] num_mem_cluster_;
	delete[] num_accum_mem_cluster_;
	delete[] centre_;
	delete[] probi_mem_cluster_;


}

#endif
#ifndef KMEANS
Cluster::~Cluster()
{
	delete[] pop_;
	delete[] fval_;

	delete[] mem_cluster_;
	delete[] num_mem_cluster_;
	delete[] num_accum_mem_cluster_;
	delete[] centre_;

}
#endif
void Cluster::InitiCluster(int num_cluster)
{
	set_num_cluster(num_cluster);
	AllocateSpace();
}
void Cluster::AllocateSpace()
{
	mem_cluster_ = new int[size_pop_];
	label_pop_ = new int[size_pop_];
	num_mem_cluster_ = new int[num_cluster_];
	num_accum_mem_cluster_ = new int[num_cluster_];
	centre_ = new double[dim_ *num_cluster_];
	probi_mem_cluster_ = new double[num_cluster_];
	ID_best_cluster_ = new int[num_cluster_];

}

void Cluster::set_num_cluster(int num_cluster)
{
	num_cluster_ = num_cluster;
}

void Cluster::set_pop_and_fval(double * pop, double *fval)
{
	pop_ = pop;
	fval_ = fval;
}

int Cluster::num_cluster()
{
	return num_cluster_;
}
int * Cluster::mem_cluster()
{
	return kmeans_->mem_cluster();
}
int * Cluster::num_mem_cluster()
{
	return kmeans_->num_mem_cluster();
}
int * Cluster::num_accum_mem_cluster()
{
	return kmeans_->num_mem_accu_cluster();
}

double * Cluster::probi_mem_cluster()
{
	return kmeans_->probi_mem_cluster();
}
double * Cluster::centre()
{
	return kmeans_->centre();
}

int * Cluster::ID_best_cluster()
{
	return ID_best_cluster_;
}

void Cluster::ClusterPop(double * pop, double * fval, double minbound, double maxbound, double * d_rand_sequence_unif)
{
	kmeans_->run_kmeans(pop, fval, centre_, label_pop_, ID_best_cluster_, probi_mem_cluster_, mem_cluster_, num_mem_cluster_, num_accum_mem_cluster_, minbound, maxbound, d_rand_sequence_unif);
#ifdef DEBUG
	Check(1);
#endif

}

void Cluster::initi_centre(double * h_rand_sequence_unif, double maxbound, double minbound)
{
	kmeans_->initi_centre(centre_, h_rand_sequence_unif, maxbound, minbound);
}

#ifdef DEBUG
void Cluster::Check(int level)
{
	FILE * file_record;
	switch (level)
	{
	case(1):

		file_record = fopen("centre.txt", "w");
		for (int k = 0; k < num_cluster_; k++)
		{
			for (int j = 0; j < dim_; j++)
				fprintf(file_record, "%.5f\t", centre_[k + j * num_cluster_]);
			fprintf(file_record, "\n");
		}
		fclose(file_record);
		break;
	case(2) :
		file_record = fopen("num_mem_cluster.txt", "w");

		for (int k = 0; k < num_cluster_; k++)
		{
			fprintf(file_record, "%d\t", num_mem_cluster_[k]);
		}
		fclose(file_record);
		break;
	case(3) :
		file_record = fopen("label_pop.txt", "w");

		for (int i = 0; i < size_pop_; i++)
		{
			fprintf(file_record, "%d\t", label_pop_[i]);
		}
		fclose(file_record);
		break;
	case(4) :
		file_record = fopen("mem_cluster.txt", "w");

		for (int i = 0; i < size_pop_; i++)
		{
			fprintf(file_record, "%d\t", mem_cluster_[i]);
		}
		fclose(file_record);
		break;
	case(5) :
		file_record = fopen("num_accum_mem_cluster.txt", "w");

		for (int i = 0; i < num_cluster_; i++)
		{
			fprintf(file_record, "%d\t", num_accum_mem_cluster_[i]);
		}
		fclose(file_record);
		break;
	case(6) :
		file_record = fopen("best_index.txt", "w");

		for (int k = 0; k < num_cluster_; k++)
		{
			fprintf(file_record, "%d\t", ID_best_cluster_[k]);
		}
		fclose(file_record);
		break;
	default:
		break;
	}

}
#endif