#include "../include/Cluster.h"

#ifdef KMEANS
Cluster::Cluster(int num_cluster, int dim, int size_pop, int ID_device)
{
	size_pop_ = size_pop;
	dim_ = dim;

	d_pop_ = NULL;
	d_fval_ = NULL;

	num_cluster_ = num_cluster;
	d_mem_cluster_ = NULL;
//	d_label_pop_ = NULL;
	d_num_mem_cluster_ = NULL;
	d_num_accum_mem_cluster_ = NULL;
	d_probi_mem_cluster_ = NULL;
	d_centre_ = NULL;

	d_ID_best_cluster_ = NULL;

	kmeans_ = new KMeans(num_cluster, dim_, size_pop, ID_device);

#ifdef DEBUG
	h_pop_ = NULL;
	h_fval_ = NULL;

	h_mem_cluster_ = NULL;
	h_num_mem_cluster_ = NULL;
	h_num_accum_mem_cluster_ = NULL;
	h_centre_ = NULL; 
//	h_label_pop_ = NULL;
	h_ID_best_cluster_ = NULL;
	h_pop_ = new double[dim_ * size_pop_];
	h_fval_ = new double[size_pop_];
	h_mem_cluster_ = new int[size_pop_];
	h_num_mem_cluster_ = new int[num_cluster_];
	h_num_accum_mem_cluster_ = new int[num_cluster_];
	h_centre_ = new double[dim_ * num_cluster_];
//	h_label_pop_ = new int[size_pop_];
	h_ID_best_cluster_ = new int[num_cluster_];
#endif

}

Cluster::~Cluster()
{
	HANDLE_CUDA_ERROR(cudaFree(d_mem_cluster_));
	HANDLE_CUDA_ERROR(cudaFree(d_num_mem_cluster_));
	HANDLE_CUDA_ERROR(cudaFree(d_num_accum_mem_cluster_));
	HANDLE_CUDA_ERROR(cudaFree(d_centre_));
	HANDLE_CUDA_ERROR(cudaFree(d_probi_mem_cluster_));
//	HANDLE_CUDA_ERROR(cudaFree(d_label_pop_));
	HANDLE_CUDA_ERROR(cudaFree(d_ID_best_cluster_));
	kmeans_->~KMeans();
#ifdef DEBUG
	delete[] h_pop_;
	delete[] h_fval_;
	delete[] h_ID_best_cluster_;
//	delete[] h_label_pop_;

	delete[] h_mem_cluster_;
	delete[] h_num_mem_cluster_;
	delete[] h_num_accum_mem_cluster_;
	delete[] h_centre_;
#endif
}

#endif
#ifndef KMEANS
Cluster::~Cluster()
{
	HANDLE_CUDA_ERROR(cudaFree(d_mem_cluster_));
	HANDLE_CUDA_ERROR(cudaFree(d_num_mem_cluster_));
	HANDLE_CUDA_ERROR(cudaFree(d_num_accum_mem_cluster_));
	HANDLE_CUDA_ERROR(cudaFree(d_centre_));
#ifdef DEBUG
	delete[] h_pop_;
	delete[] h_fval_;

	delete[] h_mem_cluster_;
	delete[] h_num_mem_cluster_;
	delete[] h_num_accum_mem_cluster_;
	delete[] h_centre_;
#endif
}
#endif
void Cluster::InitiCluster(int num_cluster)
{
	set_num_cluster(num_cluster);
	AllocateSpace();
}
void Cluster::AllocateSpace()
{
	HANDLE_CUDA_ERROR(cudaMalloc(&d_mem_cluster_, sizeof(int) * size_pop_));
	HANDLE_CUDA_ERROR(cudaMalloc(&d_num_mem_cluster_, sizeof(int) * num_cluster_));
	HANDLE_CUDA_ERROR(cudaMalloc(&d_num_accum_mem_cluster_, sizeof(int) * num_cluster_));
	HANDLE_CUDA_ERROR(cudaMalloc(&d_centre_, sizeof(double) * dim_ * num_cluster_));
//	HANDLE_CUDA_ERROR(cudaMalloc(&d_label_pop_, sizeof(double) * size_pop_));
	HANDLE_CUDA_ERROR(cudaMalloc(&d_probi_mem_cluster_, sizeof(double) * num_cluster_));
	HANDLE_CUDA_ERROR(cudaMalloc(&d_ID_best_cluster_, sizeof(int) * num_cluster_));
}

void Cluster::set_num_cluster(int num_cluster)
{
	num_cluster_ = num_cluster;
}

void Cluster::set_d_pop_and_fval_(double * d_pop, double *d_fval)
{
	d_pop_ = d_pop;
	d_fval_ = d_fval;
}

int Cluster::num_cluster()
{
	return num_cluster_;
}
int * Cluster::d_mem_cluster()
{
	return kmeans_->d_mem_cluster();
}
int * Cluster::d_num_mem_cluster()
{
	return kmeans_->d_num_mem_cluster();
}
int * Cluster::d_num_accum_mem_cluster()
{
	return kmeans_->d_num_mem_accu_cluster();}

double * Cluster::d_probi_mem_cluster()
{
	return kmeans_->d_probi_mem_cluster();
}
double * Cluster::d_centre()
{
	return kmeans_->d_centre();
}
int * Cluster::d_ID_best_cluster()
{
	return d_ID_best_cluster_;
}
void Cluster::ClusterPop(double * d_pop, double * d_fval)
{
	kmeans_->run_kmeans(d_pop, d_fval, d_centre_, d_ID_best_cluster_, d_probi_mem_cluster_, d_mem_cluster_, d_num_mem_cluster_, d_num_accum_mem_cluster_);
}
#ifdef HOST_RAND
void Cluster::initi_centre(dim3 blocks, dim3 threads, double * d_rand_sequence_unif, double maxbound, double minbound)
#endif
#ifdef DEVICE_RAND
void Cluster::initi_centre(dim3 blocks, dim3 threads, curandState *	d_rand_states, double maxbound, double minbound)
#endif
{
#ifdef HOST_RAND
	kmeans_->initi_d_centre(blocks, threads, d_centre_, d_rand_sequence_unif, maxbound, minbound);
#endif
#ifdef DEVICE_RAND
	kmeans_->initi_d_centre(blocks, threads, d_centre_, d_rand_states, maxbound, minbound);
#endif
}


#ifdef DEBUG
void Cluster::Check(int level)
{
	FILE * file_record;
	switch (level)
	{
	case(1) :

		file_record = fopen("centre.txt", "w");
		HANDLE_CUDA_ERROR(cudaMemcpy(h_centre_, d_centre_, sizeof(double) * num_cluster_ * dim_, cudaMemcpyDeviceToHost));

		for (int k = 0; k < num_cluster_; k++)
		{
			for (int j = 0; j < dim_; j++)
				fprintf(file_record, "%.5f\t", h_centre_[k + j * num_cluster_]);
			fprintf(file_record, "\n");
		}
		fclose(file_record);
		break;
	case(2) :
		file_record = fopen("num_mem_cluster.txt", "w");
		HANDLE_CUDA_ERROR(cudaMemcpy(h_num_mem_cluster_, d_num_mem_cluster_, sizeof(int) * num_cluster_, cudaMemcpyDeviceToHost));

		for (int k = 0; k < num_cluster_; k++)
		{
			fprintf(file_record, "%d\t", h_num_mem_cluster_[k]);
		}
		fclose(file_record);
		break;
	case(3) :
		//file_record = fopen("label_pop.txt", "w");
		//HANDLE_CUDA_ERROR(cudaMemcpy(h_label_pop_, d_label_pop_, sizeof(double) * size_pop_, cudaMemcpyDeviceToHost));

		//for (int i = 0; i < size_pop_; i++)
		//{
		//	fprintf(file_record, "%d\t", h_label_pop_[i]);
		//}
		//fclose(file_record);
		break;
	case(4) :
		file_record = fopen("mem_cluster.txt", "w");
		HANDLE_CUDA_ERROR(cudaMemcpy(h_mem_cluster_, d_mem_cluster_, sizeof(int) * size_pop_, cudaMemcpyDeviceToHost));

		for (int i = 0; i < size_pop_; i++)
		{
			fprintf(file_record, "%d\t", h_mem_cluster_[i]);
		}
		fclose(file_record);
		break;
	case(5) :
		file_record = fopen("num_accum_mem_cluster.txt", "w");
		HANDLE_CUDA_ERROR(cudaMemcpy(h_num_accum_mem_cluster_, d_num_accum_mem_cluster_, sizeof(int) * num_cluster_, cudaMemcpyDeviceToHost));

		for (int i = 0; i < num_cluster_; i++)
		{
			fprintf(file_record, "%d\t", h_num_accum_mem_cluster_[i]);
		}
		fclose(file_record);
		break;
	case(6) :
		file_record = fopen("best_index.txt", "w");
		HANDLE_CUDA_ERROR(cudaMemcpy(h_ID_best_cluster_, d_ID_best_cluster_, sizeof(int) * num_cluster_, cudaMemcpyDeviceToHost));

		for (int k = 0; k < num_cluster_; k++)
		{
			fprintf(file_record, "%d\t", h_ID_best_cluster_[k]);
		}
		fclose(file_record);
		break;
	default:
		break;
	}

}
#endif