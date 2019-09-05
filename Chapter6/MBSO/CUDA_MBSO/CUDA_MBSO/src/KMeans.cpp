#include "../include/KMeans.h"
#include "../include/config.h"
#ifndef GPU_KMEANS
void host_initiCentre(double * h_centre, int * h_label_pop, double * h_pop, int num_cluster, int size_pop, int dim)
{

	for (int i = 0; i < num_cluster; i++)
		for (int j = 0; j < dim; j++)
			h_centre[i + j * num_cluster] = h_pop[i + j * size_pop];

}
//calculate the distance between population individuals and centres and rebel the individuals
void host_CalDistAndRebelPop(int * label_pop, int * label_change, double * centre, double * pop, int num_cluster, int size_pop, int dim)
{
	double *distance = new double[num_cluster];
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
}

//calculate the new centre 
void host_FindCentre(double * centre, int * label_pop, double * pop, int * num_mem_cluster, int num_cluster, int size_pop, int dim)
{
	for (int j = 0; j < num_cluster; j++)
	{
		num_mem_cluster[j] = 0;
		for (int i = 0; i < size_pop; i++)
			if (label_pop[i] == j)
				num_mem_cluster[j]++;
		if (num_mem_cluster[j] == 0)
		{
			double *distance = new double[size_pop];
			distance[0] = 0;
			for (int k = 0; k < dim; k++)
				distance[0] += (pop[0 + k * size_pop] - centre[j + k * num_cluster]) * (pop[0 + k * size_pop] - centre[j + k * num_cluster]);
			double longest_distance = distance[0];
			int ID_longest_pop = 0;
			for (int i = 1; i < size_pop; i++)
			{
				distance[i] = 0;
				for (int k = 0; k < dim; k++)
					distance[i] += (pop[i + k * size_pop] - centre[j + k * num_cluster]) * (pop[i + k * size_pop] - centre[j + k * num_cluster]);
				if (longest_distance < distance[i])
				{
					longest_distance = distance[i];
					ID_longest_pop = i;
				}

			}
			for (int k = 0; k < dim; k++)
				centre[j + k * num_cluster] = pop[ID_longest_pop + +k * size_pop];
			delete[] distance;
			num_mem_cluster[label_pop[ID_longest_pop]] --;
			label_pop[ID_longest_pop] = j;
			num_mem_cluster[j] = 1;
		}
		//	printf("error");
	}

	for (int j = 0; j < num_cluster; j++)
	{
		for (int k = 0; k < dim; k++)
		{
			double tmp_centre = 0;
			for (int i = 0; i < size_pop; i++)
			{
				if (label_pop[i] == j)
					tmp_centre += pop[i + k * size_pop];
			}
			centre[j + k * num_cluster] = tmp_centre / (num_mem_cluster[j] + 0.0);
		}
	}
}



void host_kmeans(double * pop, double * centre, int * num_mem_cluster, int * mem_cluster, int * label_pop, int * num_mem_accum_cluster, double * probi_mem_cluster, \
	int num_cluster, int size_pop, int dim, int max_ilteration_kmeans)
{
	int* index_tmp = new int[num_cluster];
	int label_change[1] = { -1 };
#ifdef DEBUG
	FILE * file_record;
#endif
	//	host_initiCentre(centre, label_pop, pop, num_cluster, size_pop, dim);
	int i = 0;
	while (i < max_ilteration_kmeans)
	{
		host_CalDistAndRebelPop(label_pop, label_change, centre, pop, num_cluster, size_pop, dim);
#ifdef DEBUG
#endif
		host_FindCentre(centre, label_pop, pop, num_mem_cluster, num_cluster, size_pop, dim);
#ifdef DEBUG
#endif
		i++;
	}
	//claculate some other parameters for BSO
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
}
#endif


KMeans::KMeans(int num_cluster, int dim, int size_pop, int ID_device)
{
	num_cluster_ = num_cluster;
	dim_ = dim;
	size_pop_ = size_pop;

	d_centre_ = NULL;
	d_ID_best_cluster_ = NULL;
	d_probi_mem_cluster_ = NULL;

#ifdef GPU_KMEANS
	d_pop_ = NULL;
	d_mem_cluster_ = NULL; 
	d_num_mem_cluster_= NULL;
	d_num_accum_mem_cluster_ = NULL;
	d_label_changed_ = NULL;
	d_label_pop_ = NULL;
	d_distance_pop_cluster_ = NULL;
	HANDLE_CUDA_ERROR(cudaMalloc(&d_label_pop_, sizeof(int) * size_pop_));
	HANDLE_CUDA_ERROR(cudaMalloc(&d_distance_pop_cluster_, sizeof(double) * size_pop_ * num_cluster_));
	kernel_configuration_ = new KernelConfiguration(size_pop, dim, ID_device);
	HANDLE_CUDA_ERROR(cudaStreamCreate(&stream));
#endif
#ifdef CPU_KMEANS
	h_pop_ = NULL;
	h_fval_ = NULL;
	h_centre_ = NULL;
	h_num_mem_cluster_ = NULL;
	h_mem_cluster_ = NULL;
	h_label_pop_ = NULL;
	h_num_accum_mem_cluster_ = NULL;
	h_label_changed_ = NULL;
	h_probi_mem_cluster_ = NULL;
	h_ID_best_cluster_ = NULL;

	h_pop_ = new double[size_pop_ * dim_];
	h_fval_ = new double[size_pop_];
	h_centre_ = new double[num_cluster_ * dim_];
	h_num_mem_cluster_ = new int[num_cluster_];
	h_mem_cluster_ = new int[size_pop_];
	h_label_pop_ = new int[size_pop_];
	h_num_accum_mem_cluster_ = new int[num_cluster_];
	h_probi_mem_cluster_ = new double[num_cluster_];
	h_ID_best_cluster_ = new int[num_cluster_];
	h_label_changed_ = new int;
#endif
}
KMeans::~KMeans()
{
#ifdef GPU_KMEANS
	HANDLE_CUDA_ERROR(cudaFree(d_distance_pop_cluster_));
	HANDLE_CUDA_ERROR(cudaFree(d_label_pop_));
	kernel_configuration_->~KernelConfiguration();
#endif
#ifdef CPU_KMEANS
	delete[] h_centre_;
	delete[] h_num_mem_cluster_;
	delete[] h_mem_cluster_;
	delete[] h_label_pop_;
	delete[] h_num_accum_mem_cluster_;
	delete[] h_label_changed_;
	delete[] h_probi_mem_cluster_;
	delete[] h_ID_best_cluster_;
#endif
}

double * KMeans::d_centre()
{
	return d_centre_;
}
int * KMeans::d_ID_best_cluster()
{
	return d_ID_best_cluster_;
}
int * KMeans::d_num_mem_cluster()
{
	return d_num_mem_cluster_;
}

int * KMeans::d_mem_cluster()
{
	return d_mem_cluster_;
}

int	* KMeans::d_num_mem_accu_cluster()
{
	return d_num_accum_mem_cluster_;
}
double * KMeans::d_probi_mem_cluster()
{
	return d_probi_mem_cluster_;
}
void KMeans::set_d_mem_variable(double * d_pop, double * d_fval, double * d_centre, int * d_ID_best_cluster, double * d_probi_mem_cluster, \
	int * d_mem_cluster, int * d_num_mem_cluster, int * d_num_accum_mem_cluster)
{
	d_centre_ = d_centre;
	d_ID_best_cluster_ = d_ID_best_cluster;
	d_probi_mem_cluster_ = d_probi_mem_cluster;
	d_num_mem_cluster_ = d_num_mem_cluster;
	d_mem_cluster_ = d_mem_cluster;
	d_num_accum_mem_cluster_ = d_num_accum_mem_cluster;

#ifdef GPU_KMEANS
	d_pop_ = d_pop;
	d_fval_ = d_fval;
//	d_num_mem_cluster_ = d_num_mem_cluster;
//	d_mem_cluster_ = d_mem_cluster;
//	d_label_pop_ = d_label_pop;

//	d_num_accum_mem_cluster_ = d_num_mem_accu_cluster;
#endif
#ifdef CPU_KMEANS
	cudaMemcpy(h_pop_, d_pop, sizeof(double) * size_pop_ * dim_, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_fval_, d_fval, sizeof(double) * size_pop_, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_centre_, d_centre_, sizeof(double) * dim_ * num_cluster_, cudaMemcpyDeviceToHost);
//	cudaMemcpy(h_ID_best_cluster_, d_ID_best_cluster_, sizeof(int) * num_cluster_, cudaMemcpyDeviceToHost);
//	cudaMemcpy(h_probi_mem_cluster_, d_probi_mem_cluster_, sizeof(double) * num_cluster_, cudaMemcpyDeviceToHost);
#endif
}
#ifdef HOST_RAND
void KMeans::initi_d_centre(dim3 blocks, dim3 threads, double * d_centre, double * d_rand_sequence_unif, double maxbound, double minbound)
{

	API_global_initiCentre(blocks, threads, d_centre, d_rand_sequence_unif, num_cluster_, dim_, minbound, maxbound);
	d_centre_ = d_centre;
#ifdef CPU_KMEANS
	cudaMemcpy(h_centre_, d_centre_, sizeof(double) * num_cluster_ * dim_, cudaMemcpyDeviceToHost);
#endif
}
#endif
#ifdef DEVICE_RAND
void KMeans::initi_d_centre(dim3 blocks, dim3 threads, double * d_centre, curandState *	d_rand_states, double maxbound, double minbound)
{
	API_global_initiCentre(blocks, threads, d_centre, d_rand_states, num_cluster_, dim_, minbound, maxbound);
	d_centre_ = d_centre;
}
#endif


void KMeans::run_kmeans(double * d_pop, double * d_fval, double * d_centre, int * d_ID_best_cluster, double * d_probi_mem_cluster, int * d_mem_cluster,\
	int * d_num_mem_cluster, int * d_num_accum_mem_cluster)
{
	set_d_mem_variable(d_pop, d_fval, d_centre, d_ID_best_cluster, d_probi_mem_cluster, d_mem_cluster, d_num_mem_cluster, d_num_accum_mem_cluster);

#ifdef GPU_KMEANS
	kernel_configuration_->CalKernelConfiguration();

	//double * h_pop = new double[size_pop_ * dim_];
	//cudaMemcpy(h_pop, d_pop, size_pop_ * dim_ * sizeof(double), cudaMemcpyDeviceToHost);
	//FILE * file_record;
	//file_record = fopen("population.txt", "w");

	//for (int i = 0; i < size_pop_; i++)
	//{
	//	for (int j = 0; j < dim_; j++)
	//		fprintf(file_record, "%.10f\t", h_pop[i + j * size_pop_]);
	//	fprintf(file_record, "\n");
	//}
	//fclose(file_record);

	API_global_Clustering(kernel_configuration_->blocks_, kernel_configuration_->threads_, d_centre_, d_label_pop_, d_num_accum_mem_cluster_, d_probi_mem_cluster_, d_mem_cluster_, d_num_mem_cluster_, \
		d_pop_, d_fval_, num_cluster_, size_pop_, dim_, d_distance_pop_cluster_, d_ID_best_cluster_, stream);
//	set_centre();
#endif
#ifdef CPU_KMEANS
	host_kmeans(h_pop_, h_centre_, h_num_mem_cluster_, h_mem_cluster_, h_label_pop_, h_num_accum_mem_cluster_, h_probi_mem_cluster_, num_cluster_, size_pop_, dim_, max_ilteration_kmeans_);
	set_centre();

	cudaMemcpy(d_centre_, h_centre_, sizeof(double) * dim_ * num_cluster_, cudaMemcpyHostToDevice);
	cudaMemcpy(d_ID_best_cluster_, h_ID_best_cluster_, sizeof(int) * num_cluster_, cudaMemcpyHostToDevice);
	cudaMemcpy(d_probi_mem_cluster_, h_probi_mem_cluster_, sizeof(double) * num_cluster_, cudaMemcpyHostToDevice);
	cudaMemcpy(d_mem_cluster_, h_mem_cluster_, sizeof(int) * size_pop_, cudaMemcpyHostToDevice);
	cudaMemcpy(d_num_mem_cluster_, h_num_mem_cluster_, sizeof(int) * num_cluster_, cudaMemcpyHostToDevice);
	cudaMemcpy(d_num_accum_mem_cluster_, h_num_accum_mem_cluster_, sizeof(int) * num_cluster_, cudaMemcpyHostToDevice);


#endif
}

