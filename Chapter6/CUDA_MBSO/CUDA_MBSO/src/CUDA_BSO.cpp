#include "../include/CUDA_BSO.h"

CUDA_BSO::CUDA_BSO(int ID_device, natural ID_func, natural run, natural size_pop, natural dim, int seed,\
	int num_cluster, double	p_replace_centre_rand, double p_select_cluster_one_or_two, \
	double p_use_centre_one_cluster, double p_use_centre_two_cluster, double pr)
{
	size_pop_ = size_pop;
	dim_ = dim;
	current_feval_ = 0;
	population_ = new Population(size_pop, dim);
	population2_ = new Population(size_pop, dim);
	population_candidate_ = new Population(size_pop, dim);
	kernel_configuration_ = new KernelConfiguration(size_pop, dim, ID_device);
	CEC2014_ = new CEC2014(ID_func, size_pop, dim, ID_device);
	cluster_ = new Cluster(num_cluster, dim_, size_pop, ID_device);
	cluster_->InitiCluster(num_cluster);

	p_replace_centre_rand_ = p_replace_centre_rand;
	p_select_cluster_one_or_two_ = p_select_cluster_one_or_two;
	p_use_centre_one_cluster_ = p_use_centre_one_cluster;
	p_use_centre_two_cluster_ = p_use_centre_two_cluster;
	pr_ = pr;
#ifdef HOST_RAND
	natural length_rand_sequence_unif = 2 * (10 + dim_) * CEC2014_->max_feval() + dim_ * size_pop_;
	random_ = new Random(length_rand_sequence_unif, size_pop_, dim_, run, ID_device);
#endif
#ifdef DEVICE_RAND
	random_ = new Random(size_pop_, dim_, run, ID_device);
#endif
#ifndef IMPORT_RAND
	random_->set_seed(seed);
#endif


}

CUDA_BSO::~CUDA_BSO()
{
	population_->~Population();
	population2_->~Population();
	population_candidate_->~Population();
	random_->~Random();
	kernel_configuration_->~KernelConfiguration();
	CEC2014_->~CEC2014();
	cluster_->~Cluster();
	delete population_;
	delete population2_;
	delete population_candidate_;
	delete random_;
	delete kernel_configuration_;
	delete CEC2014_;
	delete cluster_;
}

error CUDA_BSO::BSO()
{
	natural flag_stream = 0;

	double time_start = 0, time_end = 0;

	int start_rand = 0;

	CEC2014_->LoadData();


	kernel_configuration_->CalKernelConfiguration();
	time_start = GetTime();
#ifdef DEVICE_RAND
	random_->initRandom();
#endif
#ifdef HOST_RAND
	Generate_rand_sequence();
#endif
//--------------------------------------------------------------------------//
#ifdef DEBUG																//
	random_->Check(1);														//
#endif																		//
//--------------------------------------------------------------------------//
	InitiPop();
#ifdef HOST_RAND
	start_rand = dim_ * size_pop_;
#endif
//--------------------------------------------------------------------------//
#ifdef DEBUG																//
	population_->Check();													//
#endif																		//
//--------------------------------------------------------------------------//
	EvaluateFitness(population_->d_fval(), population_->d_pop());
//--------------------------------------------------------------------------//
#ifdef DEBUG																//
	population_->Check();													//
#endif																		//
//--------------------------------------------------------------------------//
	current_feval_ += size_pop_;
#ifdef HOST_RAND
	LoopAllocation();
#endif
//--------------------------------------------------------------------------//
#ifdef DEBUG																//
	random_->Check(2);														//
#endif																		//
//--------------------------------------------------------------------------//
	int current_iteration = 0;										//set current generation times as 0
	int max_iteration = CEC2014_->max_feval() / size_pop_ - 1;		//calculate the max iteration of BSO main loop
#ifdef HOST_RAND
	cluster_->initi_centre(kernel_configuration_->blocks_, kernel_configuration_->threads_, random_->d_rand_sequence_unif() + start_rand, CEC2014_->maxbound(), CEC2014_->minbound());		//initialize the centre for clustering
	start_rand += cluster_->num_cluster() * dim_;
#endif
#ifdef DEVICE_RAND
	cluster_->initi_centre(kernel_configuration_->blocks_, kernel_configuration_->threads_, random_->d_rand_states(), CEC2014_->maxbound(), CEC2014_->minbound());		//initialize the centre for clustering
#endif
#ifdef DEBUG																//
	cluster_->Check(1); 
#endif
	while (current_feval_ + size_pop_ <= CEC2014_->max_feval())
	{
//		printf("%d\n", current_feval_);

#ifdef HOST_RAND
		cluster_->initi_centre(kernel_configuration_->blocks_, kernel_configuration_->threads_, random_->d_rand_sequence_unif() + start_rand, CEC2014_->maxbound(), CEC2014_->minbound());		//initialize the centre for clustering
		start_rand += cluster_->num_cluster() * dim_;
#endif
#ifdef DEVICE_RAND
		cluster_->initi_centre(kernel_configuration_->blocks_, kernel_configuration_->threads_, random_->d_rand_states(), CEC2014_->maxbound(), CEC2014_->minbound());		//initialize the centre for clustering
#endif
		ClusterPop(flag_stream);

#ifdef DEBUG
		cluster_->Check(1);
		cluster_->Check(2);
		cluster_->Check(4);
		cluster_->Check(5);
		cluster_->Check(6);
#endif

#ifdef HOST_RAND
		Applystrategy(flag_stream, start_rand);
		start_rand += (10 + dim_) * size_pop_ + 2 + dim_;
#endif

#ifdef DEVICE_RAND
		Applystrategy(flag_stream);
#endif

		EvaluateFitness(population_candidate_->d_fval(), population_candidate_->d_pop());
//--------------------------------------------------------------------------//
#ifdef DEBUG																//
		population_candidate_->Check();										//
#endif																		//
//--------------------------------------------------------------------------//
		GenerateNewPop(&flag_stream);
//--------------------------------------------------------------------------//
#ifdef DEBUG																//
		if (flag_stream == 0)												//
			population_->Check();											//
		else																//
			population2_->Check();											//
#endif																		//
//--------------------------------------------------------------------------//
#ifdef HOST_RAND		
		LoopAllocation();
#endif
		current_feval_ += size_pop_;
		current_iteration++;

	}
	time_end = GetTime();
	DisplayResults(flag_stream, time_end - time_start);
	RecordResults(flag_stream, time_end - time_start);
	cudaDeviceReset();
	return SUCCESS;
}

error CUDA_BSO::InitiPop()
{
#ifdef HOST_RAND
	population_->InitiPop(kernel_configuration_->blocks_, kernel_configuration_->threads_, CEC2014_->minbound(), CEC2014_->maxbound(), random_->d_rand_sequence_unif());
#endif
	
#ifdef DEVICE_RAND
	population_->InitiPop(kernel_configuration_->blocks_, kernel_configuration_->threads_, CEC2014_->minbound(), CEC2014_->maxbound(), random_->d_rand_states());
#endif
return SUCCESS;
}

error CUDA_BSO::Applystrategy(natural flag_stream
#ifdef HOST_RAND
	, int start_rand)
#endif
#ifdef DEVICE_RAND
	)
#endif
{
	if (flag_stream == 1)
#ifdef HOST_RAND
		API_applystrategy(kernel_configuration_->blocks_, kernel_configuration_->threads_, population_candidate_->d_pop(), population2_->d_pop(), cluster_->d_ID_best_cluster(), cluster_->d_centre(), cluster_->d_num_mem_cluster(), cluster_->d_num_accum_mem_cluster(), \
		cluster_->d_probi_mem_cluster(), cluster_->d_mem_cluster(), cluster_->num_cluster(), random_->d_rand_sequence_unif() + start_rand, size_pop_, dim_, CEC2014_->maxbound(), CEC2014_->minbound(), \
		p_replace_centre_rand_, p_select_cluster_one_or_two_, p_use_centre_one_cluster_, p_use_centre_two_cluster_, pr_);
	else
		API_applystrategy(kernel_configuration_->blocks_, kernel_configuration_->threads_, population_candidate_->d_pop(), population_->d_pop(), cluster_->d_ID_best_cluster(), cluster_->d_centre(), cluster_->d_num_mem_cluster(), cluster_->d_num_accum_mem_cluster(), \
		cluster_->d_probi_mem_cluster(), cluster_->d_mem_cluster(), cluster_->num_cluster(), random_->d_rand_sequence_unif() + start_rand, size_pop_, dim_, CEC2014_->maxbound(), CEC2014_->minbound(), \
		p_replace_centre_rand_, p_select_cluster_one_or_two_, p_use_centre_one_cluster_, p_use_centre_two_cluster_, pr_);
#endif
#ifdef DEVICE_RAND
	API_applystrategy(kernel_configuration_->blocks_, kernel_configuration_->threads_, population_candidate_->d_pop(), population2_->d_pop(), cluster_->d_ID_best_cluster(), cluster_->d_centre(), cluster_->d_num_mem_cluster(), cluster_->d_num_accum_mem_cluster(), \
		cluster_->d_probi_mem_cluster(), cluster_->d_mem_cluster(), cluster_->num_cluster(), random_->d_rand_states(), size_pop_, dim_, CEC2014_->maxbound(), CEC2014_->minbound(), \
		p_replace_centre_rand_, p_select_cluster_one_or_two_, p_use_centre_one_cluster_, p_use_centre_two_cluster_, pr_);
	else
		API_applystrategy(kernel_configuration_->blocks_, kernel_configuration_->threads_, population_candidate_->d_pop(), population_->d_pop(), cluster_->d_ID_best_cluster(), cluster_->d_centre(), cluster_->d_num_mem_cluster(), cluster_->d_num_accum_mem_cluster(), \
		cluster_->d_probi_mem_cluster(), cluster_->d_mem_cluster(), cluster_->num_cluster(), random_->d_rand_states(), size_pop_, dim_, CEC2014_->maxbound(), CEC2014_->minbound(), \
		p_replace_centre_rand_, p_select_cluster_one_or_two_, p_use_centre_one_cluster_, p_use_centre_two_cluster_, pr_);
#endif

	return SUCCESS;
}
#ifdef HOST_RAND
error CUDA_BSO::LoopAllocation()
{
	random_->LoopAllocation();
	return SUCCESS;
}
#endif
error CUDA_BSO::EvaluateFitness(double * d_fval, double * d_pop)
{
	CEC2014_->EvaluateFitness(d_fval, d_pop);
	return SUCCESS;
}

#ifdef IMPORT_RAND
void CUDA_BSO::RandFileToHost(char * name_file_unif, char * name_file_norm)
{
	random_->RandFileToHost(name_file_unif, name_file_norm);
}
#endif

error CUDA_BSO::GenerateNewPop(natural *flag_stream)
{
	if (*flag_stream == 1)
	{
		API_generateNewPop(kernel_configuration_->blocks_, kernel_configuration_->threads_,
								population_->d_pop(),
								population_->d_fval(),
								population_candidate_->d_pop(),
								population2_->d_pop(),
								population_candidate_->d_fval(),
								population2_->d_fval(),
								size_pop_);
		*flag_stream = 0;
	}
	else
	{	
		API_generateNewPop(kernel_configuration_->blocks_, kernel_configuration_->threads_,
								population2_->d_pop(),
								population2_->d_fval(),
								population_candidate_->d_pop(),
								population_->d_pop(),
								population_candidate_->d_fval(),
								population_->d_fval(),
								size_pop_);
		*flag_stream = 1;
	}
	return SUCCESS;
}

double CUDA_BSO::GetTime()
{
	timeval tim;
	gettimeofday(&tim);
	return tim.tv_sec + (tim.tv_usec / 1000000.0);
}
void CUDA_BSO::ClusterPop(natural flag_stream)
{
	if (flag_stream == 0)
		cluster_->ClusterPop(population_->d_pop(), population_->d_fval());
	else
		cluster_->ClusterPop(population2_->d_pop(), population2_->d_fval());
}

void CUDA_BSO::RecordResults(natural flag_stream, double duration_computation)
{
	char name_file[100];
	FILE * file_record;
	sprintf(name_file, "dim_%d_pop_%d.out", dim_, size_pop_);
	file_record = fopen(name_file, "a");

	if (flag_stream == 0)												
	{
		population_->FindIndivualBest();
		fprintf(file_record, "%d\t%d\t%12.30f\t%f\n", CEC2014_->ID_func(), random_->run(), population_->h_fval_best(), duration_computation);
		fclose(file_record);
	}
	else				
	{
		population2_->FindIndivualBest();
		fprintf(file_record, "%d\t%d\t%12.30f\t%f\n", CEC2014_->ID_func(), random_->run(), population2_->h_fval_best(), duration_computation);
		fclose(file_record);
	}

}

void CUDA_BSO::DisplayResults(natural flag_stream, double duration_computation)
{
	if (flag_stream == 0)
	{
		population_->FindIndivualBest();
		printf("Best Value %12.30f, computing time is %f\n", population_->h_fval_best(), duration_computation);
	}
	else
	{
		population2_->FindIndivualBest();
		printf("Best Value %12.30f, computing time is %f\n", population2_->h_fval_best(), duration_computation);
	}
}

#ifdef HOST_RAND
error CUDA_BSO::Generate_rand_sequence()
{
	random_->Generate_rand_sequence();
	return SUCCESS;
}
#endif