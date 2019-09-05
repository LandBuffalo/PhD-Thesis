#include "../include/CPU_BSO.h"
double * test = NULL;
CPU_BSO::CPU_BSO(natural ID_func, natural run, natural size_pop, natural dim, int seed,\
	int num_cluster, double	p_replace_centre_rand, double p_select_cluster_one_or_two, \
	double p_use_centre_one_cluster, double p_use_centre_two_cluster, double pr)
{
	size_pop_ = size_pop;
	dim_ = dim;
	current_feval_ = 0;

	max_iteration_ = 50;
	population_candidate_ = new Population(size_pop, dim);

	population_ = new Population(size_pop, dim);

	CEC2014_ = new CEC2014(ID_func, size_pop, dim);

	cluster_ = new Cluster(num_cluster, dim_, size_pop);

	cluster_->InitiCluster(num_cluster);

#ifdef HOSTCURAND
	natural length_rand_sequence_unif = 2 * (10 + dim_) * CEC2014_->max_feval() + dim_ * size_pop_; //14342;//
	natural length_rand_sequence_norm = dim_ * CEC2014_->max_feval() + dim_ * size_pop_; //8000;//

	random_ = new Random(length_rand_sequence_unif, length_rand_sequence_norm, size_pop_, dim_, run);
#ifndef IMPORT_RAND
	random_->set_seed(seed);
#endif
#endif
	p_replace_centre_rand_ = p_replace_centre_rand;
	p_select_cluster_one_or_two_ = p_select_cluster_one_or_two;
	p_use_centre_one_cluster_ = p_use_centre_one_cluster;
	p_use_centre_two_cluster_ = p_use_centre_two_cluster;
	pr_ = pr;
}

CPU_BSO::~CPU_BSO()
{
	population_->~Population();
	population_candidate_->~Population();
	random_->~Random();
//	CEC2014_->~CEC2014();
	cluster_->~Cluster();
//	delete [] population_;
//	delete [] population_candidate_;
//	delete [] random_;
//	delete [] CEC2014_;
//	delete [] cluster_;
}

error CPU_BSO::BSO()
{
	double time_start = 0, time_end = 0;
	int start_rand = 0;
	//start timer
	time_start = GetTime();											//start timing

	Generate_rand_sequence();										//generate total random number sequence
	InitiPop();														//initilize the population based on random number sequence
	start_rand = dim_ * size_pop_;
	EvaluateFitness(population_->fval(), population_->pop());		//evaluate the initialized popuilation
	current_feval_ += size_pop_;									//update the currenc fitness evaluation times
#ifdef DEBUG	
	population_->Check();			
#endif
	LoopAllocationRandSequence();									//allocate the random number sequence to specific stragety, in BSO, it does nothing


//	time_start = GetTime();
	cluster_->initi_centre(random_->h_rand_sequence_unif() + start_rand, CEC2014_->maxbound(), CEC2014_->minbound());		//initialize the centre for clustering
	start_rand += cluster_->num_cluster() * dim_;
#ifdef DEBUG	
	cluster_->Check(1);
#endif	

	while (current_feval_ + size_pop_ <= CEC2014_->max_feval())
	{									
//		printf("%d\n", current_feval_);
		cluster_->ClusterPop(population_->pop(), population_->fval(), CEC2014_->minbound(), CEC2014_->maxbound(), random_->h_rand_sequence_unif() + start_rand);					//cluster the population
		start_rand += cluster_->num_cluster() * dim_;
#ifdef DEBUG
		cluster_->Check(1);
		cluster_->Check(2);
		cluster_->Check(3);
		cluster_->Check(4);
		cluster_->Check(5);
		cluster_->Check(6);
#endif
		Applystrategy(start_rand);	
		start_rand += (10 + dim_) * size_pop_ + 2 + dim_;
		//		cluster_->Check(1);

		EvaluateFitness(population_candidate_->fval(), population_candidate_->pop());	//fitness function evaluates the population
#ifdef DEBUG
		population_candidate_->Check();
#endif
		GenerateNewPop();																//generate the new population based on original population and candidate population				

#ifdef DEBUG

#endif
		LoopAllocationRandSequence();													//allocate the random number sequence to specific stragety, in BSO, it does nothing
		current_feval_ += size_pop_;													//update the current feval and iteration
	}
	time_end = GetTime();
	DisplayResults(time_end - time_start);
	RecordResults(time_end - time_start);
	return SUCCESS;
}

error CPU_BSO::InitiPop()
{
	population_->InitiPop(CEC2014_->minbound(), CEC2014_->maxbound(), random_->h_rand_sequence_unif());
	return SUCCESS;
}



void CPU_BSO::SelectOneCluster(double * copy_rand, int starter_rand, int num_cluster, int index_pop, double *probi_mem_cluster, int * num_accum_mem_cluster, \
	int * num_mem_cluster, int * mem_cluster, double * pop_candidate, double * pop, double * centre)
{
	//generate random number when needed
	double tmp_rand = copy_rand[starter_rand + 1];
	int ID_cluster = 0;
	//find the ID of cluster based on probi_mem_cluster
	for (int k = 0; k < num_cluster; k++)
		if (tmp_rand < probi_mem_cluster[k])
		{
			ID_cluster = k;
			break;
		}
	//generate random number when needed
	tmp_rand = copy_rand[starter_rand + 2];
	//whether choose the centre or the individual to generate candidate population
	if (tmp_rand < p_use_centre_one_cluster_)
		for (int j = 0; j < dim_; j++)
			pop_candidate[index_pop + j * size_pop_] = centre[ID_cluster + j * num_cluster];
	else
	{
		//ID of individual selected to generate the candidate individual
		while (num_mem_cluster[ID_cluster] == 0)
		{
			ID_cluster = ID_cluster + 1;
			if (ID_cluster >= num_cluster)
				ID_cluster = 0;
		}
		int ID_pop = num_accum_mem_cluster[ID_cluster] + floor(copy_rand[starter_rand + 3] * num_mem_cluster[ID_cluster]);

		for (int j = 0; j < dim_; j++)
			pop_candidate[index_pop + j * size_pop_] = pop[mem_cluster[ID_pop] + j * size_pop_];
	}
}

void CPU_BSO::SelectTwoCluster(double * copy_rand, int starter_rand, int num_cluster, int index_pop, int * num_accum_mem_cluster, \
	int * num_mem_cluster, int * mem_cluster, double * pop_candidate, double * pop, double * centre)
{
	double tmp_rand = 0;

	int ID_cluster_1 = floor(copy_rand[starter_rand + 1] * num_cluster);
	while (num_mem_cluster[ID_cluster_1] == 0)
	{
		ID_cluster_1 = ID_cluster_1 + 1;
		if (ID_cluster_1 >= num_cluster)
			ID_cluster_1 = 0;
	}
	int ID_cluster_2 = floor(copy_rand[starter_rand + 2] * num_cluster);
	while (num_mem_cluster[ID_cluster_2] == 0)
	{
		ID_cluster_2 = ID_cluster_2 + 1;
		if (ID_cluster_2 >= num_cluster)
			ID_cluster_2 = 0;
	}
	int ID_pop_1 = num_accum_mem_cluster[ID_cluster_1] + floor(copy_rand[starter_rand + 3] * num_mem_cluster[ID_cluster_1]);
	int ID_pop_2 = num_accum_mem_cluster[ID_cluster_2] + floor(copy_rand[starter_rand + 4] * num_mem_cluster[ID_cluster_2]);

	//generate random number when needed
	tmp_rand = copy_rand[starter_rand + 5];
	if (copy_rand[starter_rand + 6] < p_use_centre_two_cluster_)
		for (int j = 0; j < dim_; j++)
			pop_candidate[index_pop + j * size_pop_] = tmp_rand * centre[ID_cluster_1 + j * num_cluster] + (1 - tmp_rand) * centre[ID_cluster_2 + j * num_cluster];
	else
		for (int j = 0; j < dim_; j++)
			pop_candidate[index_pop + j * size_pop_] = tmp_rand *pop[mem_cluster[ID_pop_1] + j * size_pop_] + (1 - tmp_rand) * pop[mem_cluster[ID_pop_2] + j * size_pop_];
}




error CPU_BSO::Applystrategy(int start_rand)
{
	double tmp_rand = 0;
	int ID_cluster = 0;
	//pointer to the random number sequence
	double * copy_rand = random_->h_rand_sequence_unif();

	//pointer to member variables of clustering
	double * centre = cluster_->centre();
	int num_cluster = cluster_->num_cluster();
	double * probi_mem_cluster = cluster_->probi_mem_cluster();
	int * num_accum_mem_cluster = cluster_->num_accum_mem_cluster();
	int * num_mem_cluster = cluster_->num_mem_cluster();
	int * mem_cluster = cluster_->mem_cluster();

	//pointer to population and candidate population
	double * pop_candidate = population_candidate_->pop();
	double * pop = population_->pop();

	//generate random number when needed
	tmp_rand = copy_rand[start_rand];
	//whether replace one centre randomly
	if (tmp_rand < p_replace_centre_rand_)
	{
		//randomly choose the ID of cluster to replace the centre
		ID_cluster = floor(copy_rand[start_rand + 1] * num_cluster);
		//replace the centre of selected cluster andomly
		for (int j = 0; j < dim_; j++)
			centre[ID_cluster + j * num_cluster] = CEC2014_->minbound() + (CEC2014_->maxbound() - CEC2014_->minbound()) * copy_rand[start_rand + 2 + j];
	}
	//update pointer to random number sequence
	start_rand += 2 + dim_;
	for (int i = 0; i < size_pop_; i++)
	{
		//flag signs whether this individual is the best one in its clusters(the centre os clusters), if yes, do nothing on this individual and enter candidate population
		//else, do BSO stragety to generate candidate population
		int flag_best_in_cluster = 0;
		for (int k = 0; k < num_cluster; k++)
		{
			if (i == *(cluster_->ID_best_cluster() + k))
			{
				flag_best_in_cluster = 1;
				break;
			}
		}
		// if individual is not the best one in its cluster
		if (flag_best_in_cluster == 0)
		{
			//generate random number when needed
			tmp_rand = copy_rand[start_rand];
			//whether choose one or two centre to generate candidate population, if decide use one centre
			if (tmp_rand < p_select_cluster_one_or_two_)
			{
				SelectOneCluster(copy_rand, start_rand, num_cluster, i, probi_mem_cluster, num_accum_mem_cluster, \
					num_mem_cluster, mem_cluster, pop_candidate, pop, centre);
			}
			else 		// if decide to use two centres
			{
				//SelectOneCluster(copy_rand, starter_rand, num_cluster, i, probi_mem_cluster, num_accum_mem_cluster, \
					num_mem_cluster, mem_cluster, pop_candidate, pop, centre);
				SelectTwoCluster(copy_rand, start_rand, num_cluster, i, num_accum_mem_cluster, \
					num_mem_cluster, mem_cluster, pop_candidate, pop, centre);
			}
			//weight the candidate population by step size values
			tmp_rand = copy_rand[start_rand + 7];
			if (tmp_rand < pr_)
			{
				for (int j = 0; j < dim_; j++)
					pop_candidate[i + j * size_pop_] = CEC2014_->minbound() + (CEC2014_->maxbound() - CEC2014_->minbound())*copy_rand[start_rand + 10 + j];
			}
			else
			{
				int ID_individual1 = floor(copy_rand[start_rand + 8] * size_pop_);
				int ID_individual2 = floor(copy_rand[start_rand + 9] * size_pop_);
				for (int j = 0; j < dim_; j++)
					pop_candidate[i + j * size_pop_] = pop_candidate[i + j * size_pop_] + copy_rand[start_rand + 10 + j] * (pop[ID_individual1 + j * size_pop_] - pop[ID_individual2 + j * size_pop_]);
			}
			//over bound processing
			for (int j = 0; j < dim_; j++) //----boundary constraints via random bouncing back -------
			{
				while ((pop_candidate[i + j * size_pop_] < CEC2014_->minbound()) || (pop_candidate[i + j * size_pop_] > CEC2014_->maxbound()))
				{
					if (pop_candidate[i + j * size_pop_] < CEC2014_->minbound())
						pop_candidate[i + j * size_pop_] = CEC2014_->minbound() + (CEC2014_->minbound() - pop_candidate[i + j * size_pop_]);
					if (pop_candidate[i + j * size_pop_] > CEC2014_->maxbound())
						pop_candidate[i + j * size_pop_] = CEC2014_->maxbound() - (pop_candidate[i + j * size_pop_] - CEC2014_->maxbound());
				}
			}

		}
		else // if the individual is the best one(centre) of its cluster, then it is the candidiate individual directly
		{
			for (int j = 0; j < dim_; j++)
				pop_candidate[i + j * size_pop_] = pop[i + j * size_pop_];
		}
		//update the pointer to random number sequence
		start_rand += 10 + dim_;
	}
	return SUCCESS;
}

error CPU_BSO::LoopAllocationRandSequence()
{
	random_->LoopAllocationRandSequence();
	return SUCCESS;
}

error CPU_BSO::EvaluateFitness(double * fval, double * pop)
{

	CEC2014_->EvaluateFitness(fval, pop);
	return SUCCESS;
}

#ifdef IMPORT_RAND
void CPU_BSO::RandFileToHost(char * name_file_unif, char * name_file_norm)
{
	random_->RandFileToHost(name_file_unif, name_file_norm);
}
#endif

error CPU_BSO::GenerateNewPop()
{
	for (int i = 0; i < size_pop_; i++)
	{
		if (*(population_->fval() + i) > *(population_candidate_->fval() + i))
		{
			*(population_->fval() + i) = *(population_candidate_->fval() + i);
			for (int j = 0; j < dim_; j++)
				*(population_->pop() + i + j * size_pop_) = *(population_candidate_->pop() + i + j * size_pop_);
		}
	}
	return SUCCESS;
}

double CPU_BSO::GetTime()
{
	timeval tim;
	gettimeofday(&tim);
	return tim.tv_sec + (tim.tv_usec / 1000000.0);
}


void CPU_BSO::RecordResults(double duration_computation)
{
	char name_file[100];
	FILE * file_record;
	sprintf(name_file, "dim_%d_pop_%d.out", dim_, size_pop_);
	file_record = fopen(name_file, "a");

	population_->FindIndivualBest();
	fprintf(file_record, "%d\t%d\t%12.30f\t%f\n", CEC2014_->ID_func(), random_->run(), population_->fval_best(), duration_computation);
	fclose(file_record);

}

void CPU_BSO::DisplayResults(double duration_computation)
{
	population_->FindIndivualBest();
	printf("Best Value %12.30f, computing time is %f\n", population_->fval_best(), duration_computation);
}

void CPU_BSO::set_step_size_base(int max_iteration, int current_iteration)
{
	double value = 1.0 + exp(-(0.5*max_iteration - current_iteration) / 20.0);
	step_size_base_ = 1.0 / value;
}

#ifdef HOSTCURAND
error CPU_BSO::Generate_rand_sequence()
{
	random_->Generate_rand_sequence();
	return SUCCESS;
}
#endif

#ifdef DEBUG
void CPU_BSO::Check(int level)
{
	switch (level)
	{
	case(1) :

		break;
	default:
		break;
	}
}
#endif