#include "../include/CEC2014.h"

CEC2014::CEC2014(natural ID_func, natural size_pop, natural dim, int ID_device)
{
	ID_func_ = ID_func;
	size_pop_ = size_pop;
	maxbound_ = 100;
	minbound_ = -100;
	dim_ = dim;
	max_feval_base_ = 10000;
	set_max_feval();
	d_M_ = NULL;
	d_shuffle_ = NULL;
	//d_w_ = NULL;
	//d_sumW_ = NULL;
	d_shift_ = NULL;
	flag_composition_ = true;
	kernel_configuration_ = new KernelConfiguration(size_pop, dim, ID_device);
	CalConfigCEC2014();
#ifdef DEBUG
	h_fval_ = NULL;
	h_pop_rotated_ = NULL;
	h_pop_original_ = NULL;
	h_shift_ = NULL;
	h_M_ = NULL;
	h_shuffle_ = NULL;
	h_fval_ = new double[sizeof(double) * size_pop_];
	h_pop_rotated_ = new double[sizeof(double) * size_pop_ * dim_ * num_comp_func_];
	h_pop_original_ = new double[sizeof(double) * size_pop_ * dim_];
	h_shift_ = new double[sizeof(double) * MAX_DIM * MAX_NUM_COMP_FUNC];
	h_M_ = new double[sizeof(double) * dim_ * dim_ * num_comp_func_];
	h_shuffle_ = new int[sizeof(int) * dim_ * MAX_NUM_COMP_FUNC];
#endif
	HANDLE_CUDA_ERROR(cudaMalloc(&d_M_, sizeof(double) * dim_ * dim_ * num_comp_func_));
	HANDLE_CUDA_ERROR(cudaMalloc(&d_shuffle_, sizeof(int) * dim_ * MAX_NUM_COMP_FUNC));
	HANDLE_CUDA_ERROR(cudaMalloc(&d_pop_rotated_, sizeof(double) * size_pop_ * dim_ * num_comp_func_));
	HANDLE_CUDA_ERROR(cudaMalloc(&d_shift_, sizeof(double) * MAX_DIM * MAX_NUM_COMP_FUNC));

}

CEC2014::~CEC2014()
{
	HANDLE_CUDA_ERROR(cudaFree(d_M_));
	HANDLE_CUDA_ERROR(cudaFree(d_shuffle_));
	HANDLE_CUDA_ERROR(cudaFree(d_pop_rotated_));
	HANDLE_CUDA_ERROR(cudaFree(d_M_));
	kernel_configuration_->~KernelConfiguration();
#ifdef DEBUG
	delete[] h_fval_;
	delete[] h_pop_rotated_;
	delete[] h_pop_original_;
	delete[] h_shift_;
	delete[] h_M_;
	delete[] h_shuffle_;
#endif
}

void CEC2014::set_d_pop_original_d_fval(double * d_pop_original, double * d_fval)
{
	d_pop_original_ = d_pop_original;
	d_fval_ = d_fval;
}

void CEC2014::set_max_feval()
{
	max_feval_ = max_feval_base_ * dim_;
}

natural	CEC2014::ID_func()
{
	return ID_func_;
}

double CEC2014::minbound()
{
	return minbound_;
}

double CEC2014::maxbound()
{
	return maxbound_;
}

int CEC2014::max_feval()
{
	return max_feval_;
}

error CEC2014::LoadData()
{
	double *h_shift = new double[sizeof(double) * MAX_DIM * MAX_NUM_COMP_FUNC];
	double *h_M = new double[sizeof(double) * dim_ * dim_ * num_comp_func_];
	int *h_shuffle = new int[sizeof(int) * dim_ * MAX_NUM_COMP_FUNC];
	char fileName[100];
	sprintf(fileName, "input_data/shift_data_%d.txt", ID_func_);
	FILE *file = fopen(fileName, "r");

	for (int i = 0; i < MAX_DIM * MAX_NUM_COMP_FUNC; i++)
	{
		h_shift[i] = 0;
	}
	if (flag_composition_)
	{
		for (int i = 0; i < num_comp_func_; i++)
			for (int j = 0; j < MAX_DIM; j++)
			{
				if (is_shifted_[i])
					fscanf(file, "%lf", &h_shift[i * MAX_DIM + j]);
				else
					h_shift[i] = 0;
			}
	}
	else
	{
		for (int i = 0; i < dim_; i++)
		{
			if (is_shifted_[0])
				fscanf(file, "%lf", &h_shift[i]);
			else
				h_shift[i] = 0;
		}
	}
	fclose(file);

	sprintf(fileName, "input_data/M_%d_D%d.txt", ID_func_, dim_);
	file = fopen(fileName, "r");
	for (int k = 0; k < num_comp_func_; k++)
		for (int i = 0; i < dim_; i++)
			for (int j = 0; j < dim_; j++)
			{
				if (is_rotated_[k])
				{
					fscanf(file, "%lf", &h_M[k * dim_ * dim_ + j * dim_ + i]);
				}
				else
				{
					fscanf(file, "%lf", &h_M[k * dim_ * dim_ + j * dim_ + i]);
					if (i == j)
						h_M[k * dim_ * dim_ + j * dim_ + i] = 1;
					else
						h_M[k * dim_ * dim_ + j * dim_ + i] = 0;
				}
			}
	fclose(file);
	sprintf(fileName, "input_data/shuffle_data_%d_D%d.txt", ID_func_, dim_);
	file = fopen(fileName, "r");

	for (int i = 0; i < dim_ * MAX_NUM_COMP_FUNC; i++)
		fscanf(file, "%d", &h_shuffle[i]);
	fclose(file);

	HANDLE_CUDA_ERROR(cudaMemcpy(d_M_, h_M, num_comp_func_ * dim_ * dim_ * sizeof(double), cudaMemcpyHostToDevice));
	HANDLE_CUDA_ERROR(cudaMemcpy(d_shuffle_, h_shuffle, dim_ * sizeof(int) * MAX_NUM_COMP_FUNC, cudaMemcpyHostToDevice));
	HANDLE_CUDA_ERROR(cudaMemcpy(d_shift_, h_shift, MAX_NUM_COMP_FUNC * MAX_DIM  * sizeof(double), cudaMemcpyHostToDevice));

	delete[] h_M;
	delete[] h_shift;
	delete[] h_shuffle;
	//	HANDLE_CUDA_ERROR(cudaMemcpyFromSymbol(h_shift, d_shift, MAX_FUNC_COMPOSITION * MAX_DIM  * sizeof(real)));
	return SUCCESS;

}

void CEC2014::CalConfigCEC2014()
{
	is_rotated_[0] = is_rotated_[1] = is_rotated_[2] = is_rotated_[3] = is_rotated_[4] = 0;
	is_shifted_[0] = is_shifted_[1] = is_shifted_[2] = is_shifted_[3] = is_shifted_[4] = 0;
	if (ID_func_ < 23)
	{
		num_comp_func_ = 1;
		flag_composition_ = false;
		is_rotated_[0] = 1;
		is_shifted_[0] = 1;
		shift_ = 0;
		rate_weighted_ = 1;
		num_comp_func_ = 1;
		switch (ID_func_)
		{
		case(4) :
			rate_weighted_ = 2.048 / (double)100;
			shift_ = 1;
			break;
		case(6) :
			rate_weighted_ = 0.5 / (double)100;
			break;
		case(7) :
			rate_weighted_ = 6;
			break;
		case(8) :
			is_rotated_[0] = 0;
			rate_weighted_ = 5.12 / (double)100;
			break;
		case(9) :
			rate_weighted_ = 5.12 / (double)100;
			break;
		case(10) :
			is_rotated_[0] = 0;
			rate_weighted_ = 10;
			break;
		case(11) :
			rate_weighted_ = 10;
			break;
		case(12) :
			rate_weighted_ = 5 / (double)100;
			break;
		case(13) :
			rate_weighted_ = 5 / (double)100;
			break;
		case(14) :
			rate_weighted_ = 5 / (double)100;
			break;
		case(15) :
			rate_weighted_ = 5 / (double)100;
			shift_ = 1;
			break;
		case(16) :
			shift_ = 1;
			break;
		default:
			break;
		}
	}
	else
	{
		switch (ID_func_)
		{
		case(23) :
			num_comp_func_ = 5;
			is_shifted_[0] = is_shifted_[1] = is_shifted_[2] = is_shifted_[3] = is_shifted_[4] = 1;
			is_rotated_[0] = is_rotated_[2] = is_rotated_[3] = 1;
			is_rotated_[1] = is_rotated_[4] = 0;
			sigma_[0] = 10;
			sigma_[1] = 20;
			sigma_[2] = 30;
			sigma_[3] = 40;
			sigma_[4] = 50;
			rate_weighted_ = 1;
			shift_ = 0;
			bias_ = 2300;
			break;
		case(24) :
			num_comp_func_ = 3;
			is_shifted_[0] = is_shifted_[1] = is_shifted_[2] = 1;
			is_rotated_[1] = is_rotated_[2] = 1;
			is_rotated_[0] = 0;
			sigma_[0] = 20;
			sigma_[1] = 20;
			sigma_[2] = 20;
			rate_weighted_ = 1;
			shift_ = 0;
			bias_ = 2400;
			break;
		case(25) :
			num_comp_func_ = 3;
			is_shifted_[0] = is_shifted_[1] = is_shifted_[2] = 1;
			is_rotated_[0] = is_rotated_[1] = is_rotated_[2] = 1;
			sigma_[0] = 10;
			sigma_[1] = 30;
			sigma_[2] = 50;
			rate_weighted_ = 1;
			shift_ = 0;
			bias_ = 2500;
			break;
		case(26) :
			num_comp_func_ = 5;
			is_shifted_[0] = is_shifted_[1] = is_shifted_[2] = is_shifted_[3] = is_shifted_[4] = 1;
			is_rotated_[0] = is_rotated_[1] = is_rotated_[2] = is_rotated_[3] = is_rotated_[4] = 1;
			sigma_[0] = 10;
			sigma_[1] = 10;
			sigma_[2] = 10;
			sigma_[3] = 10;
			sigma_[4] = 10;
			rate_weighted_ = 1;
			shift_ = 0;
			bias_ = 2600;
			break;
		case(27) :
			num_comp_func_ = 5;
			is_shifted_[0] = is_shifted_[1] = is_shifted_[2] = is_shifted_[3] = is_shifted_[4] = 1;
			is_rotated_[0] = is_rotated_[1] = is_rotated_[2] = is_rotated_[3] = is_rotated_[4] = 1;
			sigma_[0] = 10;
			sigma_[1] = 10;
			sigma_[2] = 10;
			sigma_[3] = 20;
			sigma_[4] = 20;
			rate_weighted_ = 1;
			shift_ = 0;
			bias_ = 2700;
			break;
		case(28) :
			num_comp_func_ = 5;
			is_shifted_[0] = is_shifted_[1] = is_shifted_[2] = is_shifted_[3] = is_shifted_[4] = 1;
			is_rotated_[0] = is_rotated_[1] = is_rotated_[2] = is_rotated_[3] = is_rotated_[4] = 1;
			sigma_[0] = 10;
			sigma_[1] = 20;
			sigma_[2] = 30;
			sigma_[3] = 40;
			sigma_[4] = 50;
			rate_weighted_ = 1;
			shift_ = 0;
			bias_ = 2800;
			break;
		case(29) :
			num_comp_func_ = 3;
			is_shifted_[0] = is_shifted_[1] = is_shifted_[2] = 1;
			is_rotated_[0] = is_rotated_[1] = is_rotated_[2] = 1;
			sigma_[0] = 10;
			sigma_[1] = 30;
			sigma_[2] = 50;
			rate_weighted_ = 1;
			shift_ = 0;
			bias_ = 2900;
			break;
		case(30) :
			num_comp_func_ = 3;
			is_shifted_[0] = is_shifted_[1] = is_shifted_[2] = 1;
			is_rotated_[0] = is_rotated_[1] = is_rotated_[2] = 1;
			sigma_[0] = 10;
			sigma_[1] = 30;
			sigma_[2] = 50;
			rate_weighted_ = 1;
			shift_ = 0;
			bias_ = 3000;
			break;
		default:
			break;
		}
	}

	kernel_configuration_->CalKernelConfiguration();
}

void CEC2014::ShiftRotate()
{
	dim3 blocks((size_pop_ - 1) / TILE_WIDTH + 1, (dim_ - 1) / TILE_WIDTH + 1, 1);
	dim3 threads(TILE_WIDTH, TILE_WIDTH, 1);
	//	real * tmppop = (real*)malloc(sizeof(real) * dim * popsize * funcNum)
	//HANDLE_CUDA_ERROR(cudaMemcpy(h_pop_original_, d_pop_original_, size_pop_ * dim_ * sizeof(double), cudaMemcpyDeviceToHost));
	//HANDLE_CUDA_ERROR(cudaMemcpy(h_pop_original_, d_pop_original_, size_pop_ * dim_ * sizeof(double), cudaMemcpyDeviceToHost));
	//HANDLE_CUDA_ERROR(cudaMemcpy(h_pop_original_, d_pop_original_, size_pop_ * dim_ * sizeof(double), cudaMemcpyDeviceToHost));

	API_rotation(d_pop_rotated_, d_pop_original_, d_M_, d_shift_, shift_, rate_weighted_, blocks, threads, size_pop_, dim_, num_comp_func_);
}

void CEC2014::EvaluateFitness(double * d_fval, double * d_pop)
{
	set_d_pop_original_d_fval(d_pop, d_fval);
	ShiftRotate();
#ifdef DEBUG
	Check(2);	// check the d_pop_original_ and d_pop_rotated_(rotated correctly)
#endif
	API_evaluateFitness(d_fval, d_pop_original_, d_pop_rotated_, d_shuffle_, d_shift_, kernel_configuration_->blocks_, kernel_configuration_->threads_, ID_func_, size_pop_, dim_);
#ifdef DEBUG
	Check(1);	// check the d_fval_(evaluate correctly)
#endif

}

#ifdef DEBUG
void CEC2014::Check(int debug_level)
{
	switch (debug_level)
	{
	case(1) :
		HANDLE_CUDA_ERROR(cudaMemcpy(h_fval_, d_fval_, size_pop_ * sizeof(double), cudaMemcpyDeviceToHost));
		break;
	case(2) :
		HANDLE_CUDA_ERROR(cudaMemcpy(h_pop_rotated_, d_pop_rotated_, size_pop_ * dim_ * sizeof(double), cudaMemcpyDeviceToHost));
		HANDLE_CUDA_ERROR(cudaMemcpy(h_pop_original_, d_pop_original_, size_pop_ * dim_ * sizeof(double), cudaMemcpyDeviceToHost));
		break;
	case(3) :
		HANDLE_CUDA_ERROR(cudaMemcpy(h_M_, d_M_, dim_ * dim_ * num_comp_func_ * sizeof(double), cudaMemcpyDeviceToHost));
		HANDLE_CUDA_ERROR(cudaMemcpy(h_shift_, d_shift_, MAX_DIM * MAX_NUM_COMP_FUNC * sizeof(double), cudaMemcpyDeviceToHost));
		HANDLE_CUDA_ERROR(cudaMemcpy(h_shuffle_, d_shuffle_, dim_ * MAX_NUM_COMP_FUNC * sizeof(int), cudaMemcpyDeviceToHost));
		break;
	default:
		break;
	}
}
#endif