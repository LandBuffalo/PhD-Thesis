#ifndef __CEC2014_H__
#define __CEC2014_H__

#include "../include/config.h"
#include "../include/KernelConfiguration.h"

#define MAX_NUM_COMP_FUNC 5
extern "C" void API_rotation(double * d_pop_rotated, double * d_pop_original, double * d_M, double * d_shift, double shift, double rate_weighted, dim3 blocks, dim3 threads, natural size_pop, natural dim, natural num_comp_func);
extern "C" void API_evaluateFitness(double * d_fval, double * d_pop_original, double * d_pop_rotated, int * d_shuffle, double * d_shift, dim3 blocks, dim3 threads, natural ID_func_, natural size_pop, natural dim);

class  CEC2014
{
private:
	int						sigma_[MAX_NUM_COMP_FUNC];
	int						is_shifted_[MAX_NUM_COMP_FUNC];
	int						is_rotated_[MAX_NUM_COMP_FUNC];
	int						max_feval_base_;
	int						max_feval_;
	bool					flag_composition_;

	natural					ID_func_;
	natural					num_comp_func_;
	natural					size_pop_;
	natural					dim_;

	double					rate_weighted_;
	double					shift_;
	double					bias_;
	double					minbound_;
	double					maxbound_;

	int *					d_shuffle_;
	double *				d_M_;
	double *				d_shift_;

	double *				d_fval_;
	double *				d_pop_rotated_;
	double *				d_pop_original_;

	KernelConfiguration *	kernel_configuration_;
#ifdef DEBUG
	int *					h_shuffle_;
	double *				h_shift_;
	double *				h_M_;
	double *				h_fval_;
	double *				h_pop_rotated_;
	double *				h_pop_original_;
#endif
	void					ShiftRotate();
	void					CalConfigCEC2014();
public:
							CEC2014(natural ID_func, natural size_pop, natural dim, int ID_device);
							~CEC2014();

	void					set_d_pop_original_d_fval(double * d_pop_original, double * d_fval);
	void					set_max_feval();

	natural					ID_func();
	int						max_feval();
	double					minbound();
	double					maxbound();

	void					EvaluateFitness(double * d_fval, double * d_pop);
	error					LoadData();

#ifdef DEBUG
	void					Check(int debug_level);
#endif
};

#endif