#ifndef __ELM_H__
#define __ELM_H__
#include <../Eigen/Dense>
#include "config.h"
#include "random.h"
#define E_2 2.71828182845904523
using namespace Eigen;
using namespace std;

struct Model
{
	MatrixXd input_weight;
	MatrixXd out_weight;
	MatrixXd M;
	MatrixXd bias;
	int L;
};

class ELM
{
private:
	ProblemInfo problem_info_;
	IslandInfo island_info_;
	Random random_;
	Model model_;
	Population train_data_pool_;
    int flag_initialized_;
	MatrixXd CalHValue(Population & train_data);
	real_float ActiveFunction(real_float input);
    int InitialiseELM();
    int OnlineTrain();
public:
	ELM();
	~ELM();
	int Initialize(IslandInfo island_info, ProblemInfo problem_info);
    int RecvData(Individual & incoming_individual_data);
    int Train();
    real_float Predict(Individual & individual);
};
#endif