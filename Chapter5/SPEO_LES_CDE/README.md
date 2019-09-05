# SPEO-HPC-GPU

This code is the implementation of LES-CDE based on SPEO-CPU on CPU-only HPC (Chapter 5 of Chen Jin's thesis).

## Getting Started

This code works on CPU-only distributed computing system (HPC). At least 2 CPU processors are required to successfully execute this code. The computing part is implemented based on C/C++, and the information exchange part is implemented based on MPI. 

### Prerequisites

You need to install below software before you build and execute the code. 

```
gcc:5.5.0
openmpi:3.0.0
```
The above versions are tested. Other versions of above software may also work on our code, but it is not guaranteed

## Running the code
Before running the code, users need to decide the parameters of SPEO. Please see the details as follows:

-dim: the problem dimension, currently it support 10, 30, 50, 100
-total_functions: i-j, where i and j are index of test functions of CEC2014 benchmarks. 1<=i<=j<=30
-total_runs: i-j, where i and j are index of repeated run instance. 1<=i<=j
-max_base_FEs: the maximal base fitness evaluations (FEs) which stop the SPEO. This parameter does not change based on dimension. So the really total FEs are max_base_FEs*dimension based on policy of CEC2014 benchmark
-interval: the migration interval (I) defined by SPEO 
-connection_rate: the connection rate (Rc) defined by SPEO 
-buffer_capacity: the buffer capacity rate (Cb) defined by SPEO  
-global_pop_size: the global population size of SPEO-LES-CDE, the island size is global_pop_size/#cores
-k_nearest: the number of k-nearest members

The shift, rotation and shuffle data is located at ./bin/input/ fold. It is provide by CEC2014 official website

If run on HPC with job submission system (Ozstar using slum system) firstly enter the ./bin fold. Then set the number of processors at the second line of SPEO.sh file: #SBATCH --ntasks=x, where x is number of CPU cores. Then run "sbatch SPEO_LES_CDE.sh". The parameters can be changed in the file of SPEO_LES_CDE.sh

## Output
The output of slum system is at the ./bin like slum-xxxxxxx.out, where xxxxxxx is the job id.

The output of algorithm is at ./bin/Results/. The file is .csv which list the computing time, optimization accuracy and parameters you used. Each line is a independent run instance.

## Authors

* **Chen Jin** - *Initial work*

## License

This project is licensed under the GPL License - see the [LICENSE.md]file for details

## Acknowledgments
