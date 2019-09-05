
#include "../include/config.h"
#include "../include/CPU_BSO.h"
#include "../include/params.h"


/*
*	main file for the project, it includes initialization, run CUDA_BSO, print parameters setting and free memory.
*	main() function has less than 4 Command Line Arguments(-d choose the GPU device, -f choose the input and output
*	files, -r choose the index of runs and -s choose the seed, if it is not specific, use the seed in input file)
*/
//initilize the config(parameters)


error initConfig(char* filename, int *seed)
{
	parsedconfigs * parsed;
	error err = parseConfig(filename, &parsed);
	if (err != SUCCESS) return err;
	//input file
	//output file

	//seed of random generation
	err = getInt(parsed, "i_seed", seed);
	if (err != SUCCESS) return err;

	freeConfig(parsed);
	return SUCCESS;
}


int main(int argc, char* argv[])
{
	natural run, size_pop, dim, ID_func;
	int seed, ID_device, num_cluster;
	double fcross, fweight;
	printf("Starting Cuda BSO with floating point numbers size = %lu bytes\n", sizeof(double));


	printf("=========================\n");
	error err = SUCCESS;

	if (isParam("-f", argv, argc))
	{
		if (isParam("-func", argv, argc))
		{
			ID_func = atoi(getParam("-func", argv, argc));
		}

		char * filename = getParam("-f", argv, argc);

		initConfig(filename, &seed);

		printf(" Config File %s \n", filename);
		printf("===============================================================================\n");
		printf("%s=%d\t", "function:", ID_func);

		if (isParam("-s", argv, argc))
		{
			seed = atoi(getParam("-s", argv, argc));
		}

		if (isParam("-id", argv, argc))
		{
			dim = atoi(getParam("-id", argv, argc));
			printf("%s=%d\t", "dim", dim);

		}
		if (isParam("-ip", argv, argc))
		{
			size_pop = atoi(getParam("-ip", argv, argc));
			printf("%s=%d\t", "size_pop", size_pop);
		}
		if (isParam("-r", argv, argc))
		{
			run = atoi(getParam("-r", argv, argc));
			printf("%s=%d\t", "run", run);
			printf("%s=%d\t", "seed", seed);
		}

		if (isParam("-c", argv, argc))
		{
			num_cluster = atoi(getParam("-c", argv, argc));
			printf("%s=%d\n", "num_cluster", num_cluster);
		}
		printf("===============================================================================\n");

	}
	CPU_BSO cuda_BSO(ID_func, run, size_pop, dim, seed, num_cluster, 0.2, 0.8, 0.4, 0.5, 0.005);
#ifdef IMPORT_RAND
	cuda_BSO.RandFileToHost("random_number_unif.txt", "random_number_norm.txt");
#endif
	cuda_BSO.BSO();
		//print the config
	return 0;
}
