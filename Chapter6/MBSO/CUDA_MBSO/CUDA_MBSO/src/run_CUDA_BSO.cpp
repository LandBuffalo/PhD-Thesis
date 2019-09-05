
#include "../include/config.h"
#include "../include/CUDA_BSO.h"
#include "../include/params.h"


/*
*	main file for the project, it includes initialization, run CUDA_BSO, print parameters setting and free memory.
*	main() function has less than 4 Command Line Arguments(-d choose the GPU device, -f choose the input and output
*	files, -r choose the index of runs and -s choose the seed, if it is not specific, use the seed in input file)
*/
//initilize the config(parameters)
void printCapabilities(cudaDeviceProp* properties)
{
	fprintf(stdout, "CUDA Device capabilities:\n");
	fprintf(stdout, "	Name: %s\n", properties->name);
	fprintf(stdout, "	Global Mem: %lu\n", properties->totalGlobalMem);
	fprintf(stdout, "	Mem: %lu\n", properties->totalGlobalMem);
	fprintf(stdout, "	Mem per Block: %lu\n", properties->sharedMemPerBlock);
	fprintf(stdout, "	Regs per Block: %d\n", properties->regsPerBlock);
	fprintf(stdout, "	Warp size: %d\n", properties->warpSize);
	fprintf(stdout, "	Mem pitch: %lu\n", properties->memPitch);
	fprintf(stdout, "	Max Threads per Block: %d\n", properties->maxThreadsPerBlock);
	fprintf(stdout, "	Max Threads Dim: %d x %d x %d\n",
		properties->maxThreadsDim[0],
		properties->maxThreadsDim[1],
		properties->maxThreadsDim[2]);
	fprintf(stdout, "	Max Grid Size: %d x %d x %d\n",
		properties->maxGridSize[0],
		properties->maxGridSize[1],
		properties->maxGridSize[2]);
	fprintf(stdout, "	Total Const Mem: %lu\n", properties->totalConstMem);
	fprintf(stdout, "	Major: %d\n", properties->major);
	fprintf(stdout, "	Minor: %d\n", properties->minor);
	fprintf(stdout, "	Clock Rate: %d\n", properties->clockRate);
	fprintf(stdout, "	Texture Alignment: %lu\n", properties->textureAlignment);
	fprintf(stdout, "	Device Overlap: %d\n", properties->deviceOverlap);
	fprintf(stdout, "	Multiprocessor Count: %d\n", properties->multiProcessorCount);
	fprintf(stdout, "	Kernel Timeout Enabled: %d\n", properties->kernelExecTimeoutEnabled);
	fprintf(stdout, "	Integrated: %d\n", properties->integrated);
	fprintf(stdout, "	Can Map host mem: %d\n", properties->canMapHostMemory);
	fprintf(stdout, "	Compute mode: %d\n", properties->computeMode);
	fprintf(stdout, "	Concurrent kernels: %d\n", properties->concurrentKernels);
	fprintf(stdout, "	ECC Enabled: %d\n", properties->ECCEnabled);
	fprintf(stdout, "	PCI Bus ID: %d\n", properties->pciBusID);
	fprintf(stdout, "	PCI Device ID: %d\n", properties->pciDeviceID);
	fprintf(stdout, "	TCC Driver: %d\n", properties->tccDriver);
}

error selectDevice(int ID_device)
{
	int count_device;
	cudaGetDeviceCount(&count_device);
	cudaDeviceProp prop_device;
	natural numdev = 0;
	fprintf(stdout, "=====================\n");
	fprintf(stdout, "List of cuda devices:\n");
	fprintf(stdout, "=====================\n\n");
	//display all the GPU's CUDA support status 
	for (int num_dev = 0; num_dev < count_device; ++num_dev)
	{
		HANDLE_CUDA_ERROR(cudaGetDeviceProperties(&prop_device, num_dev));
		if (prop_device.major == 9999 && prop_device.minor == 9999)
		{
		}
		else
		{
			printf("Device: %d\n", numdev);
			printCapabilities(&prop_device);
		}
	}

	fprintf(stdout, "\n\nSelecting device %d", ID_device);
	//choose the GOU based on the -d arguments
	HANDLE_CUDA_ERROR(cudaSetDevice(ID_device));
	//gpu.deviceProp includes the selected GPU's properties
	HANDLE_CUDA_ERROR(cudaGetDeviceProperties(&prop_device, ID_device));
	//necessary to call before any CUDA runtime function
	HANDLE_CUDA_ERROR(cudaSetDeviceFlags(cudaDeviceMapHost));

	// Major compute capability 
	//??????????????????the maximum number of threads supported by the device
	fprintf(stdout, " Success!\n");
	fprintf(stdout, "=====================\n\n");
	return SUCCESS;
}

error initConfig(char* filename, double *f_cross, double *f_weight, int *seed, natural ID_func)
{
	parsedconfigs * parsed;
	error err = parseConfig(filename, &parsed);
	if (err != SUCCESS) return err;
	//input file
	//output file
	char file_out[100];
	sprintf(file_out, "input_data/shift_data_%d.txt", ID_func);

	//maxmum number of function evaluation
	if (err != SUCCESS) return err;
	//CR parameter in BSO
	err = getDouble(parsed, "f_cross", f_cross);
	if (err != SUCCESS) return err;
	//F parameter in BSO
	err = getDouble(parsed, "f_weight", f_weight);
	if (err != SUCCESS) return err;
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
	if (isParam("-d", argv, argc))
		ID_device = atoi(getParam("-d", argv, argc));
	else
		ID_device = 0;
	selectDevice(ID_device);
	error err = SUCCESS;


	if (isParam("-f", argv, argc))
	{
		if (isParam("-func", argv, argc))
		{
			ID_func = atoi(getParam("-func", argv, argc));
		}

		char * filename = getParam("-f", argv, argc);

		initConfig(filename, &fcross, &fweight, &seed, ID_func);

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
	CUDA_BSO cuda_BSO(ID_device, ID_func, run, size_pop, dim, seed, num_cluster, 0.2, 0.8, 0.4, 0.5, 0.005);
#ifdef IMPORT_RAND
	cuda_BSO.RandFileToHost("unif_random.txt", "norm_random.txt");
#endif
	cuda_BSO.BSO();
		//print the config
	return 0;
}
