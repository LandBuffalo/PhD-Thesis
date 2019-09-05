#include "../include/KernelConfiguration.h"

Device::Device(int ID_device)
{
	ID_device_ = ID_device;
	HANDLE_CUDA_ERROR(cudaGetDeviceProperties(&prop_device_, ID_device_));
}

Device::~Device()
{

}

natural Device::getMaxThreadsPerBlock()
{
	return prop_device_.maxThreadsPerBlock;
}

natural Device::getRegsPerBlock()
{
	return prop_device_.regsPerBlock;
}

natural Device::getMaxSharedMem()
{
	return prop_device_.sharedMemPerBlock;
}

natural Device::getWarpSize()
{
	return prop_device_.warpSize;
}


KernelConfiguration::KernelConfiguration(natural size_pop, natural dim, int ID_device)
{

	dim_ = dim;
	size_pop_ = size_pop;
	device_ = new Device(ID_device);
}

KernelConfiguration::~KernelConfiguration()
{

}

natural KernelConfiguration::memPerInd() {
#ifdef COPYTOSHARED
	/* Depends on strategy */
	natural memperind = dim * 6 * sizeof(real) + sizeof(int);
#else
	/* Not copying permutation on shared memory */
	natural memperind = sizeof(double) + 1 * dim_ * sizeof(double) + sizeof(int); //Result + X + J
#endif

	memperind += 0;//neededSharedMemPerInd(config->blocks, config->threads); 

	return memperind;
}

error KernelConfiguration::CalKernelConfiguration()
{
	natural memperind = memPerInd();

	natural mem = device_->getMaxSharedMem(); //Shared memory - bounds
#if STRATEGY == 3
	mem -= dim * sizeof(real);
#endif
	//the max number of threads per block according to the share memory allocation
	natural maxInds = mem / memperind;
	//	natural rotMemIndsXY = mem / rotMemperind;
	cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
	//if the max number of thread is large the warp size, then assign the max number of threads as the warp size
	//???it means that one block must not have more than one warp
	if (maxInds > device_->getWarpSize()) {
		maxInds = device_->getWarpSize();
	}
	//if (rotMemIndsXY > getWarpSize()) {
	//	rotMemIndsXY = getWarpSize();
	//}
	//due to the whole individual(all dimensions) should be in one block to use share memory, the max number of threads is also decided by the dim 
	natural maxthreadsbymem = device_->getMaxThreadsPerBlock() / dim_;
	//???what does this mean
	natural maxthreadsbyregs = device_->getRegsPerBlock() / MAX_REGISTERS / dim_;
	//choose the small number based on memory and ragister as the max number of thread per block.
	natural maxthreads = maxthreadsbyregs < maxthreadsbymem ? maxthreadsbyregs : maxthreadsbymem;
	//	natural rotThreads = rotMemIndsXY < maxthreadsbymem ? rotMemIndsXY : maxthreadsbymem;
	//choose the small number as the max number of thread per block.
	natural threads = maxInds < maxthreads ? maxInds : maxthreads;
	//	natural rotThreads = rotMemIndsXY < rotMaxthreads ? rotMemIndsXY : rotMaxthreads;
	//~ if (threads == 0) threads = 1;
	//"size" is the polulation size, to keep all the block has same threads considering the population size
	while (size_pop_ % threads != 0) {
		threads--;
	}
	natural blocks = size_pop_ / threads;
	//finally decide the dim of block and grid
	if (threads * dim_ / blocks > 100)
		threads = 5;
	threads_ = dim3(threads, dim_);
	blocks_ = size_pop_ / threads;


	return SUCCESS;
}
