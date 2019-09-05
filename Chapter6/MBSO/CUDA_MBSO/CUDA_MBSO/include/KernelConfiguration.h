#ifndef __KERNELCONFIGURATION_H__
#define __KERNELCONFIGURATION_H__

#include <stdio.h>
#include "../include/config.h"
#include "../include/error.h"

class Device 
{
private:
	natural 		ID_device_;
	cudaDeviceProp	prop_device_;
//	size_t 			mem_current_free_;
//	size_t 			mem_needed_reserved_;

public:
					Device(int ID_device);
					~Device();
	natural			getMaxSharedMem();
	natural			getWarpSize();
	natural			getMaxThreadsPerBlock();
	natural			getRegsPerBlock();
} ;



class KernelConfiguration
{
private:
	int				needed;

	natural			dim_;
	natural			size_pop_;

	Device *		device_;

	natural			memPerInd();
public:
	dim3 			threads_;
	dim3 			blocks_;

					KernelConfiguration(natural size_pop, natural dim, int ID_device);
					~KernelConfiguration();

	error			CalKernelConfiguration();
};

#endif
