/*	
 *	Copyright (C) 2011, Federico Raimondo (fraimondo@dc.uba.ar)
 *	
 *	This file is part of Cudaica.
 *
 *  Cudaica is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  any later version.
 *
 *  Cudaica is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 * 
 *  You should have received a copy of the GNU General Public License
 *  along with Cudaica.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef __ERROR_H__
#define __ERROR_H__
#include <errno.h>

#define STR_EXPAND(tok) #tok
#define xstr(tok) STR_EXPAND(tok)

#define SUCCESS 0

#define ERRORNODEVICEMEM 	-1				//No more memory on device
#define ERRORNOPARAM 		-2				//Parameter error
#define ERRORINVALIDPARAM	-3				//Parameter is invalid
#define ERRORINVALIDCONFIG	-4				//Config file is invalid
#define ERRORNOFILE			-5				//Error opening file
#define ERRORNOMEM				-6				//No more memory on host

//display error information
#define HANDLE_CUDA_ERROR( err ) (HandleCudaError( err, __FILE__, __LINE__ ))
#define CHECK_CUDA_ERROR() (HandleCudaError(cudaGetLastError(), __FILE__, __LINE__))

#define PRINT_ERR(err, err2) \
		if ((err) == (err2)) { \
			fprintf(stderr, "ERROR::%s (%x) in %s at line %d\n", xstr(err2), (err), __FILE__, __LINE__ ); \
		}
/*HANDLE_CUBLAS_ERROR and HANDLE_CURAND_ERROR is not used in this program*/
#define HANDLE_CUBLAS_ERROR( errn ) \
	if ((errn) != 0) {\
		PRINT_ERR(errn, CUBLAS_STATUS_NOT_INITIALIZED) \
		else PRINT_ERR(errn, CUBLAS_STATUS_ALLOC_FAILED) \
		else PRINT_ERR(errn, CUBLAS_STATUS_INVALID_VALUE) \
		else PRINT_ERR(errn, CUBLAS_STATUS_ARCH_MISMATCH) \
		else PRINT_ERR(errn, CUBLAS_STATUS_MAPPING_ERROR) \
		else PRINT_ERR(errn, CUBLAS_STATUS_EXECUTION_FAILED) \
		else PRINT_ERR(errn, CUBLAS_STATUS_INTERNAL_ERROR) \
		else fprintf(stderr, "ERROR::%s (%x) in %s at line %d\n", "UNKNOWN", errn, __FILE__, __LINE__ ); \
		exit( EXIT_FAILURE ); \
   }
   
   
#define HANDLE_CURAND_ERROR( errn ) \
	if ((errn) != CURAND_STATUS_SUCCESS) {\
		PRINT_ERR(errn, CURAND_STATUS_VERSION_MISMATCH) \
		else PRINT_ERR(errn, CURAND_STATUS_NOT_INITIALIZED) \
		else PRINT_ERR(errn, CURAND_STATUS_ALLOCATION_FAILED) \
		else PRINT_ERR(errn, CURAND_STATUS_TYPE_ERROR) \
		else PRINT_ERR(errn, CURAND_STATUS_OUT_OF_RANGE) \
		else PRINT_ERR(errn, CURAND_STATUS_LENGTH_NOT_MULTIPLE) \
		else PRINT_ERR(errn, CURAND_STATUS_LAUNCH_FAILURE) \
		else PRINT_ERR(errn, CURAND_STATUS_PREEXISTING_FAILURE) \
		else PRINT_ERR(errn, CURAND_STATUS_INITIALIZATION_FAILED) \
		else PRINT_ERR(errn, CURAND_STATUS_ARCH_MISMATCH) \
		else PRINT_ERR(errn, CURAND_STATUS_INTERNAL_ERROR) \
		else fprintf(stderr, "ERROR::%s (%x) in %s at line %d\n", "UNKNOWN", errn, __FILE__, __LINE__ ); \
		exit( EXIT_FAILURE ); \
   }


#ifdef __cplusplus
extern "C" {
#endif
typedef int error;
void ResetError();

#ifdef __cplusplus
}
#endif

#endif
