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

#include <stdio.h>
#include <stdio.h>
#include "cuda_runtime.h"
#include "../include/error.h"

/*
 * Reset the errors waiting for being fetched.
 */ 



/*
 * Handles an error
 */ 
void HandleCudaError( cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) {
        fprintf(stderr, "ERROR::%s (%x) in %s at line %d\n", cudaGetErrorString( err ), err,
                file, line );
        cudaError_t newerr = cudaGetLastError();
        if (newerr != err && newerr != cudaSuccess) {
		}
        exit( EXIT_FAILURE );
    }
}
