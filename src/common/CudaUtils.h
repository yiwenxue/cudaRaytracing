#pragma once

#include <cuda.h>
#include <iostream>
#include <stdio.h>

#define cudaCheckErrors(msg)                                                                   \
    do                                                                                         \
    {                                                                                          \
        cudaError_t __err = cudaGetLastError();                                                \
        if (__err != cudaSuccess)                                                              \
        {                                                                                      \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", msg, cudaGetErrorString(__err), \
                    __FILE__, __LINE__);                                                       \
            fprintf(stderr, "*** FAILED - ABORTING\n");                                        \
            exit(1);                                                                           \
        }                                                                                      \
    } while (0)

#define checkCudaErrors(val) check_cuda((val), #val, __FILE__, __LINE__)

void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line)
{
    if (result)
    {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " << file << ":"
                  << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}
