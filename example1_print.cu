#include "cuda_runtime.h"
#include <stdio.h>

__global__ void myKernel(void)
{
    printf("Hello from GPU thread %d!\n", threadIdx.x);
}

int main()
{
    // blocks, threads (max threads per block = 1024)
    myKernel<<<1, 10>>>();
    cudaDeviceSynchronize();
    printf("Hello from CPU \n");
    return 0;
}