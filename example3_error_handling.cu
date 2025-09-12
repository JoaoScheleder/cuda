#include  "cuda_runtime.h"
#include <stdio.h>


int main () {
    int *a = 0;
    int deviceCount = 0;
    cudaError_t err = cudaSuccess;
    
    err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to get device count (error code %s)!\n", cudaGetErrorString(err));
        return -1;
    }

    err = cudaMalloc((void**)&a, 10 * sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device memory (error code %s)!\n", cudaGetErrorString(err));
        return -1;
    }

    if (deviceCount == 0) {
        fprintf(stderr, "No CUDA devices found!\n");
        return -1;
    }

    printf("Number of CUDA devices: %d\n", deviceCount);

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp deviceProp;
        err = cudaGetDeviceProperties(&deviceProp, dev);
        if (err != cudaSuccess) {
            fprintf(stderr, "Failed to get device properties for device %d (error code %s)!\n", dev, cudaGetErrorString(err));
            continue;
        }
        printf("Device %d: %s\n", dev, deviceProp.name);
        printf("  Total Global Memory: %zu bytes\n", deviceProp.totalGlobalMem);
        printf("  Shared Memory per Block: %zu bytes\n", deviceProp.sharedMemPerBlock);
        printf("  Registers per Block: %d\n", deviceProp.regsPerBlock);
        printf("  Warp Size: %d\n", deviceProp.warpSize);
        printf("  Max Threads per Block: %d\n", deviceProp.maxThreadsPerBlock);
        printf("  Compute Capability: %d.%d\n", deviceProp.major, deviceProp.minor);
    }

    return 0;
}