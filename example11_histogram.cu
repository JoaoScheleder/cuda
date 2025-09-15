#include <cuda_runtime.h>
#include <stdio.h>




__global__ void atomicHistogramKernel(int *Histogram, const int * data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        atomicAdd(&Histogram[data[idx]], 1);
    }
}

__global__ void naiveHistogramKernel(int *Histogram, const int * data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        Histogram[data[idx]] += 1; // This can cause race conditions
    }
}

int main () {

    const int N = 1 << 20; // 1M elements

    int *h_data = (int *)malloc(N * sizeof(int));
    int *h_histogram = (int *)calloc(256, sizeof(int));
    int *d_data, *d_histogram;
    cudaMalloc(&d_data, N * sizeof(int));
    cudaMalloc(&d_histogram, 256 * sizeof(int));

    // Initialize input data
    for (int i = 0; i < N; i++) {
        h_data[i] = rand() % 256;
    }

    cudaMemcpy(d_data, h_data, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_histogram, 0, 256 * sizeof(int));

    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;

    printf("Blocks : %d, Block Size: %d\n", numBlocks, blockSize);
    printf("All threads: %d\n", numBlocks * blockSize);

    atomicHistogramKernel<<<numBlocks, blockSize>>>(d_histogram, d_data, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_histogram, d_histogram, 256 * sizeof(int), cudaMemcpyDeviceToHost);

    // Print histogram
    for (int i = 0; i < 256; i++) {
        printf("Value %d: Count %d\n", i, h_histogram[i]);
    }

    // Free memory
    cudaFree(d_data);
    cudaFree(d_histogram);
    free(h_data);
    free(h_histogram);


    return 0;
}