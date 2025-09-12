#include  "cuda_runtime.h"
#include <stdio.h>

__global__ void vecAdd(const int *d_a, const int *d_b, int *d_c, int N) {
    // int idx = threadIdx.x; // Local Index, could lead to multiple threads processing the same element if N > blockDim.x
    // int stride = blockDim.x; // Local Stride, could lead to multiple threads processing the same element if N > blockDim.x

    int idx = blockIdx.x * blockDim.x + threadIdx.x;   // índice global // Avoid using global idx directly to prevent out-of-bounds access
    int stride = blockDim.x * gridDim.x;              // salto global entre threads // Total number of threads in the grid

    // idx 0 will process elements 0, stride, 2*stride, ...
    for (int i = idx; i < N; i += stride) {
        printf("Thread ID: %d, Stride: %d\n, Processing element: %d\n", idx, stride, i);
        // if(i == 0){
        //     printf("Thread ID: %d, Stride: %d\n, Processing element: %d\n", idx, stride, i);
        // }
        d_c[i] = d_a[i] + d_b[i];
    }
}

int main () {
    cudaDeviceReset();
    int N = 64;
    const int size = N * sizeof(int);
    
    int threadsPerBlock = 32;
    int blocksPerGrid = 1;
    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
    printf("Total threads: %d\n", blocksPerGrid * threadsPerBlock);
    int *h_a, *h_b, *h_c;
    int *d_a, *d_b, *d_c;

    h_a = (int*)malloc(size);
    h_b = (int*)malloc(size);
    h_c = (int*)malloc(size);
    
    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_b, size);
    cudaMalloc((void**)&d_c, size);
    
    for (int i = 0; i < N; i++) {
        h_a[i] = i;
        h_b[i] = i;
    }

    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);
    

    vecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);

    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    printf("h_a[10]: %d\n", h_a[63]);
    printf("h_b[10]: %d\n", h_b[63]);
    printf("Result: %d\n", h_c[63]);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}