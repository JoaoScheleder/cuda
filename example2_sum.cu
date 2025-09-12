#include  "cuda_runtime.h"
#include <stdio.h>

// malloc == cudaMalloc // Allocate device memory
// free == cudaFree // Free device memory
// memcpy == cudaMemcpy // Send from host (RAM) to device (GPU memory)

__global__ void add(int *a, int *b, int *c) {
    *c = *a + *b;
}

int main() {
    int a, b, c;
    int *d_a, *d_b, *d_c;
    int size = 10 * sizeof(int); // bytes

    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_b, size);
    cudaMalloc((void**)&d_c, size);

    a = 5;
    b = 7;


    // CPU -> GPU
    cudaMemcpy(d_a, &a, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, &b, sizeof(int), cudaMemcpyHostToDevice);


    add<<<1,1>>>(d_a, d_b, d_c);

    // GPU -> CPU
    cudaMemcpy(&c, d_c, sizeof(int), cudaMemcpyDeviceToHost); // Already calls cudaDeviceSynchronize() internally
    
    printf("Result of %d + %d = %d \n", a, b, c);

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    
    return 0;
}