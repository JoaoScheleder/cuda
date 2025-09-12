#include  "cuda_runtime.h"
#include <stdio.h>

__global__ void vecAdd(const int *d_a, const int *d_b, int *d_c) {
    int i = threadIdx.x;
    d_c[i] = d_a[i] + d_b[i];
}

int main () {
    cudaDeviceReset();
    const int N = 256;
    const int size = N * sizeof(int);

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
    

    vecAdd<<<1, N>>>(d_a, d_b, d_c);

    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    printf("h_a[10]: %d\n", h_a[10]);
    printf("h_b[10]: %d\n", h_b[10]);
    printf("Result: %d\n", h_c[10]);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}