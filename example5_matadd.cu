#include "cuda_runtime.h"
#include <stdio.h>

__global__ void matAdd(const int *d_a, const int *d_b, int *d_c, int rows, int cols) {
    // Correct global row and column calculation
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Stride to cover larger matrices
    int strideRow = blockDim.y * gridDim.y;
    int strideCol = blockDim.x * gridDim.x;

    for (int i = row; i < rows; i += strideRow) {
        for (int j = col; j < cols; j += strideCol) {
            d_c[i * cols + j] = d_a[i * cols + j] + d_b[i * cols + j];
        }
    }
}

int main() {
    int rows = 4;
    int cols = 6;

    // Allocate host matrices
    int *h_a = (int*)malloc(rows * cols * sizeof(int));
    int *h_b = (int*)malloc(rows * cols * sizeof(int));
    int *h_c = (int*)malloc(rows * cols * sizeof(int));

    // Initialize host matrices
    for (int i = 0; i < rows * cols; i++) {
        h_a[i] = i;
        h_b[i] = i;
    }

    // Allocate device matrices
    int *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, rows * cols * sizeof(int));
    cudaMalloc(&d_b, rows * cols * sizeof(int));
    cudaMalloc(&d_c, rows * cols * sizeof(int));

    cudaMemcpy(d_a, h_a, rows * cols * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, rows * cols * sizeof(int), cudaMemcpyHostToDevice);

    // Define block and grid sizes
    dim3 block(16, 16);  // 16x16 threads per block
    dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);

    printf("Grid dimensions: (%d, %d), Block dimensions: (%d, %d)\n", grid.x, grid.y, block.x, block.y);

    // Launch kernel
    matAdd<<<grid, block>>>(d_a, d_b, d_c, rows, cols);

    cudaMemcpy(h_c, d_c, rows * cols * sizeof(int), cudaMemcpyDeviceToHost);

    // Print result
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%d ", h_c[i * cols + j]);
        }
        printf("\n");
    }

    // Free memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}
