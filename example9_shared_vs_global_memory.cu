#include <cuda_runtime.h>
#include <stdio.h>

// Naive global-memory 1D 3-point stencil
__global__ void stencilGlobal(const float *in, float *out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= 1 && idx < N-1) {
        // Each thread reads its neighbors from global memory
        float left  = in[idx - 1];
        float mid   = in[idx];
        float right = in[idx + 1];
        out[idx] = (left + mid + right) / 3.0f;
    }
}

// Shared-memory optimized 1D 3-point stencil
__global__ void stencilShared(const float *in, float *out, int N) {
    // Allocate shared memory for the block
    extern __shared__ float s_in[];

    int tid = threadIdx.x; // unique thread index in the block
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Each block loads a tile of input with halo into shared memory
    // We need blockDim.x + 2 elements for the halo
    if (idx < N) {
        s_in[tid+1] = in[idx]; // main element
        if (tid == 0 && idx > 0)
            s_in[0] = in[idx-1]; // left halo
        if (tid == blockDim.x-1 && idx < N-1)
            s_in[blockDim.x+1] = in[idx+1]; // right halo
    }

    __syncthreads();
    // after sync, all threads can access shared memory
    // wee will have s_in to access left, mid, right in all threads

    if (idx >= 1 && idx < N-1) {
        float left  = s_in[tid];     // left neighbor
        float mid   = s_in[tid+1];   // self
        float right = s_in[tid+2];   // right neighbor
        out[idx] = (left + mid + right) / 3.0f;
    }
}

int main() {
    const int N = 1 << 20; // 1M elements
    size_t size = N * sizeof(float);
    float *h_in  = (float*)malloc(size);
    float *h_out = (float*)malloc(size);

    for (int i = 0; i < N; i++) {
        h_in[i] = (float)i;
    }

    float *d_in, *d_out;
    cudaMalloc(&d_in, size);
    cudaMalloc(&d_out, size);
    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // --- Global memory kernel
    cudaEventRecord(start, 0);
    stencilGlobal<<<numBlocks, blockSize>>>(d_in, d_out, N);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float timeGlobal;
    cudaEventElapsedTime(&timeGlobal, start, stop);

    cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);
    printf("Global kernel time: %3.3f ms\n", timeGlobal);
    printf("Sample output: %f\n", h_out[1000]);

    // --- Shared memory kernel
    size_t shmemSize = (blockSize + 2) * sizeof(float);
    cudaEventRecord(start, 0);
    stencilShared<<<numBlocks, blockSize, shmemSize>>>(d_in, d_out, N);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float timeShared;
    cudaEventElapsedTime(&timeShared, start, stop);

    cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);
    printf("Shared kernel time: %3.3f ms\n", timeShared);
    printf("Sample output: %f\n", h_out[1000]);

    cudaFree(d_in);
    cudaFree(d_out);
    free(h_in);
    free(h_out);
    return 0;
}
