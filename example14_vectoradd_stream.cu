#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

__global__ void vectorAddShared(const float *A, const float *B, float *C, int N)
{
    extern __shared__ float sharedMem[];
    float *sA = sharedMem;
    float *sB = &sharedMem[blockDim.x];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load data into shared memory
    if (idx < N) {
        sA[threadIdx.x] = A[idx];
        sB[threadIdx.x] = B[idx];
    } else {
        sA[threadIdx.x] = 0.0f;
        sB[threadIdx.x] = 0.0f;
    }
    __syncthreads();

    // Perform vector addition
    if (idx < N) {
        C[idx] = sA[threadIdx.x] + sB[threadIdx.x];
    }
}

int main () {
    const int N = 1 << 20; // 1M elements
    size_t size = N * sizeof(float);

    
    float *h_a, *h_b, *h_c;

    float *d_a0, *d_b0, *d_c0;
    float *d_a1, *d_b1, *d_c1;

    cudaStream_t stream0, stream1;
    cudaStreamCreate(&stream0);
    cudaStreamCreate(&stream1);

    cudaHostAlloc((void**)&h_a, size, cudaHostAllocDefault);
    cudaHostAlloc((void**)&h_b, size, cudaHostAllocDefault);
    cudaHostAlloc((void**)&h_c, size, cudaHostAllocDefault);

    for (int i = 0; i < N; i++) {
        h_a[i] = static_cast<float>(i);
        h_b[i] = static_cast<float>(i * 2);
    }

    cudaMalloc((void**)&d_a0, size);
    cudaMalloc((void**)&d_b0, size);
    cudaMalloc((void**)&d_c0, size);

    cudaMalloc((void**)&d_a1, size);
    cudaMalloc((void**)&d_b1, size);
    cudaMalloc((void**)&d_c1, size);

    int blockSize = 256;
    int gridSize = ceil((float)N / blockSize);

    int sharedMemSize = 2 * blockSize * sizeof(float);

    cudaMemcpyAsync(d_a0, h_a, size, cudaMemcpyHostToDevice, stream0);
    cudaMemcpyAsync(d_b0, h_b, size, cudaMemcpyHostToDevice, stream0);
    vectorAddShared<<<gridSize, blockSize, sharedMemSize, stream0>>>(d_a0, d_b0, d_c0, N);
    
    cudaMemcpyAsync(h_c, d_c0, size, cudaMemcpyDeviceToHost, stream0);
    
    cudaMemcpyAsync(d_a1, h_a, size, cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(d_b1, h_b, size, cudaMemcpyHostToDevice, stream1);
    vectorAddShared<<<gridSize, blockSize, sharedMemSize, stream1>>>(d_a1, d_b1, d_c1, N);
    
    cudaMemcpyAsync(h_c, d_c1, size, cudaMemcpyDeviceToHost, stream1);


    cudaStreamSynchronize(stream0);
    cudaStreamSynchronize(stream1);


    cudaFree(d_a0);
    cudaFree(d_b0);
    cudaFree(d_c0);

    cudaFree(d_a1);
    cudaFree(d_b1);
    cudaFree(d_c1);

    cudaFreeHost(h_a);
    cudaFreeHost(h_b);
    cudaFreeHost(h_c);

    cudaStreamDestroy(stream0);
    cudaStreamDestroy(stream1);
  
    return 0;  
};
