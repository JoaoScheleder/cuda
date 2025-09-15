#include <cuda_runtime.h>
#include <stdio.h>

__global__ void parallelReduceKernel(const float *input, float *output, int N)
{
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Load input into shared memory
    sdata[tid] = (i < N) ? input[i] : 0.0f;
    __syncthreads();

    // Perform reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write result for this block to global memory
    if (tid == 0)
    {
        output[blockIdx.x] = sdata[0];
    }
}

// Alternative kernel using atomic operations in global memory
// The performance is generally worse due to contention on global memory
__global__ void parallelReduceGlobalMemoryKernel(const float *input, float *output, int N)
{
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Perform reduction directly in global memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s && (i + s) < N)
        {
            atomicAdd(&output[blockIdx.x], input[i + s]);
        }
        __syncthreads();
    }
}

int main()
{

    const int N = 1 << 20; // 1M elements
    const int blockSize = 256;
    const int numBlocks = (N + blockSize - 1) / blockSize;

    float *h_input = (float *)malloc(N * sizeof(float));
    float *h_intermediate = (float *)malloc(numBlocks * sizeof(float));
    float *h_output = (float *)malloc(sizeof(float));

    // Initialize input data
    for (int i = 0; i < N; i++)
    {
        h_input[i] = 1.0f; // For simplicity, all elements are 1.0
    }

    float *d_input, *d_intermediate, *d_output;
    cudaMalloc((void **)&d_input, N * sizeof(float));
    cudaMalloc((void **)&d_intermediate, numBlocks * sizeof(float));
    cudaMalloc((void **)&d_output, sizeof(float));

    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);

    // First reduction step
    parallelReduceKernel<<<numBlocks, blockSize, blockSize * sizeof(float)>>>(d_input, d_intermediate, N);

    // Second reduction step (if needed)
    parallelReduceKernel<<<1, blockSize, blockSize * sizeof(float)>>>(d_intermediate, d_output, numBlocks);

    cudaMemcpy(h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost);

    printf("Sum: %f\n", h_output[0]);

    // Free memory
    free(h_input);
    free(h_intermediate);
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_intermediate);
    cudaFree(d_output);

    return 0;
}