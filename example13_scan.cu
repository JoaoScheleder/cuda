#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

// blelloch scan kernel
__global__ void scanKernel(int *input, int *output, int N)
{
    extern __shared__ int temp[]; // Shared memory for scan operation
    int thid = threadIdx.x;
    int offset = 1;

    // Load input into shared memory
    if (2 * thid < N)
        temp[2 * thid] = input[2 * thid];
    else
        temp[2 * thid] = 0;
    if (2 * thid + 1 < N)
        temp[2 * thid + 1] = input[2 * thid + 1];
    else
        temp[2 * thid + 1] = 0;

    for (int d = N >> 1; d > 0; d >>= 1)
    {
        __syncthreads();
        if (thid < d)
        {
            int ai = offset * (2 * thid + 1) - 1;
            int bi = offset * (2 * thid + 2) - 1;
            temp[bi] += temp[ai];
        }
        offset *= 2;
    }

    // Clear the last element
    if (thid == 0)
    {
        temp[N - 1] = 0;
    }

    // Down-sweep phase
    for (int d = 1; d < N; d *= 2)
    {
        offset >>= 1;
        __syncthreads();
        if (thid < d)
        {
            int ai = offset * (2 * thid + 1) - 1;
            int bi = offset * (2 * thid + 2) - 1;
            int t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }
    __syncthreads();

    // Write results to output array
    if (2 * thid < N)
        output[2 * thid] = temp[2 * thid];
    if (2 * thid + 1 < N)
        output[2 * thid + 1] = temp[2 * thid + 1];
}

// Simple scan kernel for verification
__global__ void scanKernelSimple(const int *input, int *output, int N)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        int sum = 0;
        for (int i = 0; i < idx; i++) { // compute prefix sum directly
            sum += input[i];
        }
        output[idx] = sum; // exclusive scan
    }
}

int main()
{
    cudaDeviceReset();

    const int N = 1024;

    // const int N = 1 << 20; // 1M elements for simple scan

    size_t size = N * sizeof(int);

    int *h_input  = (int*)malloc(size);
    int *h_output = (int*)malloc(size);

    if (!h_input || !h_output) {
        printf("Host malloc failed!\n");
        return -1;
    }

    // Initialize input
    for (int i = 0; i < N; i++)
        h_input[i] = 1;

    int *d_input, *d_output;
    cudaError_t err;

    err = cudaMalloc((void**)&d_input, size);
    if (err != cudaSuccess) { printf("cudaMalloc d_input failed: %s\n", cudaGetErrorString(err)); return -1; }

    err = cudaMalloc((void**)&d_output, size);
    if (err != cudaSuccess) { printf("cudaMalloc d_output failed: %s\n", cudaGetErrorString(err)); return -1; }

    err = cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { printf("cudaMemcpy H2D failed: %s\n", cudaGetErrorString(err)); return -1; }

    dim3 blockSize(N/2);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    scanKernelSimple<<<gridSize, blockSize>>>(d_input, d_output, N);
    // scanKernel<<<gridSize, blockSize, 2 * blockSize.x * sizeof(int)>>>(d_input, d_output, N);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Scan kernel execution time: %3.3f ms\n", milliseconds);

    err = cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { printf("cudaMemcpy D2H failed: %s\n", cudaGetErrorString(err)); return -1; }

    printf("Last 10 elements of output:\n");
    for (int i = N-10; i < N; i++)
        printf("%d ", h_output[i]);
    printf("\n");

    // Free memory
    free(h_input);
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}