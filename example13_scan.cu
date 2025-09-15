#include <cuda_runtime.h>
#include <stdio.h>

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

    // Up-sweep (reduce) phase
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

__global__ void globalScanKernel(int *input, int *output, int N)
{
    // This kernel would implement a global scan using atomic operations
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
    {
        // Perform an atomic addition to compute the prefix sum
        int val = input[idx];
        int sum = atomicAdd(&output[N], val); // Using output[N] as a temporary storage for the sum
        output[idx] = sum;
    }
    // Note: This is a simplified example and may not be efficient for large arrays.
}

int main()
{
    const int N = 1024;
    int size = N * sizeof(int);
    int h_input[N], h_output[N];

    // Initialize input data
    for (int i = 0; i < N; i++)
    {
        h_input[i] = 1; // Example input
    }

    int *d_input, *d_output;
    cudaMalloc((void **)&d_input, size);
    cudaMalloc((void **)&d_output, size);

    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    float elapsedTime;

    // Launch scan kernel
    int threads = N / 2;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    scanKernel<<<1, threads, N * sizeof(int)>>>(d_input, d_output, N);

    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Time for scanKernel: %f ms\n", elapsedTime);

    // Print output
    for (int i = 0; i < N; i++)
    {
        printf("%d ", h_output[i]);
    }
    printf("\n");

    cudaFree(d_input);
    cudaFree(d_output);
    return 0;
}